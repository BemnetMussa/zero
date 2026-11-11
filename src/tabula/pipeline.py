"""Durable self-play and training pipeline for Tabula Zero."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from tabula.evaluate.arena import Arena
from tabula.evaluate.mate_exam import run_mate_exam
from tabula.runtime import (
    RunPaths,
    WallClockBudget,
    atomic_jsonl_append,
    atomic_torch_save,
    checkpoint_payload,
    latest_checkpoint,
    load_checkpoint,
    peak_cuda_memory_mib,
    require_syzygy,
    utc_now,
)
from tabula.hardware.thermal import ThermalGuard
from tabula.selfplay.worker import DEFAULT_RES_BLOCKS, DEFAULT_SIMULATIONS, SelfPlayWorker
from tabula.train.train import DEFAULT_EPOCHS, TrainResult, train
from tabula.watch.live import LiveBoard

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_LATEST = PROJECT_ROOT / "models" / "tabula_v0.2_latest.pth"
LEGACY_CHAMPION = PROJECT_ROOT / "models" / "tabula_v0.2_champion.pth"


@dataclass
class SelfPlayStats:
    """Counters from one iteration's self-play phase."""

    games_requested: int
    games_attempted: int = 0
    games_completed: int = 0
    game_errors: int = 0
    full_board_games: int = 0
    cap_terminations: int = 0
    practice_games: int = 0
    practice_mates: int = 0
    stalled_rollouts_dropped: int = 0
    replay_examples: int = 0

    @property
    def cap_termination_rate(self) -> float:
        if self.full_board_games == 0:
            return 0.0
        return self.cap_terminations / self.full_board_games


def _log_record(path: Path, event: str, **fields: Any) -> None:
    atomic_jsonl_append(path, {"event": event, "timestamp": utc_now(), **fields})


def _allowed_game_errors(games_requested: int, maximum_rate: float) -> int:
    """Allow the configured fraction, with one transient failure as a floor."""

    return max(1, math.floor(games_requested * maximum_rate))


def run_selfplay_phase(
    worker: SelfPlayWorker,
    games: int,
    iteration: int,
    max_game_error_rate: float,
    metrics_path: Path,
    *,
    live: LiveBoard | None = None,
    thermal: ThermalGuard | None = None,
) -> tuple[list[Any], SelfPlayStats]:
    """Run self-play while tolerating a bounded number of game-level failures."""

    if games <= 0:
        raise ValueError("games must be positive")
    if not 0.0 <= max_game_error_rate < 1.0:
        raise ValueError("max_game_error_rate must be in [0, 1)")

    all_data: list[Any] = []
    stats = SelfPlayStats(games_requested=games)
    allowed_errors = _allowed_game_errors(games, max_game_error_rate)

    for game_index in range(games):
        practice = game_index % 2 == 1
        stats.games_attempted += 1
        if practice:
            stats.practice_games += 1

        try:
            if thermal is not None:
                thermal.check_and_cool()
            data = worker.play_game(live=live, practice=practice)
        except Exception as exc:
            stats.game_errors += 1
            error_traceback = traceback.format_exc()
            LOGGER.warning(
                "Self-play game %d failed; dropping it and continuing:\n%s",
                game_index + 1,
                error_traceback,
            )
            _log_record(
                metrics_path,
                "game_error",
                iteration_idx=iteration,
                game_index=game_index + 1,
                mate_practice=practice,
                error=repr(exc),
                traceback=error_traceback,
            )
            if stats.game_errors > allowed_errors:
                raise RuntimeError(
                    "Self-play error budget exceeded: "
                    f"{stats.game_errors} errors > {allowed_errors} allowed"
                ) from exc
            continue

        stats.games_completed += 1
        termination = worker.last_termination
        if practice:
            if termination == "CHECKMATE":
                stats.practice_mates += 1
            elif not data:
                stats.stalled_rollouts_dropped += 1
        else:
            stats.full_board_games += 1
            if termination == "MOVE_LIMIT":
                stats.cap_terminations += 1

        all_data.extend(data)

    if stats.games_completed == 0:
        raise RuntimeError("All self-play games failed; no training data was produced")
    if stats.game_errors / games > max_game_error_rate:
        raise RuntimeError(
            "Self-play error rate exceeded: "
            f"{stats.game_errors}/{games} > {max_game_error_rate:.0%}"
        )
    if not all_data:
        raise RuntimeError("Self-play completed but produced no replay examples")

    stats.replay_examples = len(all_data)
    return all_data, stats


def _seed_weights(paths: RunPaths, checkpoint: dict[str, Any] | None) -> Path | None:
    """Resolve raw weights for self-play, reconstructing them from a checkpoint."""

    if checkpoint is not None:
        # Raw files are derived artifacts. Rebuild them even when an older one
        # exists, because preemption may have landed after the canonical
        # versioned checkpoint but before its raw-weight mirrors.
        atomic_torch_save(checkpoint["model_state"], paths.latest_weights)
        if checkpoint.get("metadata", {}).get("metrics", {}).get("promoted"):
            atomic_torch_save(checkpoint["model_state"], paths.champion_weights)
        return paths.latest_weights
    for candidate in (paths.latest_weights, LEGACY_LATEST, paths.champion_weights, LEGACY_CHAMPION):
        if candidate.exists():
            return candidate
    return None


def _iteration_metrics(
    iteration: int,
    stats: SelfPlayStats,
    result: TrainResult,
    duration_seconds: float,
    peak_cuda_mib: float,
    arena_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event": "iteration_complete",
        "timestamp": utc_now(),
        "iteration_idx": iteration,
        "policy_loss": result.policy_loss,
        "value_loss": result.value_loss,
        "total_loss": result.total_loss,
        "learning_rate": result.learning_rate,
        "cap_termination_rate": stats.cap_termination_rate,
        "games_completed": stats.games_completed,
        "games_attempted": stats.games_attempted,
        "game_errors": stats.game_errors,
        "stalled_rollouts_dropped": stats.stalled_rollouts_dropped,
        "replay_examples": stats.replay_examples,
        "peak_cuda_memory_mib": peak_cuda_mib,
        "iteration_seconds": duration_seconds,
        **arena_metrics,
    }


def _print_iteration_metrics(metrics: dict[str, Any]) -> None:
    print(
        "Iteration {iteration_idx} complete | policy_loss={policy_loss:.6f} "
        "value_loss={value_loss:.6f} total_loss={total_loss:.6f} | "
        "cap_termination_rate={cap_termination_rate:.3f} | games={games_completed}/"
        "{games_attempted} errors={game_errors} stalled_dropped={stalled_rollouts_dropped} | "
        "peak_cuda_memory={peak_cuda_memory_mib:.1f} MiB".format(**metrics)
    )
    if metrics["cap_termination_rate"] > 0.15:
        LOGGER.warning(
            "CAP TERMINATION TRIPWIRE: %.1f%% exceeds 15%%",
            metrics["cap_termination_rate"] * 100,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run durable Tabula Zero training")
    parser.add_argument("--games", type=int, default=5, help="Self-play games per iteration")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--arena-games", type=int, default=0)
    parser.add_argument("--mate-exam-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("ZERO_OUTPUT_DIR", "./checkpoints"),
    )
    parser.add_argument("--time-limit-hours", type=float, default=7.5)
    parser.add_argument("--max-game-error-rate", type=float, default=0.20)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-tui", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute the training loop. A failed iteration never replaces its predecessor."""

    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.games <= 0:
        raise ValueError("games must be positive")
    if args.arena_games < 0:
        raise ValueError("arena_games cannot be negative")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.mate_exam_every < 0:
        raise ValueError("mate_exam_every cannot be negative")
    if not 0.0 <= args.max_game_error_rate < 1.0:
        raise ValueError("max_game_error_rate must be in [0, 1)")
    if args.time_limit_hours <= 0:
        raise ValueError("time_limit_hours must be positive")
    paths = RunPaths.from_value(args.output_dir)
    paths.ensure()

    # This must happen before a worker, model, or training allocation is created.
    tablebase, syzygy_path = require_syzygy()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = None if args.no_resume else latest_checkpoint(paths)
    if args.no_resume and latest_checkpoint(paths) is not None:
        raise RuntimeError(
            "--no-resume cannot reuse an output directory containing checkpoints; "
            "choose a new --output-dir"
        )
    checkpoint = (
        load_checkpoint(checkpoint_path, map_location=device)
        if checkpoint_path is not None
        else None
    )
    iteration = int(checkpoint.get("iteration", 0)) + 1 if checkpoint else 1
    final_iteration = iteration + args.iterations - 1
    model_path = _seed_weights(paths, checkpoint)

    thermal = ThermalGuard()
    budget = WallClockBudget(args.time_limit_hours)
    previous_iteration_seconds: float | None = None
    live = None if args.no_tui else LiveBoard()

    print(f"Device: {device}")
    print(f"Syzygy: {syzygy_path}")
    print(f"Output: {paths.root}")
    if checkpoint_path is not None:
        print(f"Resuming from {checkpoint_path.name} at iteration {iteration}")

    try:
        while iteration <= final_iteration:
            can_start, reason = budget.can_start_iteration(previous_iteration_seconds)
            if not can_start:
                message = f"Graceful stop before iteration {iteration}: {reason}"
                print(message)
                _log_record(
                    paths.metrics_path,
                    "time_budget_stop",
                    iteration_idx=iteration,
                    reason=reason,
                    elapsed_seconds=budget.elapsed_seconds,
                )
                return 0

            started = time.monotonic()
            candidate_path: Path | None = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            try:
                if live is not None:
                    live.begin_round(iteration, model_path is not None)
                worker = SelfPlayWorker(
                    model_path=model_path,
                    num_simulations=DEFAULT_SIMULATIONS,
                    num_res_blocks=DEFAULT_RES_BLOCKS,
                    tablebase=tablebase,
                )
                replay, stats = run_selfplay_phase(
                    worker,
                    args.games,
                    iteration,
                    args.max_game_error_rate,
                    paths.metrics_path,
                    live=live,
                    thermal=thermal,
                )
                atomic_torch_save(replay, paths.replay_path(iteration))

                result = train(
                    data_dir=paths.replay_dir,
                    model_path=model_path,
                    checkpoint=checkpoint,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    device=device,
                    live=live,
                )

                candidate_path = paths.root / f".candidate_iter_{iteration:04d}.pt"
                atomic_torch_save(result.model.state_dict(), candidate_path)
                if args.mate_exam_every and iteration % args.mate_exam_every == 0:
                    run_mate_exam(
                        str(candidate_path),
                        out_path=str(paths.root / f"mate_exam_iter_{iteration:04d}.json"),
                    )
                promoted = not paths.champion_weights.exists()
                arena_metrics: dict[str, Any] = {
                    "arena_games": 0,
                    "arena_wins": 0,
                    "arena_losses": 0,
                    "arena_draws": 0,
                    "arena_win_rate": None,
                    "promoted": promoted,
                }
                if args.arena_games > 0 and paths.champion_weights.exists():
                    arena = Arena(
                        champion_path=paths.champion_weights,
                        challenger_path=candidate_path,
                        tablebase=tablebase,
                    )
                    promoted = arena.evaluate(
                        num_games=args.arena_games,
                        live=live,
                        promote=False,
                    )
                    arena_metrics = {
                        "arena_games": arena.last_games,
                        "arena_wins": arena.last_wins,
                        "arena_losses": arena.last_losses,
                        "arena_draws": arena.last_draws,
                        "arena_win_rate": arena.last_win_rate,
                        "promoted": promoted,
                    }
                    if live is not None:
                        live.note_match(
                            arena.last_win_rate,
                            arena.last_games,
                            promoted,
                            ch_wins=arena.last_wins,
                            ch_losses=arena.last_losses,
                            draws=arena.last_draws,
                        )
                elif live is not None and promoted:
                    live.note_match(1.0, 0, True, first=True)

                duration = time.monotonic() - started
                peak_memory = peak_cuda_memory_mib(device)
                metrics = _iteration_metrics(
                    iteration,
                    stats,
                    result,
                    duration,
                    peak_memory,
                    arena_metrics,
                )
                payload = checkpoint_payload(
                    iteration=iteration,
                    model_state=result.model.state_dict(),
                    optimizer_state=result.optimizer.state_dict(),
                    scheduler_state=result.scheduler.state_dict(),
                    metadata={
                        "metrics": metrics,
                        "config": vars(args),
                        "selfplay": asdict(stats),
                        "syzygy_path": str(syzygy_path),
                    },
                )

                # The versioned checkpoint is canonical and written first. Raw
                # weights are reconstructable from it if preemption lands between files.
                atomic_torch_save(payload, paths.checkpoint_path(iteration))
                atomic_torch_save(result.model.state_dict(), paths.latest_weights)
                if promoted:
                    atomic_torch_save(result.model.state_dict(), paths.champion_weights)
                atomic_jsonl_append(paths.metrics_path, metrics)
                candidate_path.unlink(missing_ok=True)

                _print_iteration_metrics(metrics)
                checkpoint = payload
                model_path = paths.latest_weights
                previous_iteration_seconds = duration
                iteration += 1
            except Exception as exc:
                if candidate_path is not None:
                    candidate_path.unlink(missing_ok=True)
                error_traceback = traceback.format_exc()
                LOGGER.error(
                    "Iteration %d failed; the previous checkpoint remains intact:\n%s",
                    iteration,
                    error_traceback,
                )
                _log_record(
                    paths.metrics_path,
                    "iteration_error",
                    iteration_idx=iteration,
                    error=repr(exc),
                    traceback=error_traceback,
                )
                raise
    except KeyboardInterrupt:
        print("Stopped by user; the last complete checkpoint remains intact.")
        return 0
    finally:
        tablebase.close()
        if live is not None and live.tui is not None:
            live.tui.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

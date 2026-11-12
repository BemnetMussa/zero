import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tabula import pipeline
from tabula.runtime import RunPaths, load_checkpoint
from tabula.train.train import TrainResult


class FakeWorker:
    outcomes = []

    def __init__(self, **_kwargs):
        self._outcomes = list(type(self).outcomes)
        self.last_termination = None

    def play_game(self, **_kwargs):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        data, self.last_termination = outcome
        return data


class FakeState:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class FakeTablebase:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class SelfPlayResilienceTests(unittest.TestCase):
    def test_one_transient_error_continues_within_budget(self) -> None:
        FakeWorker.outcomes = [
            RuntimeError("bad rollout"),
            ([1], "CHECKMATE"),
            ([2], "CHECKMATE"),
            ([3], "CHECKMATE"),
            ([4], "CHECKMATE"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            data, stats = pipeline.run_selfplay_phase(
                FakeWorker(),
                games=5,
                iteration=1,
                max_game_error_rate=0.20,
                metrics_path=Path(tmp) / "metrics.jsonl",
            )
        self.assertEqual(data, [1, 2, 3, 4])
        self.assertEqual(stats.game_errors, 1)
        self.assertEqual(stats.games_completed, 4)

    def test_exceeding_error_budget_aborts_iteration(self) -> None:
        FakeWorker.outcomes = [
            RuntimeError("first"),
            ([1], "CHECKMATE"),
            RuntimeError("second"),
            ([2], "CHECKMATE"),
            ([3], "CHECKMATE"),
        ]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
            pipeline.run_selfplay_phase(
                FakeWorker(),
                games=5,
                iteration=1,
                max_game_error_rate=0.20,
                metrics_path=Path(tmp) / "metrics.jsonl",
            )

    def test_stalled_practice_and_full_board_cap_are_counted(self) -> None:
        FakeWorker.outcomes = [
            ([1], "MOVE_LIMIT"),
            ([], "MOVE_LIMIT"),
            ([2], "CHECKMATE"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _, stats = pipeline.run_selfplay_phase(
                FakeWorker(),
                games=3,
                iteration=1,
                max_game_error_rate=0.20,
                metrics_path=Path(tmp) / "metrics.jsonl",
            )
        self.assertEqual(stats.stalled_rollouts_dropped, 1)
        self.assertEqual(stats.cap_terminations, 1)
        self.assertEqual(stats.full_board_games, 2)
        self.assertEqual(stats.cap_termination_rate, 0.5)

    def test_cap_tripwire_warns_above_fifteen_percent(self) -> None:
        metrics = {
            "iteration_idx": 1,
            "policy_loss": 1.0,
            "value_loss": 0.5,
            "total_loss": 1.5,
            "cap_termination_rate": 0.16,
            "games_completed": 5,
            "games_attempted": 5,
            "game_errors": 0,
            "stalled_rollouts_dropped": 0,
            "peak_cuda_memory_mib": 0.0,
        }
        with self.assertLogs("tabula.pipeline", level="WARNING") as captured:
            pipeline._print_iteration_metrics(metrics)
        self.assertIn("TRIPWIRE", "\n".join(captured.output))


class PipelineArtifactTests(unittest.TestCase):
    def test_simulation_budget_is_configurable(self) -> None:
        args = pipeline.parse_args(["--simulations", "30"])
        self.assertEqual(args.simulations, 30)

    def test_low_disk_startup_exits_cleanly_before_syzygy_or_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(
                ["--no-tui", "--output-dir", tmp, "--iterations", "1"]
            )
            with (
                mock.patch.object(
                    pipeline,
                    "check_disk_capacity",
                    return_value=(False, 1024, {"checkpoints": [], "replays": []}),
                ),
                mock.patch.object(pipeline, "require_syzygy") as require_syzygy,
            ):
                self.assertEqual(pipeline.run(args), 0)
            require_syzygy.assert_not_called()

    def test_resume_rebuilds_raw_weights_from_canonical_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.from_value(tmp)
            paths.ensure()
            torch.save({"weight": torch.tensor([0.0])}, paths.latest_weights)
            checkpoint = {
                "model_state": {"weight": torch.tensor([2.0])},
                "metadata": {"metrics": {"promoted": True}},
            }
            selected = pipeline._seed_weights(paths, checkpoint)
            self.assertEqual(selected, paths.latest_weights)
            latest = torch.load(paths.latest_weights, weights_only=True)
            champion = torch.load(paths.champion_weights, weights_only=True)
            torch.testing.assert_close(latest["weight"], torch.tensor([2.0]))
            torch.testing.assert_close(champion["weight"], torch.tensor([2.0]))

    def test_one_iteration_writes_all_durable_artifacts(self) -> None:
        model = FakeState({"weight": torch.tensor([1.0])})
        optimizer = FakeState({"step": 1})
        scheduler = FakeState({"cycle": 1})
        result = TrainResult(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            policy_loss=1.25,
            value_loss=0.25,
            total_loss=1.50,
            learning_rate=0.001,
            examples=1,
        )
        tablebase = FakeTablebase()
        FakeWorker.outcomes = [([("state", "pi", "z", "weight")], "CHECKMATE")]

        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(
                [
                    "--no-tui",
                    "--output-dir",
                    tmp,
                    "--iterations",
                    "1",
                    "--games",
                    "1",
                    "--arena-games",
                    "0",
                    "--mate-exam-every",
                    "0",
                    "--snapshot-every",
                    "1",
                    "--time-limit-hours",
                    "1",
                ]
            )
            with (
                mock.patch.object(pipeline, "require_syzygy", return_value=(tablebase, Path("/tables"))),
                mock.patch.object(pipeline, "SelfPlayWorker", FakeWorker),
                mock.patch.object(pipeline, "train", return_value=result),
                mock.patch.object(pipeline.torch.cuda, "is_available", return_value=False),
                mock.patch.object(pipeline, "ThermalGuard"),
            ):
                self.assertEqual(pipeline.run(args), 0)

            paths = RunPaths.from_value(tmp)
            self.assertTrue(paths.replay_path(1).exists())
            self.assertTrue(paths.checkpoint_path(1).exists())
            self.assertTrue(paths.latest_weights.exists())
            self.assertTrue(paths.champion_weights.exists())
            self.assertTrue(paths.snapshot_path(1).exists())
            checkpoint = load_checkpoint(paths.checkpoint_path(1))
            self.assertEqual(checkpoint["iteration"], 1)
            records = [
                json.loads(line) for line in paths.metrics_path.read_text().splitlines()
            ]
            self.assertEqual(records[-1]["event"], "iteration_complete")
            self.assertEqual(records[-1]["policy_loss"], 1.25)
            snapshot = torch.load(paths.snapshot_path(1), weights_only=True)
            self.assertEqual(set(snapshot), {"weight"})
            self.assertFalse(any(paths.root.glob(".candidate*")))
            self.assertTrue(tablebase.closed)


if __name__ == "__main__":
    unittest.main()

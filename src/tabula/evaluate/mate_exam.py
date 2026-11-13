"""Mate exam: can the brain finish K+Q vs K and K+R vs K?

The same 20 practice boards every time (fixed seed), both sides playing the
search's most-visited move with no noise. Self-play mate counts are noisy;
this is the before/after ruler for practice games.

Run: PYTHONPATH=src python -m tabula.evaluate.mate_exam --model models/tabula_v0.2_latest.pth

Offline only: pass --syzygy PATH to report each board's perfect-play DTZ next
to the plies the model needed. The tables never influence the moves played.
"""

import argparse
import json
import os
import random
import time

import chess

from tabula.evaluate.benchmark import DEFAULT_MODEL, load_model
from tabula.game.encoder import encode_board
from tabula.evaluate.oracle import open_syzygy, optimal_dtz
from tabula.mcts.search import MCTS
from tabula.selfplay.practice import random_practice_board
from tabula.selfplay.worker import DEFAULT_RES_BLOCKS

EXAM_SEED = 7
EXAM_BOARDS = 20
EXAM_MAX_PLIES = 100


def exam_boards(n=EXAM_BOARDS, seed=EXAM_SEED):
    rng = random.Random(seed)
    return [random_practice_board(rng) for _ in range(n)]


def play_out(mcts, board):
    """Greedy play to the end. Returns (mated, plies, termination)."""
    board = board.copy()
    plies = 0
    while board.outcome(claim_draw=True) is None and plies < EXAM_MAX_PLIES:
        root = mcts.search(board, encode_board, add_noise=False)
        move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        board.push(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return False, plies, "MOVE_LIMIT"
    return outcome.winner is not None, plies, outcome.termination.name


def run_mate_exam(
    model_path,
    num_simulations=80,
    num_res_blocks=DEFAULT_RES_BLOCKS,
    out_path=None,
    model=None,
    tablebase=None,
):
    model = model if model is not None else load_model(model_path, num_res_blocks)
    mcts = MCTS(model, num_simulations=num_simulations, c_puct=1.25)

    results = []
    t_start = time.time()
    for idx, (board, strong) in enumerate(exam_boards(), start=1):
        piece = "Q" if board.pieces(chess.QUEEN, strong) else "R"
        _, start_value = mcts.predict(board, encode_board)
        mated, plies, termination = play_out(mcts, board)
        row = {
            "idx": idx,
            "fen": board.fen(),
            "piece": piece,
            "start_value": round(float(start_value), 3),
            "mated": mated,
            "plies": plies,
            "termination": termination,
        }
        perfect = ""
        if tablebase is not None:
            row["optimal_dtz"] = optimal_dtz(board, tablebase)
            perfect = f" | perfect DTZ={row['optimal_dtz']}"
        results.append(row)
        print(f"[{idx:02d}/{EXAM_BOARDS}] K+{piece} vs K | V={start_value:+.2f} | "
              f"{'MATE' if mated else 'no mate'} after {plies} plies ({termination}){perfect}")

    mates = [r for r in results if r["mated"]]
    summary = {
        "model": model_path,
        "num_simulations": num_simulations,
        "mates": len(mates),
        "n": len(results),
        "mates_queen": sum(r["mated"] for r in results if r["piece"] == "Q"),
        "n_queen": sum(r["piece"] == "Q" for r in results),
        "mates_rook": sum(r["mated"] for r in results if r["piece"] == "R"),
        "n_rook": sum(r["piece"] == "R" for r in results),
        "avg_plies_to_mate": round(sum(r["plies"] for r in mates) / len(mates), 1) if mates else None,
        "avg_start_value": round(sum(r["start_value"] for r in results) / len(results), 3),
        "elapsed_sec": round(time.time() - t_start, 1),
        "results": results,
    }
    if tablebase is not None:
        optimal = [r["optimal_dtz"] for r in mates if r.get("optimal_dtz") is not None]
        summary["avg_optimal_dtz_mated"] = round(sum(optimal) / len(optimal), 1) if optimal else None

    print("\n=== Zero mate exam ===")
    print(f"Mates: {summary['mates']}/{summary['n']} "
          f"(queen {summary['mates_queen']}/{summary['n_queen']}, rook {summary['mates_rook']}/{summary['n_rook']})")
    print(f"Average plies to mate: {summary['avg_plies_to_mate']}")
    print(f"Average start value for the strong side: {summary['avg_start_value']:+.3f}")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Zero mate exam (K+Q / K+R vs K)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--res-blocks", type=int, default=DEFAULT_RES_BLOCKS)
    parser.add_argument("--out", default="data/mate_exam.json")
    parser.add_argument("--syzygy", default=None, help="Offline only: Syzygy dir for perfect-play DTZ")
    args = parser.parse_args()
    status = open_syzygy(args.syzygy) if args.syzygy else None
    try:
        run_mate_exam(
            args.model,
            args.sims,
            args.res_blocks,
            args.out,
            tablebase=status.tablebase if status else None,
        )
    finally:
        if status is not None:
            status.tablebase.close()


if __name__ == "__main__":
    main()

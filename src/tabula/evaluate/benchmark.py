"""50-position tactical benchmark — the outside ruler for Zero.

Before changing how Zero learns, measure what a champion already knows:
  - Raw policy (P) top-1: what the net picks with NO search (its habit)
  - MCTS best move:        what the net finds after N simulations (its judgment)
  - Value (V) sign:        does the net know the side to move is winning?

Every position is winning for the side to move. Each entry:
  (category, FEN, expected best move in UCI)

Categories: mate-in-1 (10), mate-in-2 (10), hanging piece (10),
fork / pin (10), endgame conversion (10).

Run:
  python -m tabula.evaluate.benchmark
  python -m tabula.evaluate.benchmark --model models/tabula_v0.2_champion.pth --sims 80
"""

import argparse
import json
import os
import time

import chess
import torch

from tabula.model.model import TabulaNet
from tabula.mcts.search import MCTS
from tabula.game.encoder import encode_board

DEFAULT_RES_BLOCKS = 8
DEFAULT_MODEL = "models/tabula_v0.2_champion.pth"

POSITIONS = [
    # ---------------- Mate in 1 (10) ----------------
    ("mate-in-1", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "a1a8"),          # back rank
    ("mate-in-1", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8"),     # back rank
    ("mate-in-1", "7k/6pp/8/8/8/8/6PP/5RK1 w - - 0 1", "f1f8"),         # back rank h-corner
    ("mate-in-1", "k7/8/1K6/8/8/8/8/2R5 w - - 0 1", "c1c8"),            # K+R vs K
    ("mate-in-1", "k7/8/KQ6/8/8/8/8/8 w - - 0 1", "b6b7"),              # K+Q vs K
    ("mate-in-1", "6k1/8/6K1/8/8/8/8/Q7 w - - 0 1", "a1a8"),            # K+Q vs K back rank
    ("mate-in-1", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),  # scholar's mate
    ("mate-in-1", "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", "d8h4"),       # fool's mate (black)
    ("mate-in-1", "k7/8/1K6/8/8/8/8/7R w - - 0 1", "h1h8"),             # K+R vs K h-file
    ("mate-in-1", "6k1/5ppp/8/8/8/8/8/3Q2K1 w - - 0 1", "d1d8"),        # queen back rank
    # ---------------- Mate in 2 (10) ----------------
    ("mate-in-2", "6k1/5ppp/8/6K1/8/8/5Q2/8 w - - 0 1", "f2f7"),        # Qxf7+ Kh8 Qf8#
    ("mate-in-2", "6k1/6pp/8/6K1/8/8/5Q2/8 w - - 0 1", "f2f7"),         # Qf7+ Kh8 Qf8#
    ("mate-in-2", "6k1/8/1R6/8/8/8/8/R6K w - - 0 1", "b6b7"),           # lawnmower quiet move
    ("mate-in-2", "7k/8/8/8/1R6/8/8/R6K w - - 0 1", "b4b7"),            # lawnmower quiet move
    ("mate-in-2", "6k1/6pp/8/4K3/8/8/5Q2/8 w - - 0 1", "f2f7"),         # Qf7+ (Ke6 support)
    ("mate-in-2", "5rk1/5ppp/8/8/8/8/3R4/3R2K1 w - - 0 1", "d2d8"),     # back-rank deflection
    ("mate-in-2", "6k1/5pp1/8/6K1/8/8/5Q2/8 w - - 0 1", "f2f7"),        # Qxf7+ Kh8 Qf8#
    ("mate-in-2", "r5k1/8/8/8/8/1r6/8/6K1 b - - 0 1", "b3b2"),         # black lawnmower: Rb2! then Ra1#
    ("mate-in-2", "2k5/8/4R3/8/8/8/8/R6K w - - 0 1", "e6e7"),           # lawnmower quiet move
    ("mate-in-2", "2k5/2p1p3/8/1K6/8/8/2Q5/8 w - - 0 1", "c2c7"),       # Qxc7+ Ke8 Qc8#
    # ---------------- Hanging pieces (10) ----------------
    ("hanging", "6k1/8/8/8/4q3/3B4/8/6K1 w - - 0 1", "d3e4"),           # Bxe4 wins queen
    ("hanging", "6k1/8/8/3r4/8/8/8/3R2K1 w - - 0 1", "d1d5"),           # Rxd5
    ("hanging", "6k1/8/8/8/2n5/8/B7/6K1 w - - 0 1", "a2c4"),            # Bxc4 wins knight
    ("hanging", "6k1/8/8/8/r7/8/8/Q5K1 w - - 0 1", "a1a4"),             # Qxa4 wins rook
    ("hanging", "6k1/8/8/3n4/4P3/8/8/6K1 w - - 0 1", "e4d5"),           # exd5 wins knight
    ("hanging", "6k1/8/3r4/8/8/B7/8/6K1 w - - 0 1", "a3d6"),            # Bxd6 wins rook
    ("hanging", "6k1/8/8/8/8/2b5/8/2R3K1 w - - 0 1", "c1c3"),           # Rxc3 wins bishop
    ("hanging", "6k1/8/8/3q4/8/4N3/8/6K1 w - - 0 1", "e3d5"),           # Nxd5 wins queen
    ("hanging", "6k1/8/8/8/8/1n6/8/3Q2K1 w - - 0 1", "d1b3"),           # Qxb3 wins knight
    ("hanging", "6k1/8/8/8/8/8/1K6/2r5 w - - 0 1", "b2c1"),             # Kxc1 wins rook
    # ---------------- Forks and pins (10) ----------------
    ("fork-pin", "r2q2k1/8/8/8/4N3/8/8/6K1 w - - 0 1", "e4f6"),         # Nf6+ forks Kg8 + Qd8
    ("fork-pin", "4r1k1/8/8/8/4N3/8/8/6K1 w - - 0 1", "e4f6"),          # Nf6+ forks Kg8 + Re8
    ("fork-pin", "6k1/8/2n1b3/8/3P4/8/8/6K1 w - - 0 1", "d4d5"),        # pawn fork N + B
    ("fork-pin", "r5k1/8/8/8/8/8/1Q6/6K1 w - - 0 1", "b2b8"),           # Qb8+ skewers Ra8
    ("fork-pin", "2q3k1/8/8/3N4/8/8/8/6K1 w - - 0 1", "d5e7"),          # Ne7+ forks Kg8 + Qc8
    ("fork-pin", "4k3/4r3/8/8/8/B7/8/6K1 w - - 0 1", "a3e7"),           # Bxe7 wins pinned rook
    ("fork-pin", "r3k3/8/8/1N6/8/8/8/6K1 w - - 0 1", "b5c7"),           # Nc7+ forks Ke8 + Ra8
    ("fork-pin", "6k1/8/8/2n5/2r5/8/8/3Q2K1 w - - 0 1", "d1d5"),        # Qd5 pins Nc5 to Rc4
    ("fork-pin", "4k3/4r3/8/8/8/8/3B4/6K1 w - - 0 1", "d2g5"),          # Bg5 pins Re7 to Ke8
    ("fork-pin", "r3k3/8/8/3N4/8/8/8/6K1 w - - 0 1", "d5c7"),           # Nc7+ forks Ke8 + Ra8
    # ---------------- Endgame conversion (10) ----------------
    ("endgame", "8/8/1k6/8/8/1K6/1P6/8 w - - 0 1", "b3c4"),             # take opposition
    ("endgame", "8/8/8/1k6/8/2K5/1P6/8 w - - 0 1", "b2b4"),             # push passed pawn
    ("endgame", "6k1/8/6K1/8/8/8/1Q6/8 w - - 0 1", "b2b8"),             # K+Q mate back rank
    ("endgame", "k7/8/1K6/8/8/8/8/3R4 w - - 0 1", "d1d8"),              # K+R mate
    ("endgame", "8/8/1k6/8/8/4K3/4P3/8 w - - 0 1", "e3e4"),             # centralize king
    ("endgame", "8/8/8/k7/8/1K2P3/8/8 w - - 0 1", "b3c4"),              # escort pawn
    ("endgame", "6k1/8/6K1/8/3Q4/8/8/8 w - - 0 1", "d4g7"),             # Qg7#
    ("endgame", "7k/8/6K1/8/8/8/4Q3/8 w - - 0 1", "e2e8"),              # Qe8# h-corner
    ("endgame", "8/8/8/8/1k6/8/1K2P3/8 w - - 0 1", "e2e4"),             # seize promotion square
    ("endgame", "8/8/1k6/8/8/1K6/1P6/8 b - - 0 1", "b6c5"),             # black takes opposition
]


def load_model(model_path, num_res_blocks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabulaNet(num_res_blocks=num_res_blocks).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded {model_path} on {device}")
    else:
        print(f"WARNING: no weights at {model_path} — running with random weights on {device}")
    model.eval()
    return model


def run_benchmark(model_path, num_simulations, num_res_blocks):
    model = load_model(model_path, num_res_blocks)
    mcts = MCTS(model, num_simulations=num_simulations, c_puct=1.25)

    per_category = {}
    results = []
    t_start = time.time()

    for idx, (category, fen, expected_uci) in enumerate(POSITIONS, start=1):
        board = chess.Board(fen)
        expected = chess.Move.from_uci(expected_uci)

        # 1. Raw policy (P): the net's habit with no search.
        policy_dict, value = mcts.predict(board, encode_board)
        raw_top = max(policy_dict, key=policy_dict.get) if policy_dict else None

        # 2. MCTS best move after N simulations (no exploration noise).
        root = mcts.search(board, encode_board, add_noise=False)
        search_top = None
        best_visits = -1
        for move, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                search_top = move

        raw_ok = raw_top == expected
        mcts_ok = search_top == expected
        value_ok = value > 0.0  # every position is winning for the side to move

        cat = per_category.setdefault(category, {"raw": 0, "mcts": 0, "value": 0, "n": 0})
        cat["raw"] += int(raw_ok)
        cat["mcts"] += int(mcts_ok)
        cat["value"] += int(value_ok)
        cat["n"] += 1

        results.append({
            "idx": idx,
            "category": category,
            "fen": fen,
            "expected": expected_uci,
            "raw_top": raw_top.uci() if raw_top else None,
            "raw_ok": raw_ok,
            "mcts_top": search_top.uci() if search_top else None,
            "mcts_ok": mcts_ok,
            "value": round(float(value), 3),
            "value_ok": value_ok,
        })
        flag = "OK " if mcts_ok else "   "
        print(f"[{idx:02d}/50] {flag} {category:9s} want {expected_uci} | P: {raw_top.uci() if raw_top else '-':5s} "
              f"| MCTS: {search_top.uci() if search_top else '-':5s} | V={value:+.2f}")

    elapsed = time.time() - t_start
    n = len(POSITIONS)
    total_raw = sum(r["raw_ok"] for r in results)
    total_mcts = sum(r["mcts_ok"] for r in results)
    total_value = sum(r["value_ok"] for r in results)

    print("\n=== Zero 50-Position Tactical Benchmark ===")
    print(f"Model: {model_path} | MCTS sims: {num_simulations} | {elapsed:.0f}s total")
    print(f"{'category':10s} {'policy(P)':>10s} {'mcts(Q)':>8s} {'value(V)':>9s}")
    for category, c in per_category.items():
        print(f"{category:10s} {c['raw']:4d}/{c['n']:<5d} {c['mcts']:3d}/{c['n']:<4d} {c['value']:4d}/{c['n']:<4d}")
    print(f"{'TOTAL':10s} {total_raw:4d}/{n:<5d} {total_mcts:3d}/{n:<4d} {total_value:4d}/{n:<4d}")

    os.makedirs("data", exist_ok=True)
    report = {
        "model": model_path,
        "num_simulations": num_simulations,
        "elapsed_sec": round(elapsed, 1),
        "total": {"raw_policy": total_raw, "mcts": total_mcts, "value_sign": total_value, "n": n},
        "per_category": per_category,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("data/benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved report to data/benchmark_report.json")


def main():
    parser = argparse.ArgumentParser(description="Zero 50-position tactical benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to model weights")
    parser.add_argument("--sims", type=int, default=80, help="MCTS simulations per position")
    parser.add_argument("--res-blocks", type=int, default=DEFAULT_RES_BLOCKS)
    args = parser.parse_args()
    run_benchmark(args.model, args.sims, args.res_blocks)


if __name__ == "__main__":
    main()

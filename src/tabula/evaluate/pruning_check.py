"""Inspect raw vs pruned root visits on a few sharp positions (Step 3 gate).

Run:
  PYTHONPATH=src python -m tabula.evaluate.pruning_check
"""

import os

import chess
import torch

from tabula.model.model import TabulaNet
from tabula.mcts.search import MCTS
from tabula.game.encoder import encode_board

DEFAULT_RES_BLOCKS = 8
DEFAULT_MODEL = "models/tabula_v0.2_champion.pth"

# Positions where a blunder is legal and a real tactic exists.
POSITIONS = [
    ("mate-in-1 back rank", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "a1a8"),
    ("hanging queen", "6k1/8/8/8/4q3/3B4/8/6K1 w - - 0 1", "d3e4"),
    ("scholar mate", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabulaNet(num_res_blocks=DEFAULT_RES_BLOCKS).to(device)
    if os.path.exists(DEFAULT_MODEL):
        model.load_state_dict(torch.load(DEFAULT_MODEL, map_location=device, weights_only=True))
        print(f"Loaded {DEFAULT_MODEL} on {device}")
    else:
        print(f"WARNING: no weights at {DEFAULT_MODEL}")
    model.eval()
    mcts = MCTS(model, num_simulations=80, c_puct=1.25)

    for name, fen, expected in POSITIONS:
        board = chess.Board(fen)
        root = mcts.search(board, encode_board, add_noise=True, force_playouts=True)
        raw, pruned = mcts.pruned_visits(root)
        print(f"\n=== {name}  (want {expected}) ===")
        rows = []
        for move in raw:
            rows.append((raw[move], pruned.get(move, 0.0), move.uci()))
        rows.sort(reverse=True)
        print(f"{'move':6s} {'raw':>6s} {'pruned':>8s}")
        for raw_n, pruned_n, uci in rows[:8]:
            mark = " <- expected" if uci == expected else ""
            print(f"{uci:6s} {raw_n:6.1f} {pruned_n:8.1f}{mark}")
        zeroed = sum(1 for move, n in raw.items() if n > 0 and pruned.get(move, 0) == 0)
        kept_second = sum(1 for move, n in pruned.items() if n > 1 and move.uci() != rows[0][2])
        print(f"zeroed {zeroed} noise moves; {kept_second} non-best moves kept with visits > 1")


if __name__ == "__main__":
    main()

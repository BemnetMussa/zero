"""Inspect root visit distributions on a few sharp positions.

Run #3 dropped KataGo forced-playout pruning from the self-play path. This
helper now prints raw normalized visit counts under noisy full search.

Run:
  PYTHONPATH=src python -m tabula.evaluate.pruning_check
"""

import os

import chess
import torch

from tabula.checkpoint import load_model_weights
from tabula.game.encoder import encode_board
from tabula.mcts.search import MCTS, visit_policy_target
from tabula.model.model import TabulaNet

DEFAULT_RES_BLOCKS = 8
DEFAULT_MODEL = "models/tabula_v0.2_champion.pth"

POSITIONS = [
    ("mate-in-1 back rank", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "a1a8"),
    ("hanging queen", "6k1/8/8/8/4q3/3B4/8/6K1 w - - 0 1", "d3e4"),
    ("scholar mate", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabulaNet(num_res_blocks=DEFAULT_RES_BLOCKS).to(device)
    if os.path.exists(DEFAULT_MODEL):
        raw = torch.load(DEFAULT_MODEL, map_location=device, weights_only=True)
        load_model_weights(model, raw)
        print(f"Loaded {DEFAULT_MODEL} on {device}")
    else:
        print(f"WARNING: no weights at {DEFAULT_MODEL}")
    model.eval()
    mcts = MCTS(model, num_simulations=80, c_puct=1.25)

    for name, fen, expected in POSITIONS:
        board = chess.Board(fen)
        root = mcts.search(board, encode_board, add_noise=True)
        policy = visit_policy_target(root)
        print(f"\n=== {name}  (want {expected}) ===")
        rows = sorted(
            ((root.children[m].visit_count, policy.get(m, 0.0), m.uci()) for m in root.children),
            reverse=True,
        )
        print(f"{'move':6s} {'visits':>6s} {'pi':>8s}")
        for visits, pi, uci in rows[:8]:
            mark = " <- expected" if uci == expected else ""
            print(f"{uci:6s} {visits:6.1f} {pi:8.3f}{mark}")


if __name__ == "__main__":
    main()

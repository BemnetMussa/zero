"""Play White against Zero v0.2 (PyTorch + MCTS) in the terminal."""

from __future__ import annotations

import os
import torch

from tabula.game import Game, IllegalMove
from tabula.model.model import TabulaNet
from tabula.game.encoder import encode_board
from tabula.mcts.search import MCTS

DEFAULT_RES_BLOCKS = 8
WEIGHTS = "models/tabula_v0.2_champion.pth"


def main() -> None:
    print("Loading Zero net (v0.2, 8 ResBlocks)...")
    model = TabulaNet(num_res_blocks=DEFAULT_RES_BLOCKS)
    if os.path.exists(WEIGHTS):
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        print(f"Loaded {WEIGHTS}")
    else:
        print(f"No champion at {WEIGHTS}; playing with random weights.")
    model.eval()

    mcts = MCTS(model, num_simulations=80, c_puct=1.25)

    game = Game()
    print("You are White. Zero v0.2 (MCTS) is Black.")
    print("Moves: e2e4 or e4. Quit with q.")
    print(game.board)

    while game.result() is None:
        if game.board.turn:
            typed = input("\nYour move: ").strip()
            if typed.lower() in {"q", "quit"}:
                print("stopped")
                return
            try:
                played = game.push(typed)
            except IllegalMove:
                print(f"illegal move: {typed}")
                continue
            print(f"you: {played}")
        else:
            print("Thinking...")
            root_node = mcts.search(game.board, encode_board)

            best_move = None
            best_visits = -1
            for move, child in root_node.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = move

            played = game.push(best_move.uci())
            print(f"zero: {played}")

        print(game.board)

    print(f"\n{game.result()} ({game.reason()})")

if __name__ == "__main__":
    main()

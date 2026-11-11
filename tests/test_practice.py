import random
import unittest

import chess

from tabula.selfplay.practice import random_practice_board


class PracticeBoardTests(unittest.TestCase):
    def test_boards_are_legal_and_strong_side_moves(self) -> None:
        rng = random.Random(0)
        for _ in range(200):
            board, strong = random_practice_board(rng)
            self.assertTrue(board.is_valid())
            self.assertIsNone(board.outcome(claim_draw=True))
            self.assertEqual(board.turn, strong)
            extra = len(board.pieces(chess.QUEEN, strong)) + len(board.pieces(chess.ROOK, strong))
            self.assertEqual(extra, 1)
            self.assertEqual(len(board.piece_map()), 3)


if __name__ == "__main__":
    unittest.main()

import random
import unittest

import chess

from tabula.game.score import REPETITION_CONTEMPT, score_game
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


class PracticeScoringTests(unittest.TestCase):
    def test_mate_is_a_full_win(self) -> None:
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
        board.push_uci("a1a8")
        reward, _, term = score_game(board, board.outcome(), None, practice_side=chess.WHITE)
        self.assertEqual(reward, 1.0)
        self.assertEqual(term, "CHECKMATE")

    def test_stalemate_is_a_failure_for_the_strong_side(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        reward, result, term = score_game(board, board.outcome(), None, practice_side=chess.WHITE)
        self.assertEqual(reward, -REPETITION_CONTEMPT)
        self.assertEqual(result, "NO MATE")
        self.assertEqual(term, "STALEMATE")

    def test_move_cap_is_a_failure_for_black_too(self) -> None:
        board = chess.Board("8/8/8/4k3/8/8/2q5/K7 w - - 0 1")
        reward, _, term = score_game(board, None, "MOVE_LIMIT", practice_side=chess.BLACK)
        self.assertEqual(reward, REPETITION_CONTEMPT)
        self.assertEqual(term, "MOVE_LIMIT")

    def test_normal_games_are_unchanged(self) -> None:
        board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
        reward, result, _ = score_game(board, None, "MATERIAL")
        self.assertGreater(reward, 0.9)
        self.assertEqual(result, "WHITE AHEAD")


if __name__ == "__main__":
    unittest.main()

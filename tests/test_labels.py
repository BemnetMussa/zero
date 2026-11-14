"""Policy 6: the label math is checked on tiny games before any training run."""

import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import chess
import numpy as np
import torch

from tabula import pipeline, runtime
from tabula.game.encoder import encode_board
from tabula.evaluate import arena
from tabula.mcts import search
from tabula.mcts.search import terminal_value, tree_leaf_value
from tabula.selfplay import labels, practice, worker
from tabula.selfplay.labels import Record, game_result, label_game
from tabula.train import train
from tabula.train.train import SelfPlayDataset

PENALTY = labels.DEFAULT_REPETITION_PENALTY


def play(fen, ucis):
    """Final board plus one record per ply, as if every turn were a full search."""
    board = chess.Board(fen)
    records = []
    for ply, uci in enumerate(ucis):
        records.append(Record(
            state=encode_board(board),
            pi=np.zeros(4672, dtype=np.float32),
            turn=board.turn,
            ply=ply,
            v_prior=0.0,
            q_root=0.0,
            fen=board.fen(),
        ))
        board.push_uci(uci)
    return board, records


def zs(data):
    return [float(z[0]) for _, _, z, _ in data]


# Kf1, Kh8, Ra8#: White mates on ply 3.
BACK_RANK = ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", ["g1f1", "g8h8", "a1a8"])
KNIGHT_SHUFFLE = [
    "g1f3", "g8f6", "f3g1", "f6g8",
    "g1f3", "g8f6", "f3g1", "f6g8",
]
KQK_SHUFFLE = ("7k/8/8/8/8/8/8/KQ6 w - - 0 1", ["b1b2", "h8g8", "b2b1", "g8h8"] * 2)


class FlatMateTests(unittest.TestCase):
    def test_mate_targets_are_flat_across_plies(self) -> None:
        board, records = play(*BACK_RANK)
        self.assertTrue(board.is_checkmate())
        data = label_game(records, board, stop=None)
        expected = [1.0, -1.0, 1.0]
        np.testing.assert_allclose(zs(data), expected, rtol=1e-6)

    def test_no_gamma_discount_on_mate(self) -> None:
        board, records = play(*BACK_RANK)
        data = label_game(records, board, stop=None)
        self.assertEqual(zs(data)[0], zs(data)[2])
        self.assertNotIn("GAMMA", labels.__dict__)


class RepetitionPenaltyTests(unittest.TestCase):
    def test_threefold_charges_the_side_that_initiated_the_loop(self) -> None:
        board, records = play(chess.STARTING_FEN, KNIGHT_SHUFFLE)
        self.assertEqual(
            board.outcome(claim_draw=True).termination,
            chess.Termination.THREEFOLD_REPETITION,
        )
        white, _, term = game_result(board, None)
        self.assertEqual(term, "THREEFOLD_REPETITION")
        self.assertAlmostEqual(white, PENALTY)
        data = label_game(records, board, stop=None)
        self.assertEqual(len(data), len(records))
        # Flat: every White-to-move ply sees PENALTY, Black sees -PENALTY.
        self.assertAlmostEqual(zs(data)[-1], -PENALTY, places=6)
        self.assertAlmostEqual(zs(data)[0], PENALTY, places=6)

    def test_claim_by_move_charges_the_side_to_move_in_the_repeated_position(self) -> None:
        board, _ = play(KQK_SHUFFLE[0], KQK_SHUFFLE[1][:7])
        self.assertFalse(board.is_repetition(3))
        self.assertEqual(
            board.outcome(claim_draw=True).termination,
            chess.Termination.THREEFOLD_REPETITION,
        )
        self.assertEqual(labels.repetition_offender(board), chess.WHITE)
        self.assertAlmostEqual(game_result(board, None)[0], PENALTY)

    def test_strong_side_shuffle_is_penalised_not_rewarded(self) -> None:
        board, records = play(*KQK_SHUFFLE)
        data = label_game(records, board, stop=None)
        self.assertEqual(len(data), len(records))
        white_rows = [z for rec, z in zip(records, zs(data)) if rec.turn == chess.WHITE]
        self.assertTrue(all(z < 0.0 for z in white_rows))

    def test_penalty_is_configurable(self) -> None:
        board, _ = play(chess.STARTING_FEN, KNIGHT_SHUFFLE)
        with mock.patch.dict(os.environ, {"TABULA_REPETITION_PENALTY": "-0.5"}):
            self.assertAlmostEqual(game_result(board, None)[0], -0.5)

    def test_search_terminal_uses_penalty_for_side_to_move(self) -> None:
        board, _ = play("4k1n1/8/8/8/8/8/8/3QK1N1 w - - 0 1", KNIGHT_SHUFFLE)
        self.assertEqual(
            board.outcome(claim_draw=True).termination,
            chess.Termination.THREEFOLD_REPETITION,
        )
        self.assertAlmostEqual(terminal_value(board), PENALTY)


class TwofoldPruningTests(unittest.TestCase):
    def test_twofold_inside_the_tree_is_terminal_with_penalty(self) -> None:
        board, _ = play(KQK_SHUFFLE[0], KQK_SHUFFLE[1][:4])
        self.assertTrue(board.is_repetition(2))
        self.assertIsNone(board.outcome(claim_draw=True))
        self.assertAlmostEqual(tree_leaf_value(board), PENALTY)
        board.pop()
        self.assertIsNone(tree_leaf_value(board))

    def test_root_position_is_never_pruned_for_twofold(self) -> None:
        board, _ = play(KQK_SHUFFLE[0], KQK_SHUFFLE[1][:4])
        self.assertIsNone(tree_leaf_value(board, at_root=True))

    def test_fresh_positions_are_not_terminal(self) -> None:
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertIsNone(tree_leaf_value(board))


class DrawTests(unittest.TestCase):
    def test_full_game_at_the_cap_is_zero_even_a_queen_up(self) -> None:
        board, records = play(
            "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ["g1f3", "g8f6", "f3g1"],
        )
        data = label_game(records, board, stop="MOVE_LIMIT")
        self.assertEqual(len(data), len(records))
        self.assertTrue(all(z == 0.0 for z in zs(data)))
        self.assertEqual(game_result(board, "MOVE_LIMIT")[2], "MOVE_LIMIT")

    def test_stalemate_and_insufficient_material_stay_zero(self) -> None:
        stale, _ = play("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", ["f1f7"])
        self.assertTrue(stale.is_stalemate())
        self.assertEqual(game_result(stale, None)[0], 0.0)
        self.assertEqual(terminal_value(stale), 0.0)
        bare = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")
        self.assertTrue(bare.is_insufficient_material())
        self.assertEqual(game_result(bare, None)[0], 0.0)
        self.assertEqual(terminal_value(bare), 0.0)


class CurriculumKeepTests(unittest.TestCase):
    def test_curriculum_game_at_the_cap_is_kept(self) -> None:
        board, records = play(KQK_SHUFFLE[0], KQK_SHUFFLE[1][:3])
        data = label_game(records, board, stop="MOVE_LIMIT", practice=True)
        self.assertEqual(len(data), len(records))
        self.assertTrue(all(z == 0.0 for z in zs(data)))

    def test_curriculum_stalemate_is_kept(self) -> None:
        board, records = play("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", ["f1f7"])
        self.assertTrue(board.is_stalemate())
        data = label_game(records, board, stop=None, practice=True)
        self.assertEqual(len(data), len(records))

    def test_curriculum_mate_is_kept(self) -> None:
        board, records = play("7k/8/5K2/8/8/8/8/Q7 w - - 0 1", ["f6g6", "h8g8", "a1a8"])
        self.assertTrue(board.is_checkmate())
        data = label_game(records, board, stop=None, practice=True)
        self.assertEqual(len(data), 3)
        np.testing.assert_allclose(zs(data), [1.0, -1.0, 1.0], rtol=1e-6)


class NoHumanValuesTests(unittest.TestCase):
    def test_label_pipeline_has_no_piece_values(self) -> None:
        for module in (labels, worker, arena, search):
            source = inspect.getsource(module)
            for banned in ("material_white", "material_value", "tanh", "_VALUE", "CONTEMPT", "ADJUDICATE", "score_game"):
                self.assertNotIn(banned, source, f"{banned} in {module.__name__}")
        score_module = Path(__file__).parents[1] / "src/tabula/game/score.py"
        self.assertFalse(score_module.exists(), "human material scoring module still exists")


class ZeroOracleTests(unittest.TestCase):
    def test_training_modules_never_touch_a_tablebase(self) -> None:
        for module in (labels, worker, practice, search, arena, train, pipeline, runtime):
            source = inspect.getsource(module).lower()
            for banned in ("syzygy", "tablebase", "evaluate.oracle", "import oracle"):
                self.assertNotIn(banned, source, f"{banned} in {module.__name__}")


class ReplayBufferTests(unittest.TestCase):
    def test_buffer_tensors_hold_the_flat_targets(self) -> None:
        mate_board, mate_records = play(*BACK_RANK)
        draw_board, draw_records = play(chess.STARTING_FEN, KNIGHT_SHUFFLE)
        stale_board, stale_records = play("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", ["f1f7"])
        data = (
            label_game(mate_records, mate_board, None)
            + label_game(draw_records, draw_board, None)
            + label_game(stale_records, stale_board, None, practice=True)
        )
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(data, os.path.join(tmp, "selfplay_test.pt"))
            buffer = SelfPlayDataset(tmp)

        z = buffer.zs.reshape(-1).numpy()
        self.assertEqual(len(z), len(mate_records) + len(draw_records) + len(stale_records))
        np.testing.assert_allclose(z[:3], [1.0, -1.0, 1.0], rtol=1e-6)
        np.testing.assert_allclose(np.abs(z[3:11]), [abs(PENALTY)] * 8, rtol=1e-6)
        self.assertFalse(np.any(np.isclose(np.abs(z), 0.35)))
        self.assertTrue(np.all(buffer.weights.numpy() > 0.0))


if __name__ == "__main__":
    unittest.main()

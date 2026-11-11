"""Policy 6: the label math is checked on tiny games before any training run.

Each case builds a short game by hand, labels it, and inspects the replay
buffer the way train.py loads it.
"""

import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import chess
import numpy as np
import torch

from tabula.game.encoder import encode_board
from tabula.evaluate import arena
from tabula.mcts import search
from tabula.mcts.search import terminal_value
from tabula.selfplay import labels, worker
from tabula.selfplay.labels import GAMMA, Record, game_result, label_game, open_tablebase, tablebase_value
from tabula.train.train import SelfPlayDataset

TABLEBASE = open_tablebase()


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


class QuickMateTests(unittest.TestCase):
    def test_mate_targets_are_discounted_by_plies_to_mate(self) -> None:
        board, records = play(*BACK_RANK)
        self.assertTrue(board.is_checkmate())
        data = label_game(records, board, stop=None, practice=False)
        expected = [GAMMA ** 3, -GAMMA ** 2, GAMMA ** 1]
        np.testing.assert_allclose(zs(data), expected, rtol=1e-6)

    def test_faster_mate_is_worth_more(self) -> None:
        board, records = play(*BACK_RANK)
        data = label_game(records, board, stop=None, practice=False)
        self.assertGreater(zs(data)[2], zs(data)[0])


class DrawTests(unittest.TestCase):
    def test_threefold_shuffle_is_zero_and_kept(self) -> None:
        board, records = play(chess.STARTING_FEN, KNIGHT_SHUFFLE)
        outcome = board.outcome(claim_draw=True)
        self.assertEqual(outcome.termination, chess.Termination.THREEFOLD_REPETITION)
        data = label_game(records, board, stop=None, practice=False)
        self.assertEqual(len(data), len(records))
        self.assertTrue(all(z == 0.0 for z in zs(data)))

    def test_full_game_at_the_cap_is_zero_even_a_queen_up(self) -> None:
        # Black has no queen; White does. The label must not care.
        board, records = play(
            "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ["g1f3", "g8f6", "f3g1"],
        )
        data = label_game(records, board, stop="MOVE_LIMIT", practice=False)
        self.assertEqual(len(data), len(records))
        self.assertTrue(all(z == 0.0 for z in zs(data)))
        self.assertEqual(game_result(board, "MOVE_LIMIT", practice=False)[2], "MOVE_LIMIT")

    def test_search_repetition_is_zero_even_with_unequal_pieces(self) -> None:
        board, _ = play(
            "4k1n1/8/8/8/8/8/8/3QK1N1 w - - 0 1",
            KNIGHT_SHUFFLE,
        )
        self.assertEqual(
            board.outcome(claim_draw=True).termination,
            chess.Termination.THREEFOLD_REPETITION,
        )
        self.assertEqual(terminal_value(board), 0.0)


class StalledPracticeTests(unittest.TestCase):
    def test_practice_game_at_the_cap_is_dropped(self) -> None:
        board, records = play(KQK_SHUFFLE[0], KQK_SHUFFLE[1][:3])
        self.assertEqual(label_game(records, board, stop="MOVE_LIMIT", practice=True), [])

    def test_practice_repetition_is_dropped(self) -> None:
        board, records = play(*KQK_SHUFFLE)
        self.assertIsNotNone(board.outcome(claim_draw=True))
        self.assertEqual(label_game(records, board, stop=None, practice=True), [])

    def test_practice_stalemate_is_dropped(self) -> None:
        board, records = play("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", ["f1f7"])
        self.assertTrue(board.is_stalemate())
        self.assertEqual(label_game(records, board, stop=None, practice=True), [])

    def test_practice_mate_is_kept(self) -> None:
        board, records = play("7k/8/5K2/8/8/8/8/Q7 w - - 0 1", ["f6g6", "h8g8", "a1a8"])
        self.assertTrue(board.is_checkmate())
        data = label_game(records, board, stop=None, practice=True)
        self.assertEqual(len(data), 3)


class NoHumanValuesTests(unittest.TestCase):
    def test_label_pipeline_has_no_piece_values(self) -> None:
        for module in (labels, worker, arena, search):
            source = inspect.getsource(module)
            for banned in ("material_white", "material_value", "tanh", "_VALUE", "CONTEMPT", "ADJUDICATE", "score_game"):
                self.assertNotIn(banned, source, f"{banned} in {module.__name__}")
        score_module = Path(__file__).parents[1] / "src/tabula/game/score.py"
        self.assertFalse(score_module.exists(), "human material scoring module still exists")


class ReplayBufferTests(unittest.TestCase):
    def test_buffer_tensors_hold_the_honest_targets(self) -> None:
        mate_board, mate_records = play(*BACK_RANK)
        draw_board, draw_records = play(chess.STARTING_FEN, KNIGHT_SHUFFLE)
        stall_board, stall_records = play(*KQK_SHUFFLE)
        data = (
            label_game(mate_records, mate_board, None, practice=False)
            + label_game(draw_records, draw_board, None, practice=False)
            + label_game(stall_records, stall_board, None, practice=True)
        )
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(data, os.path.join(tmp, "selfplay_test.pt"))
            buffer = SelfPlayDataset(tmp)

        z = buffer.zs.reshape(-1).numpy()
        self.assertEqual(len(z), len(mate_records) + len(draw_records))
        np.testing.assert_allclose(z[:3], [GAMMA ** 3, -GAMMA ** 2, GAMMA], rtol=1e-6)
        self.assertTrue(np.all(z[3:] == 0.0))
        self.assertFalse(np.any(np.isclose(np.abs(z), 0.35)))
        self.assertTrue(np.all(buffer.weights.numpy() > 0.0))


class SyzygyConfigurationTests(unittest.TestCase):
    def test_syzygy_path_environment_variable_takes_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "KQvK.rtbw").touch()
            with (
                mock.patch.dict(os.environ, {"SYZYGY_PATH": tmp}, clear=False),
                mock.patch.object(labels.chess.syzygy, "open_tablebase", return_value="tablebase") as opener,
            ):
                self.assertEqual(open_tablebase(), "tablebase")
        opener.assert_called_once_with(tmp)


@unittest.skipIf(TABLEBASE is None, "no Syzygy tables (syzygy/345 or $SYZYGY_PATH)")
class TablebaseTests(unittest.TestCase):
    def test_practice_mate_uses_dtz(self) -> None:
        board, records = play("7k/8/5K2/8/8/8/8/Q7 w - - 0 1", ["f6g6", "h8g8", "a1a8"])
        data = label_game(records, board, stop=None, practice=True, tablebase=TABLEBASE)
        for rec, z in zip(records, zs(data)):
            pos = chess.Board(rec.fen)
            wdl = TABLEBASE.probe_wdl(pos)
            dtz = TABLEBASE.probe_dtz(pos)
            self.assertAlmostEqual(z, np.sign(wdl) * GAMMA ** abs(dtz), places=6)
        self.assertGreater(zs(data)[0], 0.0)
        self.assertLess(zs(data)[1], 0.0)

    def test_bare_king_dtz_is_plies_to_mate(self) -> None:
        # One move from mate: DTZ must say so (Syzygy may round by one ply).
        pos = chess.Board("6k1/8/6K1/8/8/8/8/Q7 w - - 0 1")
        self.assertIn(TABLEBASE.probe_dtz(pos), (1, 2))

    def test_full_game_stops_and_labels_from_the_tables(self) -> None:
        # With the cap lowered to 3 pieces, Qxb2 is the ply that enters the tables,
        # so the recorded position before it is labelled from the tablebase result.
        with mock.patch.object(labels, "TABLEBASE_MAX_PIECES", 3):
            board, records = play("6k1/8/6K1/8/8/8/1r6/Q7 w - - 0 1", ["a1b2"])
            self.assertFalse(labels.in_tablebase(chess.Board(records[0].fen), TABLEBASE))
            self.assertTrue(labels.in_tablebase(board, TABLEBASE))
            white, _, term = game_result(board, "TABLEBASE", practice=False, tablebase=TABLEBASE)
            self.assertEqual((white, term), (1.0, "TABLEBASE"))
            data = label_game(records, board, stop="TABLEBASE", practice=False, tablebase=TABLEBASE)
        dtz_after = abs(TABLEBASE.probe_dtz(board))
        np.testing.assert_allclose(zs(data), [GAMMA ** (1 + dtz_after)], rtol=1e-6)

    def test_tablebase_draw_is_zero(self) -> None:
        # Rook pawn, defending king in the corner.
        pos = chess.Board("k7/8/8/8/8/8/P7/7K w - - 0 1")
        self.assertEqual(TABLEBASE.probe_wdl(pos), 0)
        self.assertEqual(tablebase_value(pos, TABLEBASE), 0.0)


if __name__ == "__main__":
    unittest.main()

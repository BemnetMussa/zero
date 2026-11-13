"""Offline Syzygy oracle: evaluation only, never training."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import chess

from tabula.evaluate import oracle

LOCAL_TABLES = Path(__file__).parents[1] / "syzygy/345"
TABLES = os.environ.get("SYZYGY_PATH") or (str(LOCAL_TABLES) if LOCAL_TABLES.is_dir() else "")


class OracleConfigurationTests(unittest.TestCase):
    def test_missing_directory_fails_loudly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a directory"):
            oracle.open_syzygy("/definitely/not/a/tablebase")

    def test_open_counts_files_and_probes_kqk_and_krk(self) -> None:
        class ProbeTablebase:
            def __init__(self):
                self.boards = []

            def probe_wdl(self, board):
                self.boards.append(board.copy())
                return 2

            def probe_dtz(self, _board):
                return 1

            def close(self):
                pass

        tablebase = ProbeTablebase()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "KQvK.rtbw").touch()
            Path(tmp, "KQvK.rtbz").touch()
            with mock.patch.object(oracle.chess.syzygy, "open_tablebase", return_value=tablebase):
                status = oracle.open_syzygy(tmp)
        self.assertIs(status.tablebase, tablebase)
        self.assertEqual((status.rtbw_count, status.rtbz_count), (1, 1))
        symbols = [{p.symbol().lower() for p in b.piece_map().values()} for b in tablebase.boards]
        self.assertIn({"k", "q"}, symbols)
        self.assertIn({"k", "r"}, symbols)

    def test_probe_failure_closes_tables(self) -> None:
        failing = mock.Mock()
        failing.probe_wdl.side_effect = KeyError("missing table")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "KQvK.rtbw").touch()
            Path(tmp, "KQvK.rtbz").touch()
            with (
                mock.patch.object(oracle.chess.syzygy, "open_tablebase", return_value=failing),
                self.assertRaisesRegex(RuntimeError, "KQK/KRK probe failed"),
            ):
                oracle.open_syzygy(tmp)
        failing.close.assert_called_once()

    def test_optimal_dtz_ignores_positions_outside_the_tables(self) -> None:
        tablebase = mock.Mock()
        self.assertIsNone(oracle.optimal_dtz(chess.Board(), tablebase))
        tablebase.get_dtz.assert_not_called()


@unittest.skipUnless(TABLES, "no local Syzygy tables (syzygy/345 or $SYZYGY_PATH)")
class OracleRealTablesTests(unittest.TestCase):
    def test_kqk_mate_in_one_has_dtz_one(self) -> None:
        status = oracle.open_syzygy(TABLES)
        try:
            board = chess.Board(oracle.KQK_PROBE)
            self.assertEqual(oracle.optimal_dtz(board, status.tablebase), 1)
        finally:
            status.tablebase.close()


if __name__ == "__main__":
    unittest.main()

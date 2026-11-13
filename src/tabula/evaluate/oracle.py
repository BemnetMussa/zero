"""Offline Syzygy oracle for evaluation scripts only.

Zero policy 1: no external solver may adjudicate training games or write
training labels. Nothing under tabula.selfplay, tabula.train, tabula.mcts, or
tabula.pipeline may import this module. Offline exams use it to measure how
far the model's play is from perfect play.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.syzygy

TABLEBASE_MAX_PIECES = 5
KQK_PROBE = "6k1/8/6K1/8/8/8/8/Q7 w - - 0 1"
KRK_PROBE = "k7/8/1K6/8/8/8/8/2R5 w - - 0 1"


@dataclass(frozen=True)
class SyzygyStatus:
    tablebase: object
    path: Path
    rtbw_count: int
    rtbz_count: int
    probes: dict


def syzygy_file_counts(path: Path) -> tuple[int, int]:
    return len(list(path.glob("*.rtbw"))), len(list(path.glob("*.rtbz")))


def probe_syzygy(tablebase) -> dict:
    results = {}
    for name, fen in (("KQK", KQK_PROBE), ("KRK", KRK_PROBE)):
        board = chess.Board(fen)
        try:
            wdl = tablebase.probe_wdl(board)
            dtz = tablebase.probe_dtz(board)
        except Exception as exc:
            raise RuntimeError(f"Syzygy {name} probe failed: {exc}") from exc
        if wdl != 2 or dtz != 1:
            raise RuntimeError(
                f"Syzygy {name} probe returned WDL={wdl}, DTZ={dtz}; expected 2, 1"
            )
        results[name] = {"wdl": wdl, "dtz": dtz, "fen": fen}
    return results


def open_syzygy(path_value: str | os.PathLike[str]) -> SyzygyStatus:
    """Open and sanity-probe Syzygy tables, failing loudly on a bad mount."""
    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"Syzygy path is not a directory: {path}")
    rtbw_count, rtbz_count = syzygy_file_counts(path)
    if rtbw_count == 0 or rtbz_count == 0:
        raise RuntimeError(f"Syzygy path has no complete WDL/DTZ tables: {path}")
    try:
        tablebase = chess.syzygy.open_tablebase(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not open Syzygy tables at {path}: {exc}") from exc
    try:
        probes = probe_syzygy(tablebase)
    except Exception as exc:
        tablebase.close()
        raise RuntimeError(f"Syzygy KQK/KRK probe failed at {path}: {exc}") from exc
    return SyzygyStatus(
        tablebase=tablebase,
        path=path,
        rtbw_count=rtbw_count,
        rtbz_count=rtbz_count,
        probes=probes,
    )


def optimal_dtz(board: chess.Board, tablebase) -> int | None:
    """Perfect-play distance to zeroing for a covered position, else None."""
    if len(board.piece_map()) > TABLEBASE_MAX_PIECES or board.castling_rights:
        return None
    dtz = tablebase.get_dtz(board)
    return None if dtz is None else abs(int(dtz))

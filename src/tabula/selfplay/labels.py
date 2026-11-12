"""How a finished self-play game becomes replay-buffer entries.

Rules (Zero policies, 2026-11-11):

- A mate is worth GAMMA ** N for the side that delivers it, where N is the
  number of plies from the recorded position to the mate. The loser gets the
  negative. Faster mates are worth more.
- A draw the rules produced (stalemate, repetition, the 50-move rule,
  insufficient material) is 0.
- A full-board game that hits MAX_PLIES is 0 and stays in the buffer.
- A practice game that ends any way but mate is dropped whole.
- With Syzygy tables loaded, any recorded position with <= TABLEBASE_MAX_PIECES
  pieces takes the tablebase value instead: GAMMA ** |DTZ| with the WDL sign,
  0 for draws and for wins the 50-move rule turns into draws.
- A full-board game that enters the tablebase stops there ("TABLEBASE") and
  every earlier position is labelled from the tablebase result.

No piece values anywhere: the only inputs are the rules and the tables.
"""

import os
from dataclasses import dataclass

import chess
import chess.syzygy
import numpy as np

GAMMA = 0.99
MAX_PLIES = 200
TABLEBASE_MAX_PIECES = 5
LATE_WEIGHT_FLOOR = 0.2

DEFAULT_SYZYGY_DIRS = (
    "syzygy/345",
    "/kaggle/input/syzygy-345",
)


@dataclass
class Record:
    """One full-search turn, saved while the game is still running."""

    state: np.ndarray
    pi: np.ndarray
    turn: chess.Color
    ply: int
    v_prior: float
    q_root: float
    fen: str


def syzygy_dirs():
    """Configured path first, then portable local/Kaggle fallbacks."""
    return tuple(path for path in (
        os.environ.get("SYZYGY_PATH", ""),
        os.environ.get("ZERO_SYZYGY", ""),  # legacy name
        *DEFAULT_SYZYGY_DIRS,
    ) if path)


def open_tablebase(dirs=None):
    """First directory that holds Syzygy files, or None."""
    if dirs is None:
        dirs = syzygy_dirs()
    for path in dirs:
        if path and os.path.isdir(path) and any(f.endswith(".rtbw") for f in os.listdir(path)):
            return chess.syzygy.open_tablebase(path)
    return None


def _tablebase_eligible(board, tablebase) -> bool:
    return (
        tablebase is not None
        and len(board.piece_map()) <= TABLEBASE_MAX_PIECES
        and not board.castling_rights
    )


def tablebase_value(board, tablebase):
    """Side-to-move value from the tables, or None if the position is not covered."""
    if not _tablebase_eligible(board, tablebase):
        return None
    wdl = tablebase.get_wdl(board)
    if wdl is None:
        return None
    if wdl in (-1, 0, 1):
        return 0.0
    dtz = tablebase.get_dtz(board)
    if dtz is None:
        return None
    sign = 1.0 if wdl > 0 else -1.0
    return sign * GAMMA ** max(1, abs(dtz))


def in_tablebase(board, tablebase) -> bool:
    """Whether this exact position has enough WDL/DTZ data to adjudicate."""
    return tablebase_value(board, tablebase) is not None


def game_result(board, stop, practice, tablebase=None):
    """White's undiscounted result, a short label, and why the game ended.

    `stop` is set when the loop ended the game itself: "MOVE_LIMIT" or "TABLEBASE".
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is not None and outcome.winner is not None:
        white = 1.0 if outcome.winner == chess.WHITE else -1.0
        return white, "WHITE WINS" if white > 0 else "BLACK WINS", outcome.termination.name

    termination = outcome.termination.name if outcome is not None else (stop or "UNFINISHED")
    if practice:
        return 0.0, "NO MATE", termination

    if stop == "TABLEBASE" and outcome is None:
        value = tablebase_value(board, tablebase)
        if value:
            white = 1.0 if (value > 0) == (board.turn == chess.WHITE) else -1.0
            return white, "WHITE WINS" if white > 0 else "BLACK WINS", "TABLEBASE"
        return 0.0, "DRAW", "TABLEBASE"

    return 0.0, "DRAW", termination


def label_game(records, board, stop, practice, tablebase=None):
    """Turn a finished game into (s, π, z, w) tuples for the replay buffer.

    `board` is the final board with its move stack; each record's `ply` counts
    from the board the game started on.
    """
    white, _, termination = game_result(board, stop, practice, tablebase)
    if practice and termination != "CHECKMATE":
        return []

    plies = len(board.move_stack)
    # Plies from the final board to the end of the game: 0 after a real mate,
    # the tablebase distance when the game was stopped on entering the tables.
    tail = 0
    if termination == "TABLEBASE" and white != 0.0:
        tail = abs(tablebase.get_dtz(board))

    training_data = []
    total_plies = max(plies, 1)
    for rec in records:
        z = tablebase_value(chess.Board(rec.fen), tablebase)
        if z is None:
            if white == 0.0:
                z = 0.0
            else:
                side = white if rec.turn == chess.WHITE else -white
                z = side * GAMMA ** (plies - rec.ply + tail)
        late = LATE_WEIGHT_FLOOR + (1.0 - LATE_WEIGHT_FLOOR) * (rec.ply / total_plies)
        surprise = 1.0 + abs(rec.v_prior - rec.q_root)
        training_data.append((
            rec.state,
            rec.pi,
            np.array([z], dtype=np.float32),
            np.array([late * surprise], dtype=np.float32),
        ))
    return training_data

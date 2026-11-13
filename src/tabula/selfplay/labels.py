"""How a finished self-play game becomes replay-buffer entries.

Rules (Zero policies, 2026-11-13 — zero oracles, bare rules only):

- A mate is worth GAMMA ** N for the side that delivers it, where N is the
  number of plies from the recorded position to the mate. The loser gets the
  negative. Faster mates are worth more.
- Stalemate, the 50-move rule, and insufficient material are 0.
- Threefold / fivefold repetition costs the side that initiated the loop the
  repetition penalty (default -0.20, env TABULA_REPETITION_PENALTY); the other
  side gets the opposite sign. The initiator is the side to move in the
  repeated position: it kept choosing to leave that position on the same cycle.
- A full-board game that hits MAX_PLIES is 0 and stays in the buffer.
- A practice game is kept only if it ends in mate or repetition; anything else
  (stalemate, insufficient material, 50-move, cap) is dropped whole.

No piece values and no external solvers: the only input is the game outcome.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import chess
import numpy as np

GAMMA = 0.99
MAX_PLIES = 200
LATE_WEIGHT_FLOOR = 0.2
DEFAULT_REPETITION_PENALTY = -0.20
REPETITION_TERMINATIONS = frozenset({"THREEFOLD_REPETITION", "FIVEFOLD_REPETITION"})
PRACTICE_KEEP_TERMINATIONS = frozenset({"CHECKMATE"}) | REPETITION_TERMINATIONS


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


def repetition_penalty() -> float:
    """Value charged to the side that forces a repetition."""
    raw = os.environ.get("TABULA_REPETITION_PENALTY", "")
    return DEFAULT_REPETITION_PENALTY if raw == "" else float(raw)


def repetition_offender(board: chess.Board, reached: int = 3) -> chess.Color:
    """Side that initiated the repetition: the side to move in the repeated position.

    If the current position has occurred `reached` times, that is the side to
    move now. Otherwise the claim rests on a move the side to move could make,
    and the repeated position after it has the opponent to move.
    """
    if board.move_stack and board.is_repetition(reached):
        return board.turn
    return not board.turn


def white_repetition_value(board: chess.Board, reached: int = 3) -> float:
    """White's result for a repetition ending."""
    penalty = repetition_penalty()
    return penalty if repetition_offender(board, reached) == chess.WHITE else -penalty


def game_result(board, stop, practice):
    """White's result, a short label, and why the game ended.

    `stop` is "MOVE_LIMIT" when the loop ended the game itself.
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is not None and outcome.winner is not None:
        white = 1.0 if outcome.winner == chess.WHITE else -1.0
        return white, "WHITE WINS" if white > 0 else "BLACK WINS", outcome.termination.name

    termination = outcome.termination.name if outcome is not None else (stop or "UNFINISHED")
    if termination in REPETITION_TERMINATIONS:
        return white_repetition_value(board), "DRAW", termination
    if practice:
        return 0.0, "NO MATE", termination
    return 0.0, "DRAW", termination


def label_game(records, board, stop, practice):
    """Turn a finished game into (s, π, z, w) tuples for the replay buffer.

    `board` is the final board with its move stack; each record's `ply` counts
    from the board the game started on.
    """
    white, _, termination = game_result(board, stop, practice)
    if practice and termination not in PRACTICE_KEEP_TERMINATIONS:
        return []

    plies = len(board.move_stack)
    training_data = []
    total_plies = max(plies, 1)
    for rec in records:
        if white == 0.0:
            z = 0.0
        else:
            side = white if rec.turn == chess.WHITE else -white
            z = side * GAMMA ** (plies - rec.ply)
        late = LATE_WEIGHT_FLOOR + (1.0 - LATE_WEIGHT_FLOOR) * (rec.ply / total_plies)
        surprise = 1.0 + abs(rec.v_prior - rec.q_root)
        training_data.append((
            rec.state,
            rec.pi,
            np.array([z], dtype=np.float32),
            np.array([late * surprise], dtype=np.float32),
        ))
    return training_data

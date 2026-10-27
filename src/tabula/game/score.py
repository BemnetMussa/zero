"""How a self-play game becomes a value target.

A mate is +1 or -1 for White. A draw the rules actually produced (stalemate,
repetition, the 50-move rule, insufficient material) stays 0.

A game that never finished is different. Capping it at 0 teaches the value
head that a queen up and a dead draw are the same position. Those games are
scored from the material left on the board, squashed into (-1, 1).
"""

import chess
import numpy as np

# A queen is 9. Dividing by 6 puts a queen near tanh(1.5) ≈ 0.9, a rook near 0.7.
MATERIAL_SCALE = 6.0
# Rook plus a pawn, or a queen down a couple of pawns. Held this far, the game is decided
# for a network that still cannot find mate.
ADJUDICATE_LEAD = 7
ADJUDICATE_PLIES = 6
# v0.3: a winning side that repeats into a draw is hiding, not drawing fairly.
CONTEMPT_LEAD = 3
REPETITION_CONTEMPT = 0.35

_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_white(board) -> int:
    """White's material minus Black's, in pawn units. Kings are ignored."""
    score = 0
    for piece_type, value in _VALUE.items():
        score += value * (
            len(board.pieces(piece_type, chess.WHITE))
            - len(board.pieces(piece_type, chess.BLACK))
        )
    return score


def material_value(board) -> float:
    """White's material, squashed into (-1, 1)."""
    return float(np.tanh(material_white(board) / MATERIAL_SCALE))


def is_repetition_draw(outcome) -> bool:
    if outcome is None or outcome.winner is not None:
        return False
    term = outcome.termination
    return term in (chess.Termination.THREEFOLD_REPETITION, chess.Termination.FIVEFOLD_REPETITION)


def repetition_draw_reward(board) -> float:
    """White-perspective reward for a repetition draw.

    Even games stay 0. If one side leads by CONTEMPT_LEAD pawns or more,
    that side is penalized so hiding in a threefold is not a free 0.0.
    """
    lead = material_white(board)
    if abs(lead) < CONTEMPT_LEAD:
        return 0.0
    return -REPETITION_CONTEMPT if lead > 0 else REPETITION_CONTEMPT


def score_game(board, outcome, stop, adjudicate_scale=1.0):
    """Reward from White's point of view, plus a short result label and why it ended.

    `stop` is set when the loop ended the game itself ("MATERIAL" or "MOVE_LIMIT").
    A result python-chess already decided wins over that.

    `adjudicate_scale` < 1.0 softens material/move-limit cutoffs so they do not
    count like full checkmates in the arena gate (v0.2). Mates and rules draws
    are unchanged.
    """
    if outcome is not None and outcome.winner is not None:
        reward = 1.0 if outcome.winner == chess.WHITE else -1.0
        result = "WHITE WINS" if reward > 0 else "BLACK WINS"
        return reward, result, outcome.termination.name

    if outcome is not None and outcome.winner is None:
        if is_repetition_draw(outcome):
            reward = repetition_draw_reward(board)
            return reward, "DRAW", outcome.termination.name
        return 0.0, "DRAW", outcome.termination.name

    reward = material_value(board) * float(adjudicate_scale)
    if reward > 0.15:
        result = "WHITE AHEAD"
    elif reward < -0.15:
        result = "BLACK AHEAD"
    else:
        result = "LEVEL"
    return reward, result, stop or "UNFINISHED"

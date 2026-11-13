"""Close-to-mate curriculum boards for decisive early training signals."""

import random

import chess

MATE_IN_ONE_FENS = (
    "8/8/8/8/8/2k5/5q2/K7 b - - 0 1",
    "2r5/8/8/8/5k1K/8/8/8 b - - 0 1",
    "8/8/8/1Q6/8/2K5/8/2k5 w - - 0 1",
    "8/7R/8/8/8/K7/8/k7 w - - 0 1",
)

MATE_IN_TWO_FENS = (
    "8/3q4/8/8/8/8/4k3/7K b - - 0 1",
    "8/8/7Q/8/8/3K4/8/4k3 w - - 0 1",
    "8/8/4K3/7k/5R2/8/8/8 w - - 0 1",
    "2k5/8/8/4K3/8/8/1R6/8 w - - 0 1",
)

CORNERED_KQK_FENS = (
    "7k/8/5K2/8/8/8/8/Q7 w - - 0 1",
    "k7/8/2K5/8/8/8/8/7Q w - - 0 1",
    "7K/8/5k2/8/8/8/8/q7 b - - 0 1",
    "K7/8/2k5/8/8/8/8/7q b - - 0 1",
)

_SYMMETRIES = (
    None,
    chess.flip_horizontal,
    chess.flip_vertical,
    chess.flip_diagonal,
    chess.flip_anti_diagonal,
)


def _curriculum_group(rng):
    draw = rng.random()
    if draw < 0.45:
        return MATE_IN_ONE_FENS
    if draw < 0.80:
        return MATE_IN_TWO_FENS
    return CORNERED_KQK_FENS


def _randomize_board(board, rng):
    transform = rng.choice(_SYMMETRIES)
    if transform is not None:
        board = board.transform(transform)
    if rng.random() < 0.5:
        board = board.mirror()
    return board


def random_practice_board(rng=random):
    """Sample a symmetric mate-in-1, mate-in-2, or cornered-king setup.

    The strong side always moves. A fast mate from these boards takes one or
    three plies, so successful labels begin near +0.99 or +0.97.
    """
    board = chess.Board(rng.choice(_curriculum_group(rng)))
    board = _randomize_board(board, rng)
    if not board.is_valid() or board.outcome(claim_draw=True) is not None:
        raise RuntimeError(f"Invalid curriculum board generated: {board.fen()}")
    return board, board.turn

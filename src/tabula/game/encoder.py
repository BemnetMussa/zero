"""Board encoding for Tabula — AlphaZero side-to-move perspective.

Channels 0-5 = side-to-move pieces, 6-11 = opponent, 12 = colour of side to move
(1 = White, 0 = Black). When Black is to move the board is rotated 180° so "we"
always sit at the bottom and pawns move in the +rank direction in tensor space.
"""

import numpy as np
import chess

# 180° rotation: a1 <-> h8. Used when Black is to move.
_FLIP = 63

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def needs_orient(board) -> bool:
    """True when the board must be rotated for the side-to-move view."""
    return board.turn == chess.BLACK


def orient_square(square: int, flip: bool) -> int:
    return square ^ _FLIP if flip else square


def orient_move(move: chess.Move, flip: bool) -> chess.Move:
    """Map a real-board move into (or out of) the oriented policy tensor space."""
    if not flip:
        return move
    return chess.Move(
        orient_square(move.from_square, True),
        orient_square(move.to_square, True),
        promotion=move.promotion,
    )


def encode_board(board) -> np.ndarray:
    """
    (13, 8, 8) float32 tensor from the side-to-move's perspective.

    - 0-5: our pieces (P, N, B, R, Q, K)
    - 6-11: opponent pieces
    - 12: our colour (1 = White to move, 0 = Black to move)
    """
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    flip = needs_orient(board)
    us = board.turn
    them = not us

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        oriented = orient_square(square, flip)
        rank = chess.square_rank(oriented)
        file = chess.square_file(oriented)
        plane = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == them:
            plane += 6
        tensor[plane, rank, file] = 1.0

    if us == chess.WHITE:
        tensor[12, :, :] = 1.0

    return tensor


if __name__ == "__main__":
    b = chess.Board()
    enc = encode_board(b)
    print("shape", enc.shape)
    print("white to move, our pawns (layer 0) rank 1 sum:", enc[0, 1, :].sum())
    b.push_san("e4")
    enc_b = encode_board(b)
    print("black to move after e4 — our (black) pawns should sit on tensor rank 1")
    print("layer0 rank1 sum", enc_b[0, 1, :].sum(), "colour plane", enc_b[12].mean())

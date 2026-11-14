"""Board encoding for Zero — AlphaZero side-to-move perspective with rule state.

Channels 0-5 = side-to-move pieces, 6-11 = opponent, 12 = colour of side to move
(1 = White, 0 = Black). Channels 13-19 carry repetition, castling rights, and the
50-move clock so the network is not forced to invent a POMDP for draws.
When Black is to move the board is rotated 180° so "we" always sit at the bottom
and pawns move in the +rank direction in tensor space.
"""

import numpy as np
import chess

# 180° rotation: a1 <-> h8. Used when Black is to move.
_FLIP = 63

INPUT_PLANES = 20

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
    (20, 8, 8) float32 tensor from the side-to-move's perspective.

    - 0-5: our pieces (P, N, B, R, Q, K)
    - 6-11: opponent pieces
    - 12: our colour (1 = White to move, 0 = Black to move)
    - 13: position already seen once (is_repetition(2))
    - 14: position already seen twice (is_repetition(3))
    - 15-18: STM K-castle, STM Q-castle, opp K-castle, opp Q-castle
    - 19: halfmove clock / 100, capped at 1.0
    """
    tensor = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
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

    if board.is_repetition(2):
        tensor[13, :, :] = 1.0
    if board.is_repetition(3):
        tensor[14, :, :] = 1.0

    # Castling rights from the side-to-move's perspective after orientation.
    # With a 180° flip the kingside/queenside bits stay with the same player.
    stm_k = board.has_kingside_castling_rights(us)
    stm_q = board.has_queenside_castling_rights(us)
    opp_k = board.has_kingside_castling_rights(them)
    opp_q = board.has_queenside_castling_rights(them)
    if stm_k:
        tensor[15, :, :] = 1.0
    if stm_q:
        tensor[16, :, :] = 1.0
    if opp_k:
        tensor[17, :, :] = 1.0
    if opp_q:
        tensor[18, :, :] = 1.0

    tensor[19, :, :] = min(board.halfmove_clock, 100) / 100.0

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

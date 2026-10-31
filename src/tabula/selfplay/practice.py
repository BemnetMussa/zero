"""Practice boards: start a self-play game close to a mate.

From the normal start, two equal nets repeat into a draw long before anyone
has to finish a game. A practice board gives one side a king plus a queen or
a rook against a bare king, so the only job left is to deliver mate.
"""

import random

import chess

PRACTICE_PIECES = (chess.QUEEN, chess.ROOK)


def random_practice_board(rng=random):
    """A legal K+Q vs K or K+R vs K board, the strong side to move.

    Returns (board, strong_color). The strong side moves first so its piece
    can never start hanging to the bare king.
    """
    while True:
        strong = rng.choice((chess.WHITE, chess.BLACK))
        piece = rng.choice(PRACTICE_PIECES)
        strong_king, weak_king, piece_square = rng.sample(chess.SQUARES, 3)

        board = chess.Board(None)
        board.set_piece_at(strong_king, chess.Piece(chess.KING, strong))
        board.set_piece_at(weak_king, chess.Piece(chess.KING, not strong))
        board.set_piece_at(piece_square, chess.Piece(piece, strong))
        board.turn = strong

        if board.is_valid() and board.outcome(claim_draw=True) is None:
            return board, strong

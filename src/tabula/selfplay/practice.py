"""Oracle-free procedural curriculum: a 4-tier piece staircase."""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess

# Tier weights for sample_curriculum_board.
TIER_WEIGHTS = {
    1: 0.20,  # 3–4 pieces
    2: 0.45,  # 6–10 pieces
    3: 0.25,  # 14–20 pieces
    4: 0.10,  # 32-piece start
}

TIER_PIECE_RANGES = {
    1: (3, 4),
    2: (6, 10),
    3: (14, 20),
    4: (32, 32),
}

# Material templates for Tier 1: (white_pieces, black_pieces) using piece types.
_TIER1_TEMPLATES = (
    ((chess.KING, chess.QUEEN), (chess.KING,)),
    ((chess.KING, chess.ROOK), (chess.KING,)),
    ((chess.KING, chess.PAWN), (chess.KING,)),
    ((chess.KING, chess.ROOK), (chess.KING, chess.PAWN)),
)

@dataclass(frozen=True)
class CurriculumStart:
    board: chess.Board
    tier: int
    piece_count: int


def _empty_board() -> chess.Board:
    board = chess.Board(None)
    board.clear()
    return board


def _place(board: chess.Board, piece_type: int, color: bool, square: int) -> None:
    board.set_piece_at(square, chess.Piece(piece_type, color))


def _random_empty_square(rng, occupied: set[int]) -> int:
    free = [sq for sq in chess.SQUARES if sq not in occupied]
    return rng.choice(free)


def _accept_live_position(board: chess.Board) -> bool:
    if not board.is_valid():
        return False
    if board.outcome(claim_draw=True) is not None:
        return False
    return True


def _generate_tier1(rng) -> chess.Board:
    white_set, black_set = rng.choice(_TIER1_TEMPLATES)
    # Randomly mirror colours so either side can be the stronger one.
    if rng.random() < 0.5:
        white_set, black_set = black_set, white_set

    for _ in range(200):
        board = _empty_board()
        occupied: set[int] = set()
        try:
            for piece_type in white_set:
                sq = _random_empty_square(rng, occupied)
                occupied.add(sq)
                _place(board, piece_type, chess.WHITE, sq)
            for piece_type in black_set:
                sq = _random_empty_square(rng, occupied)
                occupied.add(sq)
                _place(board, piece_type, chess.BLACK, sq)
        except IndexError:
            continue
        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        board.castling_rights = chess.BB_EMPTY
        board.halfmove_clock = 0
        board.fullmove_number = 1
        if _accept_live_position(board):
            return board
    raise RuntimeError("Failed to sample a legal Tier-1 board")


def _add_balanced_pieces(board: chess.Board, rng, target_count: int) -> None:
    """Grow a bare K vs K board toward target_count with near-balanced material."""
    occupied = set(board.piece_map())
    # Always start from kings already placed.
    while len(occupied) < target_count:
        remaining = target_count - len(occupied)
        # Prefer pairing the same piece type for both colours when two slots left.
        if remaining >= 2 and rng.random() < 0.7:
            piece_type = rng.choice((chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
            placed = 0
            for color in (chess.WHITE, chess.BLACK):
                for _ in range(40):
                    sq = _random_empty_square(rng, occupied)
                    # Pawns cannot sit on first/eighth rank.
                    if piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                        continue
                    _place(board, piece_type, color, sq)
                    occupied.add(sq)
                    placed += 1
                    break
            if placed < 2:
                # Roll back a half-pair if we failed.
                for sq, piece in list(board.piece_map().items()):
                    if sq in occupied and piece.piece_type == piece_type and piece.piece_type != chess.KING:
                        if placed == 1 and piece.color == chess.WHITE:
                            board.remove_piece_at(sq)
                            occupied.discard(sq)
                            placed = 0
                continue
            continue

        # Single ±1 imbalance: one extra pawn or minor for a random side.
        piece_type = rng.choice((chess.PAWN, chess.KNIGHT, chess.BISHOP))
        color = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        for _ in range(40):
            sq = _random_empty_square(rng, occupied)
            if piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            _place(board, piece_type, color, sq)
            occupied.add(sq)
            break


def _generate_tier2(rng) -> chess.Board:
    low, high = TIER_PIECE_RANGES[2]
    target = rng.randint(low, high)
    for _ in range(200):
        board = _empty_board()
        occupied: set[int] = set()
        wk = _random_empty_square(rng, occupied)
        occupied.add(wk)
        _place(board, chess.KING, chess.WHITE, wk)
        for _try in range(40):
            bk = _random_empty_square(rng, occupied)
            if chess.square_distance(wk, bk) > 1:
                occupied.add(bk)
                _place(board, chess.KING, chess.BLACK, bk)
                break
        else:
            continue
        _add_balanced_pieces(board, rng, target)
        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        board.castling_rights = chess.BB_EMPTY
        board.halfmove_clock = 0
        board.fullmove_number = 1
        if low <= len(board.piece_map()) <= high and _accept_live_position(board):
            return board
    raise RuntimeError("Failed to sample a legal Tier-2 board")


def _generate_tier3(rng) -> chess.Board:
    """Near-symmetrical middlegame by deleting pieces from the start position."""
    low, high = TIER_PIECE_RANGES[3]
    target = rng.randint(low, high)
    for _ in range(200):
        board = chess.Board()
        # Remove pairs (same type, both colours) to keep material near-symmetric.
        removable = [
            (sq, piece)
            for sq, piece in board.piece_map().items()
            if piece.piece_type != chess.KING
        ]
        rng.shuffle(removable)
        # Group by piece type for pair deletion.
        by_type: dict[int, list[tuple[int, chess.Piece]]] = {}
        for sq, piece in removable:
            by_type.setdefault(piece.piece_type, []).append((sq, piece))

        while len(board.piece_map()) > target:
            # Prefer deleting a white+black pair of the same type.
            candidates = [
                ptype
                for ptype, entries in by_type.items()
                if sum(1 for _, pc in entries if pc.color == chess.WHITE and board.piece_at(_) == pc) >= 1
                and sum(1 for _, pc in entries if pc.color == chess.BLACK and board.piece_at(_) == pc) >= 1
            ]
            # Recompute live candidates from the board.
            live = {}
            for sq, piece in board.piece_map().items():
                if piece.piece_type == chess.KING:
                    continue
                live.setdefault(piece.piece_type, {chess.WHITE: [], chess.BLACK: []})
                live[piece.piece_type][piece.color].append(sq)
            pair_types = [
                ptype
                for ptype, colours in live.items()
                if colours[chess.WHITE] and colours[chess.BLACK]
            ]
            if pair_types and len(board.piece_map()) - 2 >= low:
                ptype = rng.choice(pair_types)
                w_sq = rng.choice(live[ptype][chess.WHITE])
                b_sq = rng.choice(live[ptype][chess.BLACK])
                board.remove_piece_at(w_sq)
                board.remove_piece_at(b_sq)
                continue
            # Fall back to deleting one non-king piece (±1 imbalance).
            non_kings = [
                sq for sq, piece in board.piece_map().items() if piece.piece_type != chess.KING
            ]
            if not non_kings or len(board.piece_map()) <= low:
                break
            board.remove_piece_at(rng.choice(non_kings))

        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        # Clear castling if kings/rooks moved off start — python-chess handles rights.
        if not _accept_live_position(board):
            continue
        count = len(board.piece_map())
        if low <= count <= high:
            return board
    raise RuntimeError("Failed to sample a legal Tier-3 board")


def _generate_tier4(_rng) -> chess.Board:
    return chess.Board()


_GENERATORS = {
    1: _generate_tier1,
    2: _generate_tier2,
    3: _generate_tier3,
    4: _generate_tier4,
}


def choose_tier(rng=random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for tier, weight in TIER_WEIGHTS.items():
        cumulative += weight
        if draw < cumulative:
            return tier
    return 4


def sample_curriculum_board(rng=random, tier: int | None = None) -> CurriculumStart:
    """Sample a legal non-terminal board from the 4-tier piece staircase."""
    chosen = choose_tier(rng) if tier is None else int(tier)
    if chosen not in _GENERATORS:
        raise ValueError(f"Unknown curriculum tier: {chosen}")
    board = _GENERATORS[chosen](rng)
    return CurriculumStart(board=board, tier=chosen, piece_count=len(board.piece_map()))


def random_practice_board(rng=random, tier: int | None = None):
    """Backward-compatible wrapper: (board, side_to_move)."""
    start = sample_curriculum_board(rng, tier=tier)
    return start.board, start.board.turn


def sample_mate_exam_board(rng) -> tuple[chess.Board, chess.Color]:
    """Tier-1 K+Q or K+R vs K only — stable mate-exam material."""
    templates = (
        ((chess.KING, chess.QUEEN), (chess.KING,)),
        ((chess.KING, chess.ROOK), (chess.KING,)),
    )
    for _ in range(400):
        white_set, black_set = rng.choice(templates)
        if rng.random() < 0.5:
            white_set, black_set = black_set, white_set
        board = _empty_board()
        occupied: set[int] = set()
        try:
            for piece_type in white_set:
                sq = _random_empty_square(rng, occupied)
                occupied.add(sq)
                _place(board, piece_type, chess.WHITE, sq)
            for piece_type in black_set:
                sq = _random_empty_square(rng, occupied)
                occupied.add(sq)
                _place(board, piece_type, chess.BLACK, sq)
        except IndexError:
            continue
        # Strong side to move.
        white_extra = any(
            p.piece_type in (chess.QUEEN, chess.ROOK) for p in board.piece_map().values() if p.color == chess.WHITE
        )
        board.turn = chess.WHITE if white_extra else chess.BLACK
        board.castling_rights = chess.BB_EMPTY
        board.halfmove_clock = 0
        board.fullmove_number = 1
        if _accept_live_position(board) and len(board.piece_map()) == 3:
            return board, board.turn
    raise RuntimeError("Failed to sample a mate-exam board")

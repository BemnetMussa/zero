"""Zero vs Zero self-play under Pure AlphaZero search (Run #3)."""

from __future__ import annotations

import os
import time

import chess
import chess.pgn
import numpy as np
import torch

from tabula.checkpoint import load_model_weights
from tabula.game.action_encoder import ActionEncoder
from tabula.game.encoder import encode_board, needs_orient, orient_move
from tabula.mcts.search import MCTS, advance_root, visit_policy_target
from tabula.model.model import TabulaNet
from tabula.selfplay.labels import MAX_PLIES, Record, game_result, label_game
from tabula.selfplay.practice import sample_curriculum_board

DEFAULT_RES_BLOCKS = 8
DEFAULT_SIMULATIONS = 128

# Smooth temperature schedule (all game tiers).
TEMP_TAU_OPENING = 1.0
TEMP_TAU_MID = 0.35
TEMP_TAU_LATE = 0.10
TEMP_OPENING_PLIES = 30
TEMP_MID_PLIES = 60


class SelfPlayWorker:
    def __init__(
        self,
        model_path=None,
        num_simulations=DEFAULT_SIMULATIONS,
        num_res_blocks=DEFAULT_RES_BLOCKS,
        model_state=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"Initializing TabulaNet for Self-Play on {self.device} "
            f"({num_res_blocks} ResBlocks, {num_simulations} sims)..."
        )
        self.model = TabulaNet(num_res_blocks=num_res_blocks).to(self.device)

        if model_state is not None:
            load_model_weights(self.model, model_state)
        elif model_path and os.path.exists(model_path):
            raw = torch.load(model_path, map_location=self.device, weights_only=True)
            # Full checkpoints nest weights under model_state.
            if isinstance(raw, dict) and "model_state" in raw:
                load_model_weights(self.model, raw["model_state"])
            else:
                load_model_weights(self.model, raw)
        self.model.eval()

        self.mcts = MCTS(self.model, num_simulations=num_simulations, c_puct=1.25)
        self.action_encoder = ActionEncoder()
        self.last_termination = None
        self.last_root_noise_calls = 0
        self.last_telemetry = None
        self.last_pgn = None
        self.last_tier = None

    def play_game(
        self,
        render=False,
        delay=0.5,
        live=None,
        hold=0.0,
        practice=False,
        tier: int | None = None,
    ):
        """Play one curriculum game; every ply is a full noisy search.

        `practice` is accepted for call-site compatibility. Run #3 always samples
        from the 4-tier piece staircase (tier 4 = standard start).
        """
        del practice
        start = sample_curriculum_board(tier=tier)
        board = start.board
        tier_id = start.tier
        start_pieces = start.piece_count
        start_fen = board.fen()

        records = []
        last_move = None
        captures = 0
        pawn_pushes = 0
        root = None

        if not render:
            print(f"Starting Tier {tier_id} game ({start_pieces} pieces) from {start_fen}")

        moves = 0
        stop = None
        root_noise_calls = 0
        while board.outcome(claim_draw=True) is None:
            if render:
                os.system("cls" if os.name == "nt" else "clear")
                print(f"--- Zero Self-Play (Move {moves}) ---")
                print(board.unicode())
                time.sleep(delay)

            _, v_prior = self.mcts.predict(board, encode_board)
            root = self.mcts.search(
                board,
                encode_board,
                add_noise=True,
                num_simulations=self.mcts.num_simulations,
                root=root,
            )
            root_noise_calls += 1
            policy = visit_policy_target(root)

            total_visits = sum(child.visit_count for child in root.children.values())
            if total_visits <= 0 or not policy:
                break

            flip = needs_orient(board)
            pi = np.zeros(4672, dtype=np.float32)
            for move, prob in policy.items():
                idx = self.action_encoder.encode(orient_move(move, flip))
                pi[idx] = prob

            tau = temperature_for_ply(moves)
            selection = f"tau={tau:.2f}"
            played = choose_move(root, tau)

            if live is not None:
                live.show_search(
                    phase="self-play",
                    headline=f"Self-play · ply {moves} · {selection}",
                    board=board,
                    root=root,
                    ply=moves,
                    selection=selection,
                    last_move=last_move,
                )
                if hold:
                    time.sleep(hold)

            records.append(
                Record(
                    state=encode_board(board),
                    pi=pi,
                    turn=board.turn,
                    ply=moves,
                    v_prior=float(v_prior),
                    q_root=float(root.q_value),
                    fen=board.fen(),
                )
            )

            if board.is_capture(played):
                captures += 1
            piece = board.piece_at(played.from_square)
            if piece is not None and piece.piece_type == chess.PAWN:
                pawn_pushes += 1

            board.push(played)
            last_move = played
            moves += 1

            # Subtree reuse: advance root to the played child.
            root = advance_root(root, played)

            if not render and moves % 10 == 0:
                q = root.q_value if root is not None else 0.0
                print(f"Move {moves}... (Eval: {q:.2f})")

            if moves >= MAX_PLIES:
                stop = "MOVE_LIMIT"
                print(f"Hit the {MAX_PLIES}-ply cap.")
                break

        reward, result, termination = game_result(board, stop)
        self.last_termination = termination
        self.last_root_noise_calls = root_noise_calls
        end_pieces = len(board.piece_map())
        self.last_tier = tier_id
        self.last_telemetry = {
            "tier": tier_id,
            "termination": termination,
            "plies": moves,
            "pieces_start": start_pieces,
            "pieces_end": end_pieces,
            "captures": captures,
            "pawn_pushes": pawn_pushes,
            "result": result,
            "reward_white": reward,
            "start_fen": start_fen,
        }
        self.last_pgn = _board_to_pgn(board, start_fen, result, termination, tier_id)

        xray = (
            f"[Tier {tier_id}] | {termination} | plies={moves} | "
            f"pieces={start_pieces}->{end_pieces} | captures={captures} | "
            f"pawn_pushes={pawn_pushes}"
        )
        print(xray)

        if render:
            os.system("cls" if os.name == "nt" else "clear")
            print("--- Zero Self-Play (Game Over) ---")
            print(board.unicode())
            time.sleep(2.0)

        print(
            f"Game over after {moves} plies. {result} ({termination}). "
            f"Reward for White: {reward} | recorded {len(records)}/{moves} "
            f"(noisy_roots={root_noise_calls})"
        )
        if live is not None:
            winner = "WHITE" if reward > 0.05 else ("BLACK" if reward < -0.05 else "DRAW")
            live.record_game_outcome("self-play", winner, termination, moves)
            live.show_search(
                phase="self-play",
                headline=f"{result} · {termination.replace('_', ' ').lower()} · {moves} plies",
                board=board,
                root=None,
                ply=moves,
                selection=None,
                last_move=last_move,
                result=result,
                termination=termination,
            )
            if hold:
                time.sleep(max(hold, 1.0))

        return label_game(records, board, stop)


def temperature_for_ply(ply: int) -> float:
    if ply < TEMP_OPENING_PLIES:
        return TEMP_TAU_OPENING
    if ply < TEMP_MID_PLIES:
        return TEMP_TAU_MID
    return TEMP_TAU_LATE


def choose_move(root, tau: float):
    """Sample a root move with temperature tau over visit counts.

    Uses float64 + log-space softmax so late-game τ=0.10 cannot overflow when
    visit counts are large (visits ** (1/τ) in float32 would be unsafe).
    """
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    if visits.sum() <= 0:
        return moves[0]
    if tau <= 1e-8:
        return moves[int(np.argmax(visits))]
    # log(visits^(1/τ)) = log(visits)/τ — stable even for τ=0.10 and N≈200.
    log_visits = np.log(np.maximum(visits, 1e-12)) / tau
    log_visits -= log_visits.max()
    weights = np.exp(log_visits)
    weights /= weights.sum()
    return moves[int(np.random.choice(len(moves), p=weights))]


def _board_to_pgn(board, start_fen, result, termination, tier) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = f"Zero Run3 Tier {tier}"
    game.headers["Site"] = "self-play"
    game.headers["FEN"] = start_fen
    game.headers["SetUp"] = "1"
    game.headers["Termination"] = termination
    if result == "WHITE WINS":
        game.headers["Result"] = "1-0"
    elif result == "BLACK WINS":
        game.headers["Result"] = "0-1"
    else:
        game.headers["Result"] = "1/2-1/2"

    # Rebuild from start_fen + move stack.
    node = game
    replay = chess.Board(start_fen)
    for move in board.move_stack:
        node = node.add_variation(move)
        replay.push(move)
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter) + "\n"


def main():
    champ = "models/tabula_v0.2_champion.pth"
    model_path = champ if os.path.exists(champ) else None
    worker = SelfPlayWorker(model_path=model_path, num_simulations=DEFAULT_SIMULATIONS)

    print("--- GENERATING 1 TEST GAME ---")
    start_time = time.time()
    data = worker.play_game()
    os.makedirs("data", exist_ok=True)
    filename = f"data/selfplay_test_{int(time.time())}.pt"
    torch.save(data, filename)
    elapsed = time.time() - start_time
    print(f"\nSaved {len(data)} board positions to {filename} in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()

import os
import time
import numpy as np
import torch
import chess

from tabula.model.model import TabulaNet
from tabula.mcts.search import MCTS
from tabula.game.encoder import encode_board, needs_orient, orient_move
from tabula.game.action_encoder import ActionEncoder
from tabula.selfplay.labels import MAX_PLIES, Record, game_result, in_tablebase, label_game, open_tablebase
from tabula.selfplay.practice import random_practice_board

# v0.2 defaults (AlphaZero used 800; laptop budget stays far below that).
DEFAULT_RES_BLOCKS = 8
DEFAULT_SIMULATIONS = 80
# v0.3 playout cap (KataGo §3.1): most turns are cheap; only full turns train π.
FAST_SIMULATIONS = 16
FULL_SEARCH_P = 0.25

class SelfPlayWorker:
    def __init__(self, model_path=None, num_simulations=DEFAULT_SIMULATIONS, num_res_blocks=DEFAULT_RES_BLOCKS, tablebase=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing TabulaNet for Self-Play on {self.device} ({num_res_blocks} ResBlocks, {num_simulations} sims)...")
        self.model = TabulaNet(num_res_blocks=num_res_blocks).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.mcts = MCTS(self.model, num_simulations=num_simulations, c_puct=1.25)
        self.action_encoder = ActionEncoder()
        self.tablebase = tablebase if tablebase is not None else open_tablebase()
        print(f"Syzygy tablebase: {'loaded' if self.tablebase else 'not found (labels from game results only)'}")

    def play_game(self, render=False, delay=0.5, live=None, hold=0.0, sampling_moves=30, practice=False):
        """Plays a single game of Zero vs Zero and returns the training examples.

        v0.3 playout cap: a fraction FULL_SEARCH_P of turns run a full noisy
        search (N sims, forced playouts, pruned π recorded). The rest run a
        fast quiet search (n sims, no noise) just to move the game forward.

        practice=True starts from a K+Q or K+R vs K board instead of the normal
        start. Only mate counts as finishing there; anything else is dropped.
        A normal game stops when it enters the tablebase or hits MAX_PLIES.
        """
        if practice:
            board, practice_side = random_practice_board()
        else:
            board, practice_side = chess.Board(), None
        records = []
        last_move = None
        full_turns = 0
        fast_turns = 0

        if not render:
            if practice:
                print(f"Starting practice game from {board.fen()}")
            else:
                print("Starting new self-play game...")

        moves = 0
        stop = None
        root_noise_calls = 0
        while board.outcome(claim_draw=True) is None:
            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Zero Self-Play (Move {moves}) ---")
                print(board.unicode())
                time.sleep(delay)

            # Curriculum positions are short and need dense, decisive labels.
            # Normal games retain the playout cap.
            is_full = practice_side is not None or np.random.random() < FULL_SEARCH_P
            if is_full:
                _, v_prior = self.mcts.predict(board, encode_board)
                root_node = self.mcts.search(
                    board,
                    encode_board,
                    add_noise=True,
                    num_simulations=self.mcts.num_simulations,
                    force_playouts=True,
                )
                root_noise_calls += 1
                policy = self.mcts.pruned_policy_target(root_node)
                full_turns += 1
            else:
                v_prior = None
                root_node = self.mcts.search(
                    board,
                    encode_board,
                    add_noise=False,
                    num_simulations=FAST_SIMULATIONS,
                    force_playouts=False,
                )
                policy = None
                fast_turns += 1

            total_visits = sum(child.visit_count for child in root_node.children.values())
            if total_visits <= 0:
                break

            # Policy targets live in side-to-move coordinates (same as encode_board).
            flip = needs_orient(board)
            pi = np.zeros(4672, dtype=np.float32)
            if policy:
                for move, prob in policy.items():
                    idx = self.action_encoder.encode(orient_move(move, flip))
                    pi[idx] = prob

            # Normal self-play uses τ=1 visit sampling for every move in the
            # opening window, including cheap fast-search turns. Curriculum
            # games execute greedily so obvious mates are not sampled away.
            selection = _selection_mode(
                practice=practice_side is not None,
                ply=moves,
                sampling_moves=sampling_moves,
            )
            played = _choose_move(root_node, selection)

            if live is not None:
                live.show_search(
                    phase="self-play",
                    headline=f"Self-play · ply {moves} · {selection}",
                    board=board,
                    root=root_node,
                    ply=moves,
                    selection=selection,
                    last_move=last_move,
                )
                if hold:
                    time.sleep(hold)

            # 3. Only full-search turns enter the training buffer.
            if is_full and policy:
                records.append(Record(
                    state=encode_board(board),
                    pi=pi,
                    turn=board.turn,
                    ply=moves,
                    v_prior=float(v_prior),
                    q_root=float(root_node.q_value),
                    fen=board.fen(),
                ))

            # 4. Push the move to the board
            board.push(played)
            last_move = played
            moves += 1

            if not render and moves % 10 == 0:
                print(f"Move {moves}... (Eval: {root_node.q_value:.2f})")

            # Practice boards start inside the tables, so only normal games stop here.
            if practice_side is None and board.outcome(claim_draw=True) is None and in_tablebase(board, self.tablebase):
                stop = "TABLEBASE"
                break

            if moves >= MAX_PLIES:
                stop = "MOVE_LIMIT"
                print(f"Hit the {MAX_PLIES}-ply cap.")
                break

        # 5. Game over: result for the log and the live view.
        reward, result, termination = game_result(board, stop, practice_side is not None, self.tablebase)
        self.last_termination = termination
        self.last_root_noise_calls = root_noise_calls

        if render:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Tabula AI Self-Play (Game Over) ---")
            print(board.unicode())
            time.sleep(2.0)

        kind = "Practice game" if practice_side is not None else "Game"
        print(
            f"{kind} over after {moves} plies. {result} ({termination}). "
            f"Reward for White: {reward} | recorded {len(records)}/{moves} "
            f"(full={full_turns} fast={fast_turns} noisy_roots={root_noise_calls})"
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
        
        # 6. Discounted z and a LATE × surprise weight for each full-search turn.
        return label_game(records, board, stop, practice_side is not None, self.tablebase)


def _selection_mode(practice, ply, sampling_moves):
    if not practice and ply < sampling_moves:
        return "sample"
    return "greedy"


def _choose_move(root, selection):
    """Sample proportional to visits for the opening, then take the most visited move."""
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    if selection == "greedy" or visits.sum() <= 0:
        return moves[int(np.argmax(visits))]
    weights = visits / visits.sum()
    return moves[int(np.random.choice(len(moves), p=weights))]

def main():
    champ = "models/tabula_v0.2_champion.pth"
    model_path = champ if os.path.exists(champ) else None
    worker = SelfPlayWorker(model_path=model_path, num_simulations=DEFAULT_SIMULATIONS)
    
    print("--- GENERATING 1 TEST GAME ---")
    start_time = time.time()
    
    # Play 1 full game autonomously
    data = worker.play_game()
    
    # Save the (s, pi, z) dataset to disk for PyTorch to use later
    os.makedirs("data", exist_ok=True)
    filename = f"data/selfplay_test_{int(time.time())}.pt"
    
    torch.save(data, filename)
    
    elapsed = time.time() - start_time
    print(f"\nSaved {len(data)} board positions to {filename} in {elapsed:.1f} seconds.")
    print("The self-play pipeline is fully functional!")

if __name__ == "__main__":
    main()

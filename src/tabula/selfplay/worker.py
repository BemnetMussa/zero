import os
import time
import numpy as np
import torch
import chess

from tabula.model.model import TabulaNet
from tabula.mcts.search import MCTS
from tabula.game.encoder import encode_board, needs_orient, orient_move
from tabula.game.action_encoder import ActionEncoder
from tabula.game.score import ADJUDICATE_LEAD, ADJUDICATE_PLIES, material_white, score_game

# v0.2 defaults (AlphaZero used 800; laptop budget stays far below that).
DEFAULT_RES_BLOCKS = 8
DEFAULT_SIMULATIONS = 80

class SelfPlayWorker:
    def __init__(self, model_path=None, num_simulations=DEFAULT_SIMULATIONS, num_res_blocks=DEFAULT_RES_BLOCKS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing TabulaNet for Self-Play on {self.device} ({num_res_blocks} ResBlocks, {num_simulations} sims)...")
        self.model = TabulaNet(num_res_blocks=num_res_blocks).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.mcts = MCTS(self.model, num_simulations=num_simulations, c_puct=1.25)
        self.action_encoder = ActionEncoder()

    def play_game(self, render=False, delay=0.5, live=None, hold=0.0, sampling_moves=30):
        """Plays a single game of AlphaZero vs AlphaZero and returns the training examples.

        Self-play follows the AlphaZero pseudocode: Dirichlet noise at the root, and for the
        first `sampling_moves` plies the move is sampled from the visit counts. After that
        it plays the most visited move. The saved policy target is still the visit counts.
        """
        board = chess.Board()
        examples = [] # Will store temporary tuples of (board_tensor, pi_tensor, player_turn)
        last_move = None

        if not render:
            print("Starting new self-play game...")

        moves = 0
        stop = None
        lead_run = 0
        while board.outcome(claim_draw=True) is None:
            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Tabula AI Self-Play (Move {moves}) ---")
                print(board.unicode())
                time.sleep(delay)

            # 1. Run MCTS. Noise is on so this game is not a copy of the last one.
            root_node = self.mcts.search(board, encode_board, add_noise=True)

            # 2. Policy target π is the visit distribution. Temperature only picks the move.
            pi = np.zeros(4672, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root_node.children.values())
            if total_visits <= 0:
                break

            # Policy targets live in side-to-move coordinates (same as encode_board).
            flip = needs_orient(board)
            for move, child in root_node.children.items():
                idx = self.action_encoder.encode(orient_move(move, flip))
                pi[idx] = child.visit_count / total_visits

            selection = "sample" if moves < sampling_moves else "greedy"
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

            # 3. Store the State (s), Policy (π), and whose turn it is
            state_tensor = encode_board(board)
            examples.append([state_tensor, pi, board.turn])

            # 4. Push the move to the board
            board.push(played)
            last_move = played
            moves += 1

            if not render and moves % 10 == 0:
                print(f"Move {moves}... (Eval: {root_node.q_value:.2f})")

            # A held material lead is a decided game for a net that cannot find mate yet.
            # Scoring that 0 is what glued the value head to "everything is a draw".
            if abs(material_white(board)) >= ADJUDICATE_LEAD:
                lead_run += 1
            else:
                lead_run = 0
            if lead_run >= ADJUDICATE_PLIES:
                stop = "MATERIAL"
                print("One side kept a piece lead. Scoring the game from the material.")
                break

            # Cap max moves so a wandering game cannot run forever.
            if moves > 150:
                stop = "MOVE_LIMIT"
                print("Hit 150 move limit. Scoring the unfinished game from the material.")
                break

        # 5. Game Over - Calculate Rewards (z)
        # Mate stays ±1. A draw the rules produced stays 0. Anything we cut off
        # is the material on the board, or the value head never sees a winner.
        outcome = board.outcome(claim_draw=True)
        reward, result, termination = score_game(board, outcome, stop)

        if render:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Tabula AI Self-Play (Game Over) ---")
            print(board.unicode())
            time.sleep(2.0)

        print(f"Game over after {moves} plies. {result} ({termination}). Reward for White: {reward}")
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
        
        # 6. Retroactively assign the final reward to all saved turns
        training_data = []
        for state_tensor, pi, player_turn in examples:
            # If it was White's turn, the reward is `reward`. If Black's turn, it's `-reward`.
            actual_reward = reward if player_turn == chess.WHITE else -reward
            training_data.append((state_tensor, pi, np.array([actual_reward], dtype=np.float32)))
            
        return training_data

def _choose_move(root, selection):
    """Sample proportional to visits for the opening, then take the most visited move."""
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    if selection == "greedy" or visits.sum() <= 0:
        return moves[int(np.argmax(visits))]
    weights = visits / visits.sum()
    return moves[int(np.random.choice(len(moves), p=weights))]

def main():
    # We use fewer simulations for this test script so it runs fast in the terminal
    worker = SelfPlayWorker(num_simulations=DEFAULT_SIMULATIONS)
    
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

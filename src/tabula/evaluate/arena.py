import os
import time
import chess
import torch
from tabula.model.model import TabulaNet
from tabula.mcts.search import MCTS
from tabula.game.encoder import encode_board
from tabula.game.score import ADJUDICATE_LEAD, ADJUDICATE_PLIES, material_white, score_game

# v0.2: material/move-limit cutoffs count less than real mates in the gate.
ARENA_ADJUDICATE_SCALE = 0.5
DEFAULT_RES_BLOCKS = 8

# 10 openings × 2 colors = 20 unique paired games (matches pipeline num_games=20).
# With only 5 openings, games 11–20 were deterministic clones of games 1–10.
OPENING_SUITE = [
    ["e2e4", "e7e5"],       # Open Game (1. e4 e5)
    ["d2d4", "d7d5"],       # Queen's Gambit / closed (1. d4 d5)
    ["c2c4", "e7e5"],       # English (1. c4 e5)
    ["g1f3", "d7d5"],       # Reti (1. Nf3 d5)
    ["e2e4", "c7c5"],       # Sicilian (1. e4 c5)
    ["e2e4", "e7e6"],       # French (1. e4 e6)
    ["e2e4", "c7c6"],       # Caro-Kann (1. e4 c6)
    ["d2d4", "g8f6"],       # Indian Defence (1. d4 Nf6)
    ["c2c4", "c7c5"],       # Symmetrical English (1. c4 c5)
    ["g1f3", "g8f6"],       # Reti / flexible (1. Nf3 Nf6)
]

class Arena:
    def __init__(self, champion_path, challenger_path, num_simulations=100, num_res_blocks=DEFAULT_RES_BLOCKS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Arena running on: {self.device} ({num_res_blocks} ResBlocks, {num_simulations} sims)")
        
        # 1. Load the Champion
        self.champion = TabulaNet(num_res_blocks=num_res_blocks).to(self.device)
        if os.path.exists(champion_path):
            self.champion.load_state_dict(torch.load(champion_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: No champion found at {champion_path}. Using random weights.")
        self.champion.eval()
        
        # 2. Load the Challenger
        self.challenger = TabulaNet(num_res_blocks=num_res_blocks).to(self.device)
        if os.path.exists(challenger_path):
            self.challenger.load_state_dict(torch.load(challenger_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: No challenger found at {challenger_path}. Using random weights.")
        self.challenger.eval()
        
        # 3. Setup MCTS for both
        self.champ_mcts = MCTS(self.champion, num_simulations=num_simulations)
        self.chall_mcts = MCTS(self.challenger, num_simulations=num_simulations)
        
        self.champion_path = champion_path

    def play_game(self, challenger_is_white, live=None, hold=0.0, arena_info=None, opening_moves=None):
        """Plays a single tournament game between Champion and Challenger."""
        board = chess.Board()
        last_move = None
        if opening_moves:
            for uci in opening_moves:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    board.push(mv)
                    last_move = mv

        moves = len(board.move_stack)
        stop = None
        lead_run = 0
        
        # Assign colors
        white_mcts = self.chall_mcts if challenger_is_white else self.champ_mcts
        black_mcts = self.champ_mcts if challenger_is_white else self.chall_mcts
        
        while board.outcome(claim_draw=True) is None:
            # Determine whose turn it is
            current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts
            
            # Search the tree. No Dirichlet noise: evaluation stays greedy.
            root_node = current_mcts.search(board, encode_board, add_noise=False)
            
            # Deterministic move selection (Argmax) for tournament play
            best_move = None
            best_visits = -1
            
            for move, child in root_node.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = move

            if live is not None:
                color = "White" if challenger_is_white else "Black"
                live.show_arena(
                    headline=f"Arena · challenger as {color} · ply {moves}",
                    board=board,
                    root=root_node,
                    ply=moves,
                    last_move=last_move,
                    result=None,
                    score_line=None,
                    arena_info=arena_info,
                )
                if hold:
                    time.sleep(hold)

            board.push(best_move)
            last_move = best_move
            moves += 1

            if abs(material_white(board)) >= ADJUDICATE_LEAD:
                lead_run += 1
            else:
                lead_run = 0
            if lead_run >= ADJUDICATE_PLIES:
                stop = "MATERIAL"
                break

            # Cap moves to prevent infinite loops from bad networks
            if moves > 150:
                stop = "MOVE_LIMIT"
                break

        # Mate is a full point. A rules draw is half. A game we cut off is scored
        # from the material, so a queen-up challenger is not recorded as a draw.
        outcome = board.outcome(claim_draw=True)
        white_reward, result, termination = score_game(
            board, outcome, stop, adjudicate_scale=ARENA_ADJUDICATE_SCALE
        )
        self.last_result = f"{result} ({termination}) white {white_reward:+.2f}"
        if live is not None:
            winner = "WHITE" if white_reward > 0.05 else ("BLACK" if white_reward < -0.05 else "DRAW")
            live.record_game_outcome("arena", winner, termination, moves)
            live.show_arena(
                headline=f"{result} · {termination.replace('_', ' ').lower()} · {moves} plies",
                board=board,
                root=None,
                ply=moves,
                last_move=last_move,
                result=result,
                score_line=self.last_result,
                arena_info=arena_info,
            )
            if hold:
                time.sleep(max(hold, 1.0))
        if challenger_is_white:
            return (white_reward + 1.0) / 2.0
        return (1.0 - white_reward) / 2.0

    def evaluate(self, num_games=10, win_threshold=0.55, live=None, hold=0.0):
        """
        Plays a tournament. If Challenger scores > win_threshold (e.g. 55%), 
        it becomes the new Champion.
        """
        unique_pairs = 2 * len(OPENING_SUITE)
        print(f"\n--- TOURNAMENT: Challenger vs Champion ({num_games} games) ---")
        print(f"Opening suite: {len(OPENING_SUITE)} lines → {unique_pairs} unique color-paired games before any repeat")
        if num_games > unique_pairs:
            print(
                f"WARNING: num_games={num_games} > {unique_pairs}; "
                "later games will recycle openings (clone risk)."
            )
        challenger_score = 0.0
        challenger_wins = 0
        champion_wins = 0
        draws = 0
        
        for i in range(num_games):
            start_time = time.time()
            # Paired openings: Game 0 & 1 share opening 0, Game 2 & 3 share opening 1, etc.
            op_idx = (i // 2) % len(OPENING_SUITE)
            opening = OPENING_SUITE[op_idx]
            opening_label = " ".join(opening)

            # Alternate colors: Game 0 (Challenger=White), Game 1 (Challenger=Black)
            challenger_is_white = (i % 2 == 0)
            color_str = "White" if challenger_is_white else "Black"
            
            arena_info = {
                "game": i + 1,
                "totalGames": num_games,
                "challengerIsWhite": challenger_is_white,
                "challengerScore": challenger_score,
                "challengerWins": challenger_wins,
                "championWins": champion_wins,
                "draws": draws,
                "winThreshold": win_threshold
            }
            result = self.play_game(
                challenger_is_white,
                live=live,
                hold=hold if i == 0 else 0.0,
                arena_info=arena_info,
                opening_moves=opening
            )
            challenger_score += result
            if result > 0.5:
                challenger_wins += 1
            elif result < 0.5:
                champion_wins += 1
            else:
                draws += 1
            
            elapsed = time.time() - start_time
            detail = getattr(self, "last_result", "")
            print(
                f"Game {i+1}/{num_games} | Challenger as {color_str} | "
                f"Opening {op_idx+1}/{len(OPENING_SUITE)} [{opening_label}] | "
                f"Score: {result:.2f} | {detail} | Time: {elapsed:.1f}s"
            )
            
        win_rate = challenger_score / num_games
        self.last_win_rate = win_rate
        self.last_games = num_games
        self.challenger_wins = challenger_wins
        self.champion_wins = champion_wins
        self.draws = draws
        print(f"\nFinal Score: Challenger {challenger_score}/{num_games} (Win Rate: {win_rate*100:.1f}%) | Record: {challenger_wins}W - {draws}D - {champion_wins}L")
        
        if win_rate >= win_threshold:
            print("GATING PASSED: Challenger is strictly better!")
            print(f"Promoting Challenger to new Champion at {self.champion_path}...")
            # Overwrite the champion weights with the challenger weights
            torch.save(self.challenger.state_dict(), self.champion_path)
            return True
        else:
            print("GATING FAILED: Challenger rejected. Keeping old Champion.")
            return False

if __name__ == "__main__":
    # Test the Arena (v0.2 paths)
    champ_file = "models/tabula_v0.2_champion.pth"
    chall_file = "models/tabula_v0.2_latest.pth"
    
    # If there's no champion yet, the latest trained network automatically becomes it.
    if not os.path.exists(champ_file) and os.path.exists(chall_file):
        print("No champion found! Promoting the latest trained network to Champion automatically.")
        import shutil
        os.makedirs("models", exist_ok=True)
        shutil.copy(chall_file, champ_file)
        
    # Manual smoke test — lighter than the pipeline gate (100 sims / 20 games).
    arena = Arena(champion_path=champ_file, challenger_path=chall_file, num_simulations=40)
    arena.evaluate(num_games=4, win_threshold=0.55)

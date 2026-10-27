import os
import shutil
import time
import torch

from tabula.selfplay.worker import SelfPlayWorker, DEFAULT_SIMULATIONS, DEFAULT_RES_BLOCKS
from tabula.train.train import train
from tabula.evaluate.arena import Arena
from tabula.watch.live import LiveBoard
from tabula.hardware.thermal import ThermalGuard

# v0.2: new encoding + 8-block net — do not mix with v0.1 absolute-channel data/weights.
CHAMP_PATH = "models/tabula_v0.2_champion.pth"
CHALL_PATH = "models/tabula_v0.2_latest.pth"
V01_DATA_ARCHIVE = "data/archive_v1_absolute"
ARENA_SIMULATIONS = 100  # Exam at least as deep as self-play (80); slight edge for the gate.


def get_next_iteration() -> int:
    """Auto-detect next iteration number from existing files in data/ (not archives)."""
    existing = []
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.startswith("selfplay_iter_") and f.endswith(".pt"):
                try:
                    parts = f.split("_")
                    existing.append(int(parts[2]))
                except (IndexError, ValueError):
                    pass
    return (max(existing) + 1) if existing else 1


def archive_v01_data_if_needed():
    """Move pre-perspective .pt files out of the active replay window once.

    v0.1 boards used absolute White/Black channels. Training v0.2 on that mix would
    poison the new side-to-move encoder. Safe to re-run: only moves top-level *.pt
    when no v0.2 session has started yet (empty check: skip if champion already exists).
    """
    if os.path.exists(CHAMP_PATH) or os.path.exists(CHALL_PATH):
        return  # v0.2 already has weights; active data/ is the new stream
    if not os.path.exists("data"):
        return
    leftovers = [
        f for f in os.listdir("data")
        if f.endswith(".pt") and os.path.isfile(os.path.join("data", f))
    ]
    if not leftovers:
        return
    os.makedirs(V01_DATA_ARCHIVE, exist_ok=True)
    moved = 0
    for name in leftovers:
        src = os.path.join("data", name)
        dst = os.path.join(V01_DATA_ARCHIVE, name)
        if os.path.exists(dst):
            dst = os.path.join(V01_DATA_ARCHIVE, f"{int(time.time())}_{name}")
        shutil.move(src, dst)
        moved += 1
    print(f"[v0.2] Archived {moved} v0.1 self-play files → {V01_DATA_ARCHIVE}/")


def main():
    champ_path = CHAMP_PATH
    chall_path = CHALL_PATH

    archive_v01_data_if_needed()

    live = LiveBoard(enable_tui=True)
    thermal = ThermalGuard(warm_threshold=72.0, emergency_threshold=80.0, target_cool_temp=62.0, cooldown_sleep_sec=2.0)
    iteration = get_next_iteration()

    print(
        f"[Tabula v0.2] {DEFAULT_RES_BLOCKS} ResBlocks | "
        f"self-play {DEFAULT_SIMULATIONS} sims | arena {ARENA_SIMULATIONS} sims | "
        f"perspective encoding | cosine LR"
    )

    # We loop forever, generating data, training, and evaluating
    while True:
        # Pre-check thermals
        thermal.check_and_cool()
        
        # ---------------------------------------------------------
        # PHASE 1: SELF PLAY (Generate Data)
        # ---------------------------------------------------------
        print(f"\n[PHASE 1] Generating Self-Play Data (Iteration {iteration})...")
        # AlphaZero continuous latest: rejected nets still generate data; the gate
        # only protects the champion crown (research-corrected vs AGZ champion-only).
        latest_exists = os.path.exists(chall_path)
        play_path = chall_path if latest_exists else champ_path
        live.begin_round(iteration, latest_exists)
        worker = SelfPlayWorker(
            model_path=play_path,
            num_simulations=DEFAULT_SIMULATIONS,
            num_res_blocks=DEFAULT_RES_BLOCKS,
        )
        
        # We generate a few games per iteration to keep it fast for laptops
        num_games = 5
        all_data = []
        
        for g in range(num_games):
            thermal.check_and_cool()
            # The first game of each iteration is the one on the board.
            data = worker.play_game(live=live if g == 0 else None, hold=0.03 if g == 0 else 0.0)
            all_data.extend(data)
            print(f"Completed game {g+1}/{num_games}")
            thermal.rest(2.0) # 2s breather to dissipate heat between games
                
        # Save the dataset
        os.makedirs("data", exist_ok=True)
        filename = f"data/selfplay_iter_{iteration}_{int(time.time())}.pt"
        torch.save(all_data, filename)
        print(f"Saved {len(all_data)} moves to {filename}")
        
        # ---------------------------------------------------------
        # PHASE 2: TRAIN (Update Neural Network)
        # ---------------------------------------------------------
        thermal.check_and_cool()
        print(f"======================================")
        print(f"       ITERATION {iteration} - TRAINING")
        print(f"======================================")
        
        # Train loads recent .pt files in data/ and saves tabula_v0.2_latest.pth
        train(live=live)
        thermal.rest(3.0) # 3s breather after backpropagation
        
        # ---------------------------------------------------------
        # PHASE 3: EVALUATE (Arena Gating)
        # ---------------------------------------------------------
        thermal.check_and_cool()
        print(f"======================================")
        print(f"       ITERATION {iteration} - ARENA")
        print(f"======================================")
        
        # If there's no champion yet, skip arena and just promote the challenger
        if not os.path.exists(champ_path):
            print("No champion found. Promoting the first trained network to Champion!")
            shutil.copy(chall_path, champ_path)
            live.note_match(1.0, 0, True, first=True)
        else:
            # v0.2: exam ≥ self-play depth; soft material scoring inside Arena.
            arena = Arena(
                champion_path=champ_path,
                challenger_path=chall_path,
                num_simulations=ARENA_SIMULATIONS,
                num_res_blocks=DEFAULT_RES_BLOCKS,
            )
            promoted = arena.evaluate(num_games=20, win_threshold=0.55, live=live, hold=0.02)
            live.note_match(
                arena.last_win_rate,
                arena.last_games,
                promoted,
                ch_wins=arena.challenger_wins,
                ch_losses=arena.champion_wins,
                draws=arena.draws
            )
            
        print(f"\nIteration {iteration} complete. Cooling down 10 seconds before next cycle...")
        thermal.rest(10.0)
        thermal.check_and_cool()
        iteration += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import sys
        sys.stdout.write("\033[?25h\n\n[Tabula AI] Pipeline paused cleanly by user.\n")
        sys.stdout.flush()

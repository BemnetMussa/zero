import os
import glob
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tabula.checkpoint import load_model_weights, load_training_state
from tabula.game.encoder import INPUT_PLANES
from tabula.model.model import TabulaNet
from tabula.runtime import atomic_torch_save

import numpy as np

# v0.2 capacity + schedule (v0.1 was 5 blocks / flat Adam).
DEFAULT_RES_BLOCKS = 8
MODEL_LATEST = "tabula_v0.2_latest.pth"
# v0.3: keep the replay window tight so Gen-1 noise is forgotten faster.
DEFAULT_MAX_FILES = 10
DEFAULT_BATCH_SIZE = 48
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
ENTROPY_BONUS = 0.01


def policy_loss_with_entropy(policy_logits, targets, weights):
    """Weighted soft-target cross-entropy minus a 0.01 entropy bonus."""

    log_probs = F.log_softmax(policy_logits, dim=1)
    probs = log_probs.exp()
    cross_entropy = -(targets * log_probs).sum(dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return ((cross_entropy - ENTROPY_BONUS * entropy) * weights).mean()


class SelfPlayDataset(Dataset):
    def __init__(self, data_dir, max_files=DEFAULT_MAX_FILES):
        """
        Loads the most recent self-play .pt files (sliding replay buffer).
        Pre-vectorizes data into contiguous tensors to eliminate Python per-item overhead.

        Each item may be (s, π, z) from v0.2 or (s, π, z, w) from v0.3.
        Missing weights default to 1.0.

        Run #3: only INPUT_PLANES (20) board tensors are accepted. Stale Run #2
        13-channel rows are skipped so a leftover replay file cannot crash training
        mid-batch with a silent shape mismatch.
        """
        pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")), key=os.path.getmtime)
        selected_files = pt_files[-max_files:] if len(pt_files) > max_files else pt_files
        
        all_states, all_pis, all_zs, all_ws = [], [], [], []
        skipped_incompatible = 0
        for f in selected_files:
            data = torch.load(f, weights_only=False)
            for item in data:
                if len(item) == 4:
                    s, pi, z, w = item
                else:
                    s, pi, z = item
                    w = np.array([1.0], dtype=np.float32)
                s = np.asarray(s)
                if s.ndim != 3 or s.shape[0] != INPUT_PLANES:
                    skipped_incompatible += 1
                    continue
                all_states.append(s)
                all_pis.append(pi)
                all_zs.append(z)
                all_ws.append(w)
                
        if all_states:
            self.states = torch.from_numpy(np.array(all_states, dtype=np.float32))
            self.pis = torch.from_numpy(np.array(all_pis, dtype=np.float32))
            self.zs = torch.from_numpy(np.array(all_zs, dtype=np.float32))
            self.weights = torch.from_numpy(np.array(all_ws, dtype=np.float32)).reshape(-1)
        else:
            self.states = torch.empty((0, INPUT_PLANES, 8, 8), dtype=torch.float32)
            self.pis = torch.empty((0, 4672), dtype=torch.float32)
            self.zs = torch.empty((0, 1), dtype=torch.float32)
            self.weights = torch.empty((0,), dtype=torch.float32)

        msg = (
            f"Loaded {len(self.states)} examples from {len(selected_files)} recent files "
            f"(Replay buffer window)."
        )
        if skipped_incompatible:
            msg += (
                f" Skipped {skipped_incompatible} incompatible board tensors "
                f"(expected {INPUT_PLANES} planes)."
            )
        print(msg)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.pis[idx], self.zs[idx], self.weights[idx]


@dataclass
class TrainResult:
    model: TabulaNet
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    policy_loss: float
    value_loss: float
    total_loss: float
    learning_rate: float
    examples: int


def train(
    live=None,
    *,
    data_dir="data",
    model_path=None,
    checkpoint=None,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    device=None,
):
    """Train one iteration and return all state needed for an atomic checkpoint."""
    print("=== Zero - AlphaZero Training Loop (v0.3) ===")
    
    # --- Configuration ---
    lr = DEFAULT_LR
    weight_decay = DEFAULT_WEIGHT_DECAY
    num_res_blocks = DEFAULT_RES_BLOCKS
    
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device} | ResBlocks={num_res_blocks} | cosine LR | weighted LATE/surprise")
    
    # 1. Load Data
    print("\n--- 1. Loading Self-Play Data ---")
    dataset = SelfPlayDataset(data_dir, max_files=DEFAULT_MAX_FILES)
    if len(dataset) == 0:
        raise RuntimeError("No training data found; refusing to checkpoint an untrained iteration")
        
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False,
        pin_memory=(device.type == "cuda")
    )
    
    # 2. Initialize Model
    print("\n--- 2. Initializing Neural Network ---")
    model = TabulaNet(num_res_blocks=num_res_blocks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # One cosine cycle per pipeline iteration, while retaining optimizer and
    # scheduler continuity across process restarts.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=epochs,
        eta_min=lr * 0.05,
    )

    if checkpoint is not None:
        print(f"Restoring iteration {checkpoint['iteration']} training state...")
        padded = load_training_state(model, optimizer, scheduler, checkpoint)
        if padded:
            print(
                f"Warm-started model to {INPUT_PLANES} input planes; "
                "optimizer/scheduler reset."
            )
    elif model_path and os.path.exists(model_path):
        print(f"Found existing weights. Loading from {model_path}...")
        raw = torch.load(model_path, map_location=device, weights_only=True)
        state = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
        padded = load_model_weights(model, state)
        if padded:
            print(f"Warm-started weights to {INPUT_PLANES} input planes.")
    else:
        print("Starting with fresh randomly initialized weights (8 blocks / 20 planes).")
    
    # 3. Loss functions
    # Per-sample losses so LATE / surprise weights can scale each row.
    value_loss_fn = nn.MSELoss(reduction="none")
    
    # 4. Training Loop
    print("\n--- 3. Starting Training Loop ---")
    model.train()
    
    avg_p_loss = 0.0
    avg_v_loss = 0.0
    avg_total_loss = 0.0
    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (states, pis, zs, weights) in enumerate(dataloader):
            # Move data to GPU asynchronously
            states = states.to(device, non_blocking=True)
            pis = pis.to(device, non_blocking=True)
            zs = zs.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, values = model(states)
            
            loss_v_row = value_loss_fn(values, zs).view(-1)
            loss_p = policy_loss_with_entropy(policy_logits, pis, weights)
            loss_v = (loss_v_row * weights).mean()
            
            # Combine into joint loss
            total_loss = loss_p + loss_v
            
            # Backward pass & optimize
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += loss_p.item()
            epoch_value_loss += loss_v.item()
            epoch_total_loss += total_loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate averages for logging
        avg_p_loss = epoch_policy_loss / len(dataloader)
        avg_v_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)
        
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:02d}/{epochs} [{elapsed:.1f}s] lr={current_lr:.6f} - "
            f"Total Loss: {avg_total_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})"
        )
        if live is not None:
            live.show_training(epoch + 1, epochs, avg_total_loss, avg_p_loss, avg_v_loss)

    return TrainResult(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        policy_loss=avg_p_loss,
        value_loss=avg_v_loss,
        total_loss=avg_total_loss,
        learning_rate=current_lr,
        examples=len(dataset),
    )

if __name__ == "__main__":
    result = train(model_path=os.path.join("models", MODEL_LATEST))
    atomic_torch_save(result.model.state_dict(), os.path.join("models", MODEL_LATEST))
    print(f"Training complete. Weights safely stored to models/{MODEL_LATEST}.")

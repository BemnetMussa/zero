import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tabula.model.model import TabulaNet

import numpy as np

# v0.2 capacity + schedule (v0.1 was 5 blocks / flat Adam).
DEFAULT_RES_BLOCKS = 8
MODEL_LATEST = "tabula_v0.2_latest.pth"
# v0.3: keep the replay window tight so Gen-1 noise is forgotten faster.
DEFAULT_MAX_FILES = 10


class SelfPlayDataset(Dataset):
    def __init__(self, data_dir, max_files=DEFAULT_MAX_FILES):
        """
        Loads the most recent self-play .pt files (sliding replay buffer).
        Pre-vectorizes data into contiguous tensors to eliminate Python per-item overhead.

        Each item may be (s, π, z) from v0.2 or (s, π, z, w) from v0.3.
        Missing weights default to 1.0.
        """
        pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")), key=os.path.getmtime)
        selected_files = pt_files[-max_files:] if len(pt_files) > max_files else pt_files
        
        all_states, all_pis, all_zs, all_ws = [], [], [], []
        for f in selected_files:
            data = torch.load(f, weights_only=False)
            for item in data:
                if len(item) == 4:
                    s, pi, z, w = item
                else:
                    s, pi, z = item
                    w = np.array([1.0], dtype=np.float32)
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
            self.states = torch.empty((0, 13, 8, 8), dtype=torch.float32)
            self.pis = torch.empty((0, 4672), dtype=torch.float32)
            self.zs = torch.empty((0, 1), dtype=torch.float32)
            self.weights = torch.empty((0,), dtype=torch.float32)
            
        print(f"Loaded {len(self.states)} examples from {len(selected_files)} recent files (Replay buffer window).")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.pis[idx], self.zs[idx], self.weights[idx]


def train(live=None):
    print("=== Zero - AlphaZero Training Loop (v0.3) ===")
    
    # --- Configuration ---
    data_dir = "data"
    models_dir = "models"
    batch_size = 48  # Slightly under v1's 64 to leave headroom for 8 ResBlocks on 4GB
    epochs = 10
    lr = 0.001
    weight_decay = 1e-4  # L2 regularization (c ||theta||^2)
    num_res_blocks = DEFAULT_RES_BLOCKS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | ResBlocks={num_res_blocks} | cosine LR | weighted LATE/surprise")
    
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, MODEL_LATEST)
    
    # 1. Load Data
    print("\n--- 1. Loading Self-Play Data ---")
    dataset = SelfPlayDataset(data_dir, max_files=DEFAULT_MAX_FILES)
    if len(dataset) == 0:
        print("No training data found. Run self-play first (python -m tabula.selfplay.worker).")
        return
        
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
    
    if os.path.exists(model_path):
        print(f"Found existing weights. Loading from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("Starting with fresh randomly initialized weights (v0.2 / perspective / 8 blocks).")
    
    # 3. Optimizer, cosine schedule, loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    
    # Per-sample losses so LATE / surprise weights can scale each row.
    policy_loss_fn = nn.CrossEntropyLoss(reduction="none")
    value_loss_fn = nn.MSELoss(reduction="none")
    
    # 4. Training Loop
    print("\n--- 3. Starting Training Loop ---")
    model.train()
    
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
            
            loss_p_row = policy_loss_fn(policy_logits, pis)
            loss_v_row = value_loss_fn(values, zs).view(-1)
            loss_p = (loss_p_row * weights).mean()
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
        
    # 5. Save the trained model
    print("\n--- 4. Saving Model ---")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Weights safely stored to {model_path}.")

if __name__ == "__main__":
    train()

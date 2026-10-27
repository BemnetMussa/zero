import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    A standard Residual Block. 
    It passes the input through two convolutional layers and then adds the original 
    input back to the output (the 'skip connection') before the final ReLU activation.
    This prevents the vanishing gradient problem in deep networks.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class TabulaNet(nn.Module):
    """
    The Version 1 PyTorch Neural Network for Tabula AI (AlphaZero architecture).
    Input: (Batch Size, 13, 8, 8) tensor representing the static board.
    Outputs: 
        - Policy: Probability distribution over all possible moves.
        - Value: A single number [-1, 1] estimating who is winning.
    """
    def __init__(self, num_res_blocks=5, num_hidden=256, action_size=4672):
        super().__init__()
        
        # 1. The Input Block
        # Takes our 13-layer board representation and expands it to 'num_hidden' filters
        self.start_block = nn.Sequential(
            nn.Conv2d(13, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # 2. The Body (Residual Blocks)
        # Deep stack of blocks to understand spatial chess patterns
        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )
        
        # 3. The Dual Heads
        
        # POLICY HEAD: \"What move should I play?\"
        # Shrinks to 2 filters, flattens, and maps to the action space.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, action_size)
            # Note: We don't apply Softmax here; we use CrossEntropyLoss later which expects raw logits.
        )
        
        # VALUE HEAD: how good is this position for the side about to move?
        # A hard ReLU here died completely after a run of all-draw labels, and a dead
        # unit never gets a gradient again. The leak keeps that from being permanent.
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Pass through the input block
        x = self.start_block(x)
        
        # Pass through all residual blocks in the backbone
        for block in self.backbone:
            x = block(x)
            
        # Split into the two heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

if __name__ == "__main__":
    # --- QUICK TEST TO VERIFY SHAPES ---
    print("Initializing TabulaNet (v0.2 default shape check)...")
    model = TabulaNet(num_res_blocks=8)
    
    # Create a dummy input tensor matching our 13-layer V1 design
    # Shape: (Batch_Size=1, Channels=13, Height=8, Width=8)
    dummy_input = torch.randn(1, 13, 8, 8)
    
    # Run the dummy input through the network
    policy, value = model(dummy_input)
    
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Policy Shape: {policy.shape} -> (Batch Size, Action Space Size)")
    print(f"Value Shape:  {value.shape}  -> (Batch Size, Evaluation)")
    print("\nModel compiles and dual heads output correctly! Ready for training.")

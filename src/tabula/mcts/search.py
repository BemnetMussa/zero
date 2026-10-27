import math
import numpy as np
import torch
import chess

def terminal_value(board):
    """Value for the side to move, or None if the game is still going.

    A draw the rules will claim (repetition, the 50-move rule) is 0.
    Checkmate is -1 for the side that has to move.
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class Node:
    """
    A single board state in the Monte Carlo Tree.
    """
    def __init__(self, prior_prob: float, parent=None):
        self.prior_prob = prior_prob       # P: Probability predicted by TabulaNet
        self.parent = parent               # Parent Node
        self.children = {}                 # Dictionary mapping chess.Move -> child Node
        self.visit_count = 0               # N: Number of times this node was visited
        self.value_sum = 0.0               # W: Total value accumulated from sub-tree

    @property
    def q_value(self) -> float:
        """Q: The average value (W / N) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_probs: dict[chess.Move, float]):
        """
        Populate the node with its legal children.
        action_probs is a dictionary of {legal_move: probability} returned by TabulaNet.
        """
        for move, prob in action_probs.items():
            if move not in self.children:
                self.children[move] = Node(prior_prob=prob, parent=self)

    def is_expanded(self) -> bool:
        """If it has children, it has already been expanded by the Neural Net."""
        return len(self.children) > 0


class MCTS:
    """
    The Monte Carlo Tree Search algorithm for AlphaZero.
    """
    def __init__(self, model, num_simulations=100, c_puct=1.25):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct  # Hyperparameter controlling exploration

    def search(self, board, encoder, add_noise=False, dirichlet_alpha=0.3, noise_fraction=0.25):
        """
        Run the simulations to build the tree and return the root node.
        We can then pick the best move based on the visit counts of the root's children.

        add_noise is for self-play only. AlphaZero mixes Dirichlet noise into the
        root prior so the search sometimes tries moves the network currently ignores.
        Arena games and human play leave it off.
        """
        root = Node(prior_prob=1.0)
        
        # 0. Initial Expansion
        policy_dict, _ = self.predict(board, encoder)
        root.expand(policy_dict)
        if add_noise and root.children:
            self._add_dirichlet(root, dirichlet_alpha, noise_fraction)

        for _ in range(self.num_simulations):
            node = root
            search_board = board.copy()
            
            # 1. SELECT
            # Traverse down the tree picking the highest PUCT score until we hit a leaf
            while node.is_expanded():
                move, node = self.select_child(node)
                search_board.push(move)
                
            # 2. EXPAND & EVALUATE
            # claim_draw matches the game loop. Without it, a threefold repetition
            # is not over inside the tree, so the search walks into the draw.
            value = terminal_value(search_board)
            if value is None:
                # Leaf node reached - Ask TabulaNet for P (Policy) and V (Value)
                policy_dict, value = self.predict(search_board, encoder)
                node.expand(policy_dict)
                
            # 3. BACKPROPAGATE
            # Pass the value back up the tree
            self.backpropagate(node, value)
            
        return root

    def _add_dirichlet(self, root: Node, alpha: float, fraction: float):
        """P(root, a) = (1 - ε) p_a + ε η_a, with η ~ Dir(α) and ε = 0.25 for chess."""
        children = list(root.children.values())
        noise = np.random.dirichlet([alpha] * len(children))
        for child, eta in zip(children, noise):
            child.prior_prob = child.prior_prob * (1.0 - fraction) + float(eta) * fraction

    def select_child(self, node: Node):
        """Pick the child with the highest PUCT score."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = self.puct_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def puct_score(self, parent: Node, child: Node) -> float:
        """
        AlphaZero PUCT:
        Exploitation (Q) + Exploration (c * P * sqrt(N_parent) / (1 + N_child))

        The value stored on a node is for the player about to move there.
        The parent wants the opposite, so Q is negated.
        """
        q_val = -child.q_value
        exploration_term = self.c_puct * child.prior_prob * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return q_val + exploration_term

    def backpropagate(self, node: Node, value: float):
        """Update N (visit count) and W (value sum) all the way up to the root."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            # In chess, a move that is good for White is bad for Black. 
            # We invert the value as we step up the tree for the opponent's turn.
            value = -value 
            current = current.parent

    def predict(self, board, encoder):
        """
        Helper to wrap the PyTorch model prediction. 
        Passes the board to TabulaNet, extracts the scalar Value, and maps the Policy back to legal moves.
        """
        # encoder is the encode_board function which returns a numpy array
        numpy_board = encoder(board)
        # Find the device the model is on
        device = next(self.model.parameters()).device
        tensor_board = torch.tensor(numpy_board, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, 13, 8, 8)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, value_tensor = self.model(tensor_board)
            
        v = value_tensor.item()
        logits = policy_logits[0] # Shape: (4672,)
        
        # We need the action encoder to map legal moves to their 4672 indices
        from tabula.game.action_encoder import ActionEncoder
        import torch.nn.functional as F
        
        from tabula.game.encoder import needs_orient, orient_move

        encoder_4672 = ActionEncoder()
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return {}, v

        # Policy tensor is in side-to-move coordinates; map real moves before indexing.
        flip = needs_orient(board)
        legal_indices = [encoder_4672.encode(orient_move(m, flip)) for m in legal_moves]
        
        # Mask out illegal moves by setting their logits to negative infinity
        mask = torch.full((4672,), float('-inf'), device=logits.device)
        mask[legal_indices] = logits[legal_indices]
        
        # Apply Softmax to convert the raw logits into actual probabilities (summing to 1.0)
        probs = F.softmax(mask, dim=0)
        
        # Map probabilities back onto real-board moves
        policy_dict = {
            m: probs[encoder_4672.encode(orient_move(m, flip))].item()
            for m in legal_moves
        }
            
        return policy_dict, v

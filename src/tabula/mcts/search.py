"""Monte Carlo Tree Search for Zero (Pure AlphaZero self-play path)."""

from __future__ import annotations

import math

import chess
import numpy as np
import torch

from tabula.selfplay.labels import REPETITION_TERMINATIONS, white_repetition_value


def _side_to_move(white_value, board):
    return white_value if board.turn == chess.WHITE else -white_value


def terminal_value(board):
    """Value for the side to move, or None if the game is still going.

    Checkmate is ±1. Stalemate, insufficient material, and the 50-move rule
    are 0. Repetition charges the repetition penalty to the offending side.
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is not None:
        return 1.0 if outcome.winner == board.turn else -1.0
    if outcome.termination.name in REPETITION_TERMINATIONS:
        return _side_to_move(white_repetition_value(board), board)
    return 0.0


def tree_leaf_value(board, at_root=False):
    """Terminal value inside the search tree, with twofold pruning.

    Below the root, returning to any earlier position (game history or the
    current search path) ends the line with the repetition penalty charged to
    the side that initiated the loop, so search stops spending visits on it.
    """
    value = terminal_value(board)
    if value is not None or at_root:
        return value
    if board.move_stack and board.is_repetition(2):
        return _side_to_move(white_repetition_value(board, reached=2), board)
    return None


def advance_root(root: "Node", move: chess.Move) -> "Node | None":
    """Reuse the played child's subtree as the next root (parent detached).

    Explicitly drops sibling subtrees and clears the old root's child map so
    abandoned nodes are unreachable and can be garbage-collected between plies
    (important on long Kaggle runs with 128-sim trees).
    """
    child = root.children.pop(move, None)
    if child is None:
        return None
    for sibling in root.children.values():
        sibling.parent = None
    root.children.clear()
    child.parent = None
    return child


def visit_policy_target(root: "Node") -> dict[chess.Move, float]:
    """Normalized visit counts at the root (sums to 1)."""
    raw = {move: child.visit_count for move, child in root.children.items()}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {move: count / total for move, count in raw.items()}


class Node:
    """A single board state in the Monte Carlo Tree."""

    def __init__(self, prior_prob: float, parent=None):
        self.prior_prob = prior_prob
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_probs: dict[chess.Move, float]):
        for move, prob in action_probs.items():
            if move not in self.children:
                self.children[move] = Node(prior_prob=prob, parent=self)

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """Monte Carlo Tree Search (AlphaZero)."""

    def __init__(self, model, num_simulations=100, c_puct=1.25):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(
        self,
        board,
        encoder,
        add_noise=False,
        dirichlet_alpha=0.3,
        noise_fraction=0.25,
        num_simulations=None,
        root: Node | None = None,
    ):
        """Run simulations and return the root node.

        When `root` is a reused subtree from the previous ply, its children keep
        their visit counts. Fresh Dirichlet noise is mixed into the new root
        priors when `add_noise` is True.
        """
        n_sims = self.num_simulations if num_simulations is None else int(num_simulations)

        if root is None:
            root = Node(prior_prob=1.0)

        if not root.is_expanded():
            policy_dict, _ = self.predict(board, encoder)
            root.expand(policy_dict)

        # Apply Dirichlet once per root object. Re-entering search() on the same
        # reused root must not stack noise into already-noised child priors.
        if add_noise and root.children and not getattr(root, "_dirichlet_applied", False):
            self._add_dirichlet(root, dirichlet_alpha, noise_fraction)
            root._dirichlet_applied = True

        for _ in range(n_sims):
            node = root
            search_board = board.copy()

            while node.is_expanded():
                move, node = self.select_child(node)
                search_board.push(move)

            value = tree_leaf_value(search_board, at_root=node is root)
            if value is None:
                policy_dict, value = self.predict(search_board, encoder)
                node.expand(policy_dict)

            self.backpropagate(node, value)

        return root

    def _add_dirichlet(self, root: Node, alpha: float, fraction: float):
        """P(root, a) = (1 - ε) p_a + ε η_a, with η ~ Dir(α)."""
        children = list(root.children.values())
        noise = np.random.dirichlet([alpha] * len(children))
        for child, eta in zip(children, noise):
            child.prior_prob = child.prior_prob * (1.0 - fraction) + float(eta) * fraction

    def select_child(self, node: Node):
        """Pick the child with the highest PUCT score."""
        best_score = -float("inf")
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
        q_val = -child.q_value
        exploration_term = (
            self.c_puct
            * child.prior_prob
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )
        return q_val + exploration_term

    def backpropagate(self, node: Node, value: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value
            current = current.parent

    def predict(self, board, encoder):
        numpy_board = encoder(board)
        device = next(self.model.parameters()).device
        tensor_board = torch.tensor(numpy_board, dtype=torch.float32).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            policy_logits, value_tensor = self.model(tensor_board)

        v = value_tensor.item()
        logits = policy_logits[0]

        from tabula.game.action_encoder import ActionEncoder
        import torch.nn.functional as F
        from tabula.game.encoder import needs_orient, orient_move

        encoder_4672 = ActionEncoder()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, v

        flip = needs_orient(board)
        legal_indices = [encoder_4672.encode(orient_move(m, flip)) for m in legal_moves]

        mask = torch.full((4672,), float("-inf"), device=logits.device)
        mask[legal_indices] = logits[legal_indices]
        probs = F.softmax(mask, dim=0)

        policy_dict = {
            m: probs[encoder_4672.encode(orient_move(m, flip))].item()
            for m in legal_moves
        }
        return policy_dict, v

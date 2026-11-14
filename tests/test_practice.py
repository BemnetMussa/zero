import random
import unittest
from types import SimpleNamespace
from unittest import mock

import chess
import numpy as np

import torch

from tabula.game.action_encoder import ActionEncoder
from tabula.game.encoder import encode_board
from tabula.mcts.search import MCTS, Node, advance_root, visit_policy_target
from tabula.selfplay import worker as worker_module
from tabula.selfplay.practice import (
    TIER_PIECE_RANGES,
    TIER_WEIGHTS,
    sample_curriculum_board,
    sample_mate_exam_board,
)
from tabula.selfplay.worker import (
    SelfPlayWorker,
    choose_move,
    temperature_for_ply,
)


class TemperatureScheduleTests(unittest.TestCase):
    def test_smooth_temperature_across_plies(self) -> None:
        self.assertEqual(temperature_for_ply(0), 1.0)
        self.assertEqual(temperature_for_ply(29), 1.0)
        self.assertEqual(temperature_for_ply(30), 0.35)
        self.assertEqual(temperature_for_ply(59), 0.35)
        self.assertEqual(temperature_for_ply(60), 0.10)

    def test_temperature_samples_with_visit_powers(self) -> None:
        first = chess.Move.from_uci("e2e4")
        second = chess.Move.from_uci("d2d4")
        root = SimpleNamespace(
            children={
                first: SimpleNamespace(visit_count=2),
                second: SimpleNamespace(visit_count=6),
            }
        )
        with mock.patch(
            "tabula.selfplay.worker.np.random.choice",
            return_value=1,
        ) as choice:
            selected = choose_move(root, tau=1.0)
        self.assertEqual(selected, second)
        probabilities = choice.call_args.kwargs["p"]
        np.testing.assert_allclose(probabilities, [0.25, 0.75])

    def test_late_temperature_stays_finite_for_large_visits(self) -> None:
        moves = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "c2c4", "g1f3")]
        root = SimpleNamespace(
            children={m: SimpleNamespace(visit_count=v) for m, v in zip(moves, (200, 128, 64, 1))}
        )
        with mock.patch(
            "tabula.selfplay.worker.np.random.choice",
            return_value=0,
        ) as choice:
            choose_move(root, tau=0.10)
        weights = choice.call_args.kwargs["p"]
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)


class CurriculumBoardTests(unittest.TestCase):
    def test_tier_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(TIER_WEIGHTS.values()), 1.0)

    def test_each_tier_matches_piece_count_range(self) -> None:
        rng = random.Random(0)
        for tier, (low, high) in TIER_PIECE_RANGES.items():
            for _ in range(40):
                start = sample_curriculum_board(rng, tier=tier)
                self.assertEqual(start.tier, tier)
                self.assertTrue(start.board.is_valid())
                self.assertIsNone(start.board.outcome(claim_draw=True))
                self.assertGreaterEqual(start.piece_count, low)
                self.assertLessEqual(start.piece_count, high)
                self.assertEqual(len(start.board.piece_map()), start.piece_count)

    def test_mate_exam_boards_are_three_piece_qrk(self) -> None:
        rng = random.Random(7)
        for _ in range(20):
            board, strong = sample_mate_exam_board(rng)
            self.assertTrue(board.is_valid())
            self.assertEqual(board.turn, strong)
            self.assertEqual(len(board.piece_map()), 3)
            extras = len(board.pieces(chess.QUEEN, strong)) + len(board.pieces(chess.ROOK, strong))
            self.assertEqual(extras, 1)


class SelfPlayPathTests(unittest.TestCase):
    def test_curriculum_selfplay_uses_noisy_full_search_every_ply(self) -> None:
        board = chess.Board("8/8/8/8/8/2k5/5q2/K7 b - - 0 1")

        class FakeMCTS:
            def __init__(self):
                self.num_simulations = 50
                self.search_calls = []

            def predict(self, _board, _encoder):
                return {}, 0.0

            def search(self, position, _encoder, **kwargs):
                self.search_calls.append(kwargs)
                mating_move = None
                for move in position.legal_moves:
                    child = position.copy()
                    child.push(move)
                    if child.is_checkmate():
                        mating_move = move
                        break
                if mating_move is None:
                    raise AssertionError("curriculum board has no mating move")
                child = Node(0.5)
                child.visit_count = 10
                root = Node(1.0)
                root.children = {mating_move: child}
                child.parent = root
                root.visit_count = 10
                return root

        worker = object.__new__(SelfPlayWorker)
        worker.mcts = FakeMCTS()
        worker.action_encoder = ActionEncoder()
        worker.last_termination = None
        worker.last_root_noise_calls = 0
        worker.last_telemetry = None
        worker.last_pgn = None
        worker.last_tier = None

        start = sample_curriculum_board(random.Random(0), tier=1)
        # Force the known mate-in-1 board.
        start = type(start)(board=board, tier=1, piece_count=3)
        with mock.patch.object(
            worker_module,
            "sample_curriculum_board",
            return_value=start,
        ):
            with mock.patch.object(worker_module, "advance_root", return_value=None):
                data = worker.play_game()

        self.assertTrue(data)
        self.assertEqual(worker.last_termination, "CHECKMATE")
        self.assertEqual(worker.last_root_noise_calls, 1)
        self.assertEqual(len(worker.mcts.search_calls), 1)
        self.assertTrue(worker.mcts.search_calls[0]["add_noise"])
        self.assertNotIn("force_playouts", worker.mcts.search_calls[0])

    def test_dirichlet_noise_is_mixed_into_root_priors(self) -> None:
        root = Node(prior_prob=1.0)
        root.children = {
            chess.Move.from_uci("e2e4"): Node(0.5, root),
            chess.Move.from_uci("d2d4"): Node(0.5, root),
        }
        with mock.patch(
            "tabula.mcts.search.np.random.dirichlet",
            return_value=[1.0, 0.0],
        ):
            MCTS._add_dirichlet(None, root, alpha=0.3, fraction=0.25)
        priors = [child.prior_prob for child in root.children.values()]
        self.assertEqual(priors, [0.625, 0.375])

    def test_search_does_not_stack_dirichlet_on_same_root(self) -> None:
        """Re-entering search() on one root must not re-noise child priors."""

        class TinyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                b = x.shape[0]
                return torch.zeros(b, 4672), torch.zeros(b, 1)

        model = TinyNet()
        mcts = MCTS(model, num_simulations=2, c_puct=1.25)
        board = chess.Board()
        root = mcts.search(board, encode_board, add_noise=True)
        first = {m: c.prior_prob for m, c in root.children.items()}
        root2 = mcts.search(board, encode_board, add_noise=True, root=root)
        second = {m: c.prior_prob for m, c in root2.children.items()}
        self.assertIs(root2, root)
        self.assertEqual(first, second)
        self.assertTrue(getattr(root, "_dirichlet_applied", False))

    def test_visit_policy_is_normalized_visits(self) -> None:
        root = Node(1.0)
        a = chess.Move.from_uci("e2e4")
        b = chess.Move.from_uci("d2d4")
        root.children[a] = Node(0.5, root)
        root.children[b] = Node(0.5, root)
        root.children[a].visit_count = 25
        root.children[b].visit_count = 75
        policy = visit_policy_target(root)
        self.assertAlmostEqual(policy[a], 0.25)
        self.assertAlmostEqual(policy[b], 0.75)


class TreeReuseTests(unittest.TestCase):
    def test_advance_root_preserves_child_visit_counts(self) -> None:
        root = Node(1.0)
        move = chess.Move.from_uci("e2e4")
        other = chess.Move.from_uci("d2d4")
        child = Node(0.4, root)
        child.visit_count = 17
        sibling = Node(0.3, root)
        sibling.visit_count = 3
        grandchild = Node(0.2, child)
        grandchild.visit_count = 5
        child.children[chess.Move.from_uci("e7e5")] = grandchild
        root.children[move] = child
        root.children[other] = sibling

        new_root = advance_root(root, move)
        self.assertIs(new_root, child)
        self.assertIsNone(new_root.parent)
        self.assertEqual(new_root.visit_count, 17)
        self.assertEqual(
            new_root.children[chess.Move.from_uci("e7e5")].visit_count,
            5,
        )
        # Old root no longer anchors sibling subtrees.
        self.assertEqual(root.children, {})
        self.assertIsNone(sibling.parent)


if __name__ == "__main__":
    unittest.main()

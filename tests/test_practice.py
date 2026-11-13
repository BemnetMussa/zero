import random
import unittest
from types import SimpleNamespace
from unittest import mock

import chess

from tabula.game.action_encoder import ActionEncoder
from tabula.mcts.search import MCTS, Node
from tabula.selfplay import worker as worker_module
from tabula.selfplay.practice import (
    CORNERED_KQK_FENS,
    MATE_IN_ONE_FENS,
    MATE_IN_TWO_FENS,
    random_practice_board,
)
from tabula.selfplay.worker import SelfPlayWorker, _choose_move, _selection_mode


def can_force_mate(board, attacker, plies):
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return outcome.winner == attacker
    if plies == 0:
        return False
    results = []
    for move in board.legal_moves:
        child = board.copy()
        child.push(move)
        results.append(can_force_mate(child, attacker, plies - 1))
    return any(results) if board.turn == attacker else all(results)


class PracticeBoardTests(unittest.TestCase):
    def test_normal_games_sample_all_first_thirty_plies(self) -> None:
        for ply in range(30):
            self.assertEqual(_selection_mode(False, ply, 30), "sample")
        self.assertEqual(_selection_mode(False, 30, 30), "greedy")
        self.assertEqual(_selection_mode(True, 0, 30), "greedy")

    def test_temperature_one_samples_proportional_to_visit_counts(self) -> None:
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
            selected = _choose_move(root, "sample")
        self.assertEqual(selected, second)
        probabilities = choice.call_args.kwargs["p"]
        self.assertEqual(probabilities.tolist(), [0.25, 0.75])

    def test_boards_are_legal_and_strong_side_moves(self) -> None:
        rng = random.Random(0)
        for _ in range(200):
            board, strong = random_practice_board(rng)
            self.assertTrue(board.is_valid())
            self.assertIsNone(board.outcome(claim_draw=True))
            self.assertEqual(board.turn, strong)
            extra = len(board.pieces(chess.QUEEN, strong)) + len(board.pieces(chess.ROOK, strong))
            self.assertEqual(extra, 1)
            self.assertEqual(len(board.piece_map()), 3)

    def test_fixed_mate_in_one_positions_are_forced(self) -> None:
        for fen in MATE_IN_ONE_FENS:
            board = chess.Board(fen)
            self.assertTrue(can_force_mate(board, board.turn, 1), fen)

    def test_fixed_mate_in_two_positions_are_forced_but_not_mate_in_one(self) -> None:
        for fen in MATE_IN_TWO_FENS:
            board = chess.Board(fen)
            attacker = board.turn
            self.assertFalse(can_force_mate(board, attacker, 1), fen)
            self.assertTrue(can_force_mate(board, attacker, 3), fen)

    def test_cornered_kqk_positions_keep_the_weak_king_in_a_corner(self) -> None:
        for fen in CORNERED_KQK_FENS:
            board = chess.Board(fen)
            weak_king = board.king(not board.turn)
            self.assertIn(
                weak_king,
                (chess.A1, chess.A8, chess.H1, chess.H8),
            )
            self.assertEqual(len(board.pieces(chess.QUEEN, board.turn)), 1)

    def test_curriculum_selfplay_uses_noisy_full_search_and_greedy_move(self) -> None:
        board = chess.Board(MATE_IN_ONE_FENS[0])

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
                self.assert_mating_move(mating_move)
                child = SimpleNamespace(visit_count=10)
                return SimpleNamespace(
                    children={mating_move: child},
                    q_value=1.0,
                )

            def pruned_policy_target(self, root):
                return {next(iter(root.children)): 1.0}

            @staticmethod
            def assert_mating_move(move):
                if move is None:
                    raise AssertionError("curriculum board has no mating move")

        worker = object.__new__(SelfPlayWorker)
        worker.mcts = FakeMCTS()
        worker.action_encoder = ActionEncoder()
        with mock.patch.object(
            worker_module,
            "random_practice_board",
            return_value=(board, board.turn),
        ):
            data = worker.play_game(practice=True)

        self.assertTrue(data)
        self.assertEqual(worker.last_termination, "CHECKMATE")
        self.assertEqual(worker.last_root_noise_calls, 1)
        self.assertEqual(len(worker.mcts.search_calls), 1)
        self.assertTrue(worker.mcts.search_calls[0]["add_noise"])
        self.assertTrue(worker.mcts.search_calls[0]["force_playouts"])

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

    def test_forced_root_playouts_do_not_skip_unvisited_moves(self) -> None:
        root = Node(prior_prob=1.0)
        visited_move = chess.Move.from_uci("e2e4")
        unvisited_move = chess.Move.from_uci("d2d4")
        root.children = {
            visited_move: Node(0.9, root),
            unvisited_move: Node(0.1, root),
        }
        root.children[visited_move].visit_count = 10
        selected, child = MCTS.select_child(None, root, force_playouts=True)
        self.assertEqual(selected, unvisited_move)
        self.assertIs(child, root.children[unvisited_move])


if __name__ == "__main__":
    unittest.main()

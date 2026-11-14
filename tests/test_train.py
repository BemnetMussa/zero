import math
import os
import tempfile
import unittest

import numpy as np
import torch

from tabula.game.encoder import INPUT_PLANES
from tabula.train.train import ENTROPY_BONUS, SelfPlayDataset, policy_loss_with_entropy


class PolicyObjectiveTests(unittest.TestCase):
    def test_policy_loss_includes_point_zero_one_entropy_bonus(self) -> None:
        logits = torch.zeros((1, 2), requires_grad=True)
        targets = torch.tensor([[1.0, 0.0]])
        weights = torch.ones(1)

        loss = policy_loss_with_entropy(logits, targets, weights)

        expected = math.log(2.0) * (1.0 - ENTROPY_BONUS)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertIsNotNone(logits.grad)


class ReplayCompatibilityTests(unittest.TestCase):
    def test_skips_stale_13_channel_run2_tensors(self) -> None:
        pi = np.zeros(4672, dtype=np.float32)
        z = np.array([1.0], dtype=np.float32)
        w = np.array([1.0], dtype=np.float32)
        old = (np.zeros((13, 8, 8), dtype=np.float32), pi, z, w)
        new = (np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32), pi, z, w)
        with tempfile.TemporaryDirectory() as tmp:
            torch.save([old, old], os.path.join(tmp, "replay_old.pt"))
            torch.save([new], os.path.join(tmp, "replay_new.pt"))
            dataset = SelfPlayDataset(tmp)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(tuple(dataset.states.shape), (1, INPUT_PLANES, 8, 8))

    def test_all_stale_13_channel_files_yield_empty_buffer(self) -> None:
        pi = np.zeros(4672, dtype=np.float32)
        z = np.array([0.0], dtype=np.float32)
        w = np.array([1.0], dtype=np.float32)
        old = (np.zeros((13, 8, 8), dtype=np.float32), pi, z, w)
        with tempfile.TemporaryDirectory() as tmp:
            torch.save([old], os.path.join(tmp, "replay_old.pt"))
            dataset = SelfPlayDataset(tmp)
        self.assertEqual(len(dataset), 0)
        self.assertEqual(tuple(dataset.states.shape), (0, INPUT_PLANES, 8, 8))


if __name__ == "__main__":
    unittest.main()


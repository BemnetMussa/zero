import math
import unittest

import torch

from tabula.train.train import ENTROPY_BONUS, policy_loss_with_entropy


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


if __name__ == "__main__":
    unittest.main()

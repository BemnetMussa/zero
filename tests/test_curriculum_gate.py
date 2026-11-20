import unittest

from tabula.selfplay.curriculum_gate import CurriculumGate, weights_for_max_tier
from tabula.selfplay.practice import choose_tier


class CurriculumGateTests(unittest.TestCase):
    def test_tier1_only_until_unlock(self) -> None:
        self.assertEqual(weights_for_max_tier(1), {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0})

    def test_weights_sum_to_one_for_each_max_tier(self) -> None:
        for max_tier in (1, 2, 3, 4):
            weights = weights_for_max_tier(max_tier)
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            for tier, weight in weights.items():
                if tier > max_tier:
                    self.assertEqual(weight, 0.0)

    def test_unlock_requires_streak(self) -> None:
        gate = CurriculumGate(max_tier=1, unlock_up=16, drop_down=12, unlock_streak_needed=2)
        first = gate.update_from_exam(18, 20)
        self.assertEqual(first["action"], "streak")
        self.assertEqual(gate.max_tier, 1)
        second = gate.update_from_exam(17, 20)
        self.assertEqual(second["action"], "unlock")
        self.assertEqual(gate.max_tier, 2)
        self.assertEqual(gate.unlock_streak, 0)

    def test_drop_reduces_tier(self) -> None:
        gate = CurriculumGate(max_tier=2, unlock_up=16, drop_down=12)
        event = gate.update_from_exam(10, 20)
        self.assertEqual(event["action"], "drop")
        self.assertEqual(gate.max_tier, 1)

    def test_mid_band_resets_streak(self) -> None:
        gate = CurriculumGate(max_tier=1, unlock_streak=1, unlock_up=16, drop_down=12)
        event = gate.update_from_exam(14, 20)
        self.assertEqual(event["action"], "hold")
        self.assertEqual(gate.unlock_streak, 0)

    def test_choose_tier_respects_zero_weights(self) -> None:
        class FakeRng:
            def random(self) -> float:
                return 0.99

        tier = choose_tier(FakeRng(), weights={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0})
        self.assertEqual(tier, 1)

    def test_from_state_roundtrip(self) -> None:
        gate = CurriculumGate(max_tier=3, unlock_streak=1)
        restored = CurriculumGate.from_state(gate.to_state())
        self.assertEqual(restored.max_tier, 3)
        self.assertEqual(restored.unlock_streak, 1)


if __name__ == "__main__":
    unittest.main()

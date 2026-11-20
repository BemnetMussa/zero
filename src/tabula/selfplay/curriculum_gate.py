"""Bidirectional curriculum gate for Run #4.

Unlock the next piece-count tier only when the mate exam is good enough.
Drop back if the exam collapses. Protects the easy-checkmate skill from
catastrophic forgetting when harder boards enter training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


# Locked 2026-11-21 (wiki Development journey Run #4).
DEFAULT_UNLOCK_UP = 16  # mates / 20
DEFAULT_DROP_DOWN = 12  # mates / 20
DEFAULT_UNLOCK_STREAK = 2  # consecutive exams at/above unlock_up


def weights_for_max_tier(max_tier: int) -> dict[int, float]:
    """Sampling mix given the highest unlocked tier.

    Tier 1 only until the gate opens. After that, keep a slice of easier
    boards so free throws are not erased while half-court is added.
    """
    m = max(1, min(4, int(max_tier)))
    if m == 1:
        return {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}
    if m == 2:
        return {1: 0.35, 2: 0.65, 3: 0.0, 4: 0.0}
    if m == 3:
        return {1: 0.20, 2: 0.30, 3: 0.50, 4: 0.0}
    return {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.25}


@dataclass
class CurriculumGate:
    """Tracks which board-size tier may appear in self-play."""

    max_tier: int = 1
    unlock_streak: int = 0
    unlock_up: int = DEFAULT_UNLOCK_UP
    drop_down: int = DEFAULT_DROP_DOWN
    unlock_streak_needed: int = DEFAULT_UNLOCK_STREAK
    last_exam_mates: int | None = None
    last_exam_n: int | None = None
    enabled: bool = True

    def weights(self) -> dict[int, float]:
        if not self.enabled:
            # Legacy Run #3 flat mix (kept for explicit opt-out).
            return {1: 0.20, 2: 0.45, 3: 0.25, 4: 0.10}
        return weights_for_max_tier(self.max_tier)

    def to_state(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, state: dict | None, **overrides) -> "CurriculumGate":
        gate = cls()
        if state:
            for key in (
                "max_tier",
                "unlock_streak",
                "unlock_up",
                "drop_down",
                "unlock_streak_needed",
                "last_exam_mates",
                "last_exam_n",
                "enabled",
            ):
                if key in state and state[key] is not None:
                    setattr(gate, key, state[key])
        for key, value in overrides.items():
            if value is not None and hasattr(gate, key):
                setattr(gate, key, value)
        gate.max_tier = max(1, min(4, int(gate.max_tier)))
        gate.unlock_streak = max(0, int(gate.unlock_streak))
        if gate.drop_down >= gate.unlock_up:
            raise ValueError(
                f"drop_down ({gate.drop_down}) must be < unlock_up ({gate.unlock_up})"
            )
        return gate

    def update_from_exam(self, mates: int, n: int = 20) -> dict:
        """Apply one mate-exam result. Returns a small event dict for metrics."""

        mates = int(mates)
        n = int(n)
        self.last_exam_mates = mates
        self.last_exam_n = n
        event = {
            "event": "curriculum_gate",
            "mates": mates,
            "n": n,
            "max_tier_before": self.max_tier,
            "unlock_streak_before": self.unlock_streak,
            "action": "hold",
        }
        if not self.enabled:
            event["action"] = "disabled"
            event["max_tier_after"] = self.max_tier
            event["unlock_streak_after"] = self.unlock_streak
            event["weights"] = self.weights()
            return event

        if mates <= self.drop_down:
            if self.max_tier > 1:
                self.max_tier -= 1
                event["action"] = "drop"
            else:
                event["action"] = "hold_floor"
            self.unlock_streak = 0
        elif mates >= self.unlock_up:
            self.unlock_streak += 1
            if self.unlock_streak >= self.unlock_streak_needed and self.max_tier < 4:
                self.max_tier += 1
                self.unlock_streak = 0
                event["action"] = "unlock"
            else:
                event["action"] = "streak"
        else:
            # Between drop and unlock: freeze streak so we don't climb on noise.
            self.unlock_streak = 0
            event["action"] = "hold"

        event["max_tier_after"] = self.max_tier
        event["unlock_streak_after"] = self.unlock_streak
        event["weights"] = self.weights()
        return event

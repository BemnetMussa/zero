"""Warm-start helpers when the board tensor grows (13 → 20 planes)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tabula.game.encoder import INPUT_PLANES

START_CONV_KEY = "start_block.0.weight"


def adapt_model_state_dict(
    model: nn.Module,
    state: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Copy pretrained conv1 channels and zero-pad new planes when shapes grow.

    Returns (adapted_state, padded) where padded is True if input channels grew.
    """
    adapted = {k: v for k, v in state.items()}
    if START_CONV_KEY not in adapted or START_CONV_KEY not in model.state_dict():
        return adapted, False

    old_weight = adapted[START_CONV_KEY]
    new_weight = model.state_dict()[START_CONV_KEY]

    if old_weight.shape == new_weight.shape:
        return adapted, False

    if (
        old_weight.ndim != 4
        or new_weight.ndim != 4
        or old_weight.shape[0] != new_weight.shape[0]
        or old_weight.shape[2:] != new_weight.shape[2:]
    ):
        raise ValueError(
            f"Cannot warm-start conv weights: old {tuple(old_weight.shape)} "
            f"vs new {tuple(new_weight.shape)}"
        )

    old_in = int(old_weight.shape[1])
    new_in = int(new_weight.shape[1])
    if old_in > new_in:
        raise ValueError(f"Cannot shrink input planes from {old_in} to {new_in}")

    padded = new_weight.detach().clone()
    padded.zero_()
    padded[:, :old_in] = old_weight.to(dtype=padded.dtype, device="cpu")
    adapted[START_CONV_KEY] = padded
    return adapted, True


def load_model_weights(
    model: nn.Module,
    state: dict[str, Any] | None,
    *,
    strict: bool = True,
) -> bool:
    """Load weights into model, padding 13→20 input channels when needed.

    Returns True when input-channel padding was applied.
    """
    if state is None:
        return False
    adapted, padded = adapt_model_state_dict(model, state)
    model.load_state_dict(adapted, strict=strict)
    return padded


def load_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint: dict[str, Any],
) -> bool:
    """Restore a training checkpoint, dropping optimizer/scheduler on shape change."""

    padded = load_model_weights(model, checkpoint["model_state"])
    if padded:
        # Fresh optimizer/scheduler state: Adam moments do not match new channels.
        return True
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return False


def assert_input_planes(model: nn.Module, expected: int = INPUT_PLANES) -> None:
    weight = model.state_dict()[START_CONV_KEY]
    if weight.shape[1] != expected:
        raise ValueError(f"model expects {weight.shape[1]} input planes, want {expected}")

"""Learning rate schedules for coregrad (WSD and cosine)."""

from __future__ import annotations

import math


def wsd_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
    decay_style: str = "cosine",
) -> float:
    """Warmup-Stable-Decay (WSD) learning rate schedule.

    - [0, warmup_steps): linear warmup from 0 -> peak_lr
    - [warmup_steps, total_steps - decay_steps): constant peak_lr
    - [total_steps - decay_steps, total_steps]: decay peak_lr -> min_lr
    """
    if step < 0:
        return 0.0
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)

    decay_start = max(warmup_steps, total_steps - max(decay_steps, 0))
    if decay_steps <= 0 or step < decay_start:
        return peak_lr

    progress = min(1.0, float(step - decay_start) / float(max(decay_steps, 1)))
    if decay_style == "linear":
        coeff = 1.0 - progress
    elif decay_style == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported decay_style: {decay_style!r}")
    return min_lr + coeff * (peak_lr - min_lr)

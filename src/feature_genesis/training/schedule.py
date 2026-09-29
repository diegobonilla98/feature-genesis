from __future__ import annotations

import math


def cosine_learning_rate(
    step: int,
    total_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_learning_rate * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return minimum_learning_rate + (peak_learning_rate - minimum_learning_rate) * cosine

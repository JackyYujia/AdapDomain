"""Map raw msprof counters into [0, 1] coordinates per domain."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterRange:
    minimum: float
    saturation: float


def normalize(value: float, rng: CounterRange) -> float:
    if rng.saturation <= rng.minimum:
        raise ValueError("saturation must be > minimum")
    norm = (value - rng.minimum) / (rng.saturation - rng.minimum)
    return float(max(0.0, min(1.0, norm)))

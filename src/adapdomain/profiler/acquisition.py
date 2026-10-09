"""ActiveVC acquisition function (paper Eq. activevc_acquisition)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .candidate_grid import GridPoint


@dataclass(frozen=True)
class AcquisitionWeights:
    alpha: float
    beta: float
    gamma: float
    tau: float


def acquisition(
    *,
    candidate: GridPoint,
    uncertainty: float,
    distance_to_frontier: float,
    marginal_goodput_gain: float,
    weights: AcquisitionWeights,
) -> float:
    return (
        weights.alpha * uncertainty
        + weights.beta * math.exp(-distance_to_frontier / max(weights.tau, 1e-6))
        + weights.gamma * marginal_goodput_gain
    )


def best_candidate(
    candidates: Iterable[GridPoint],
    *,
    uncertainty_fn,
    distance_fn,
    gain_fn,
    weights: AcquisitionWeights,
) -> GridPoint:
    best: tuple[GridPoint, float] | None = None
    for c in candidates:
        score = acquisition(
            candidate=c,
            uncertainty=uncertainty_fn(c),
            distance_to_frontier=distance_fn(c),
            marginal_goodput_gain=gain_fn(c),
            weights=weights,
        )
        if best is None or score > best[1]:
            best = (c, score)
    assert best is not None
    return best[0]

"""ActiveVC frontier-recovery baselines and HarmonicFit scoring ablations."""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from adapdomain.profiler.activevc import ActiveVC, ActiveVCConfigRuntime
from adapdomain.profiler.acquisition import AcquisitionWeights
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights, compute_score


@dataclass(frozen=True)
class FrontierRecoveryResult:
    name: str
    budget: int
    boundary_error: float
    missed_feasible: int


def random_baseline(*, budget: int, evaluate: Callable[[GridPoint], MsprofObservation], rng: random.Random) -> list[MsprofObservation]:
    obs: list[MsprofObservation] = []
    for _ in range(budget):
        p = GridPoint(rng.random(), rng.random(), rng.random())
        obs.append(evaluate(p))
    return obs


def binary_baseline(*, budget: int, evaluate: Callable[[GridPoint], MsprofObservation]) -> list[MsprofObservation]:
    obs: list[MsprofObservation] = []
    coords = []
    n = max(2, int(budget ** 0.5))
    for i in range(n):
        for j in range(n):
            coords.append(GridPoint(i / max(n - 1, 1), j / max(n - 1, 1), 0.5))
    for p in coords[:budget]:
        obs.append(evaluate(p))
    return obs


def bo_baseline(*, budget: int, evaluate: Callable[[GridPoint], MsprofObservation], rng: random.Random) -> list[MsprofObservation]:
    # Simple BO surrogate: random first, then exploit-only
    obs = random_baseline(budget=max(1, budget // 2), evaluate=evaluate, rng=rng)
    feasible = [o for o in obs if o.feasible]
    if feasible:
        center = (
            statistics.mean(o.raw_vector for o in feasible),
            statistics.mean(o.raw_cube for o in feasible),
        )
    else:
        center = (0.5, 0.5)
    while len(obs) < budget:
        jitter_v = max(0.0, min(1.0, center[0] + rng.gauss(0.0, 0.05)))
        jitter_m = max(0.0, min(1.0, center[1] + rng.gauss(0.0, 0.05)))
        obs.append(evaluate(GridPoint(jitter_v, jitter_m, 0.5)))
    return obs


def boundary_error(observations: Sequence[MsprofObservation]) -> float:
    feasible = [o for o in observations if o.feasible]
    infeasible = [o for o in observations if not o.feasible]
    if not feasible or not infeasible:
        return 1.0
    import math
    return min(
        math.hypot(f.raw_vector - i.raw_vector, f.raw_cube - i.raw_cube)
        for f in feasible for i in infeasible
    )


def harmonicfit_variant_score(
    *,
    variant: str,
    rho: list[float],
    risks: RiskTerms,
    weights: ScoreWeights,
) -> float:
    masked_weights = weights
    masked_risks = risks
    if variant == "no_ceff":
        masked_weights = ScoreWeights(0.0, weights.lambda_x, weights.lambda_h, weights.lambda_q, weights.lambda_f, weights.lambda_u)
    elif variant == "no_interference":
        masked_risks = RiskTerms(x=0.0, h=risks.h, q=risks.q, f=risks.f, u=risks.u)
    elif variant == "no_hbm":
        masked_risks = RiskTerms(x=risks.x, h=0.0, q=risks.q, f=risks.f, u=risks.u)
    elif variant == "no_fragment":
        masked_risks = RiskTerms(x=risks.x, h=risks.h, q=risks.q, f=0.0, u=risks.u)
    elif variant == "no_empty_penalty":
        masked_risks = RiskTerms(x=risks.x, h=risks.h, q=risks.q, f=risks.f, u=0.0)
    return compute_score(rho=rho, weights=masked_weights, risks=masked_risks).final

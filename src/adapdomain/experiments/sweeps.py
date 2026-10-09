"""Stress sweeps over arrival rate, SLO threshold, workload mix, and tenant churn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from adapdomain.experiments.replay import ReplayInputs, ReplayResult, run_replay
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights


@dataclass(frozen=True)
class SweepPoint:
    factor_name: str
    factor_value: float
    result: ReplayResult


def sweep_arrival_rate(
    *,
    base_inputs: ReplayInputs,
    rates: Iterable[float],
    build_inputs: Callable[[ReplayInputs, float], ReplayInputs],
    seeds: list[int],
    weights: ScoreWeights,
    feasibility: FeasibilityFilter,
    eta: float,
    backend,
    output_root: str,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for r in rates:
        scaled = build_inputs(base_inputs, r)
        res = run_replay(
            inputs=scaled, seeds=seeds, weights=weights, feasibility=feasibility,
            eta=eta, backend=backend, output_root=f"{output_root}/arrival_{r}",
        )
        points.append(SweepPoint(factor_name="arrival_rate", factor_value=r, result=res))
    return points

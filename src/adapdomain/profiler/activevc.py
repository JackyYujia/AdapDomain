"""ActiveVC main loop.

Stages:
1. sparse seed across a 4x4x3 V:M:HBM grid
2. boundary refinement at 8x8 in V:M holding HBM at the median safe value
3. uncertainty-guided sampling using `acquisition.best_candidate`

This module never imports `acl` or `msprof`; the caller passes an
`evaluate(point) -> MsprofObservation`-like callable.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable

from adapdomain.core.frontier import FrontierSnapshot

from .acquisition import AcquisitionWeights, acquisition
from .candidate_grid import CandidateGrid, GridPoint
from .frontier_builder import FrontierBuilder, MsprofObservation


@dataclass
class ActiveVCConfigRuntime:
    budget: int
    weights: AcquisitionWeights
    seed_dims: tuple[int, int, int] = (4, 4, 3)
    boundary_dims: tuple[int, int] = (8, 8)


@dataclass
class ActiveVCRunOutput:
    snapshot: FrontierSnapshot
    probed_points: list[MsprofObservation]
    spent_budget: int


class ActiveVC:
    def __init__(
        self,
        *,
        config: ActiveVCConfigRuntime,
        builder: FrontierBuilder,
    ) -> None:
        self._config = config
        self._builder = builder

    def run(
        self,
        *,
        model_name: str,
        version: int,
        source_run_id: str,
        evaluate: Callable[[GridPoint], MsprofObservation],
    ) -> ActiveVCRunOutput:
        grid = CandidateGrid(seed_dims=self._config.seed_dims, boundary_dims=self._config.boundary_dims)
        observations: list[MsprofObservation] = []
        seen: set[tuple[float, float, float]] = set()

        budget_left = self._config.budget

        # Stage 1: sparse seeds (cap by budget)
        for seed in self._sparse_seeds(grid, max_count=min(budget_left, 8)):
            obs = evaluate(seed)
            observations.append(obs)
            seen.add((seed.v, seed.m, seed.h))
            budget_left -= 1
            if budget_left == 0:
                return self._finalize(model_name, version, source_run_id, observations, spent=self._config.budget)

        # Stage 2: V:M boundary refinement at median safe HBM
        feasible_hbm = [o.raw_hbm for o in observations if o.feasible]
        median_hbm = statistics.median(feasible_hbm) if feasible_hbm else 0.5
        for point in self._boundary_targets(grid, hbm=median_hbm, observations=observations, max_count=min(budget_left, 8)):
            key = (point.v, point.m, point.h)
            if key in seen:
                continue
            obs = evaluate(point)
            observations.append(obs)
            seen.add(key)
            budget_left -= 1
            if budget_left == 0:
                return self._finalize(model_name, version, source_run_id, observations, spent=self._config.budget)

        # Stage 3: uncertainty-guided
        while budget_left > 0:
            candidates = [p for p in grid.boundary_points(hbm=median_hbm) if (p.v, p.m, p.h) not in seen]
            if not candidates:
                break

            def uncertainty(c: GridPoint) -> float:
                return self._uncertainty(observations, c)

            def distance(c: GridPoint) -> float:
                return self._distance_to_frontier(observations, c)

            def gain(c: GridPoint) -> float:
                return self._marginal_gain(observations, c)

            best = max(
                candidates,
                key=lambda c: acquisition(
                    candidate=c,
                    uncertainty=uncertainty(c),
                    distance_to_frontier=distance(c),
                    marginal_goodput_gain=gain(c),
                    weights=self._config.weights,
                ),
            )
            obs = evaluate(best)
            observations.append(obs)
            seen.add((best.v, best.m, best.h))
            budget_left -= 1

        return self._finalize(model_name, version, source_run_id, observations, spent=self._config.budget - budget_left)

    @staticmethod
    def _sparse_seeds(grid: CandidateGrid, *, max_count: int) -> list[GridPoint]:
        seeds = list(grid.seed_points())
        if len(seeds) <= max_count:
            return seeds
        step = max(1, len(seeds) // max_count)
        return seeds[::step][:max_count]

    @staticmethod
    def _boundary_targets(grid: CandidateGrid, *, hbm: float, observations: list[MsprofObservation], max_count: int) -> list[GridPoint]:
        # pick points near the empirical boundary by alternating feasible/infeasible neighbours
        candidates = list(grid.boundary_points(hbm=hbm))
        candidates.sort(key=lambda c: abs(0.5 - c.v) + abs(0.5 - c.m))
        return candidates[:max_count]

    @staticmethod
    def _uncertainty(obs: list[MsprofObservation], c: GridPoint) -> float:
        # 1 / (1 + distance to nearest observation)
        if not obs:
            return 1.0
        d = min(math.hypot(c.v - o.raw_vector, c.m - o.raw_cube) for o in obs)
        return 1.0 / (1.0 + d)

    @staticmethod
    def _distance_to_frontier(obs: list[MsprofObservation], c: GridPoint) -> float:
        feasible = [o for o in obs if o.feasible]
        infeasible = [o for o in obs if not o.feasible]
        if not feasible or not infeasible:
            return 0.5
        return min(math.hypot(c.v - o.raw_vector, c.m - o.raw_cube) for o in feasible + infeasible)

    @staticmethod
    def _marginal_gain(obs: list[MsprofObservation], c: GridPoint) -> float:
        feasible = [o for o in obs if o.feasible]
        if not feasible:
            return 0.0
        best = max(o.goodput_qps for o in feasible)
        return max(0.0, best * (1.0 - 0.5 * (abs(c.v - 0.5) + abs(c.m - 0.5))))

    def _finalize(self, model_name: str, version: int, source_run_id: str, observations: list[MsprofObservation], *, spent: int) -> ActiveVCRunOutput:
        snap = self._builder.build(
            model_name=model_name,
            version=version,
            observations=observations,
            source_run_id=source_run_id,
        )
        return ActiveVCRunOutput(snapshot=snap, probed_points=observations, spent_budget=spent)

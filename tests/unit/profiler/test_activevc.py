import math

import pytest

from adapdomain.profiler.activevc import ActiveVC, ActiveVCConfigRuntime
from adapdomain.profiler.acquisition import AcquisitionWeights
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation


def _synthetic_eval(point: GridPoint) -> MsprofObservation:
    feasible = point.v + point.m <= 1.2 and point.h <= 0.9
    p95 = 5.0 + 20.0 * max(0.0, point.v + point.m - 1.0)
    return MsprofObservation(
        model="m", point_id=f"v{point.v:.2f}-m{point.m:.2f}-h{point.h:.2f}",
        raw_cube=point.m, raw_vector=point.v, raw_hbm=point.h,
        batch_size=1, slo_ms=20.0,
        p50_ms=p95 / 2, p95_ms=p95, p99_ms=p95 * 1.1,
        goodput_qps=100.0 if feasible else 5.0,
        feasible=feasible,
    )


def test_activevc_runs_within_budget():
    cfg = ActiveVCConfigRuntime(
        budget=24,
        weights=AcquisitionWeights(alpha=0.40, beta=0.40, gamma=0.20, tau=0.15),
    )
    builder = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0))
    out = ActiveVC(config=cfg, builder=builder).run(
        model_name="m", version=1, source_run_id="syn", evaluate=_synthetic_eval,
    )
    assert out.spent_budget == 24
    assert any(p.feasible for p in out.snapshot.points)
    assert any(not p.feasible for p in out.snapshot.points)


def test_smaller_budget_runs_to_completion():
    cfg = ActiveVCConfigRuntime(
        budget=8,
        weights=AcquisitionWeights(alpha=0.40, beta=0.40, gamma=0.20, tau=0.15),
    )
    builder = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0))
    out = ActiveVC(config=cfg, builder=builder).run(
        model_name="m", version=1, source_run_id="syn", evaluate=_synthetic_eval,
    )
    assert out.spent_budget == 8

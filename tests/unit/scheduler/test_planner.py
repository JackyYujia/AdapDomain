from datetime import datetime, timezone

import pytest

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.planner import HarmonicFitPlanner
from adapdomain.scheduler.residual_state import ClusterState
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights


def _snap(model: str, cube: float, vec: float) -> FrontierSnapshot:
    pt = FrontierPoint(
        model_name=model, point_id="p",
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=cube, norm_vector=vec,
        batch_size=1, slo_ms=15.0,
        p50_ms=4.0, p95_ms=6.0, p99_ms=7.0, goodput_qps=100.0,
        feasible=True, confidence=0.9,
    )
    return FrontierSnapshot(model_name=model, version=1, k_domains=["cube", "vector"],
                            points=[pt], safe_point_id=pt.point_id, dominant_domain="balanced",
                            confidence_summary=0.9, source_run_id="r",
                            created_at=datetime.now(timezone.utc))


def test_planner_prefers_complementary_device():
    state = ClusterState.from_residuals([
        DeviceResidual(device_id=0, residual={"cube": 0.6, "vector": 0.2}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
        DeviceResidual(device_id=1, residual={"cube": 0.2, "vector": 0.6}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
    ])
    replicas = [TenantShape(model_name="bert-base", replica_id="r0",
                            domains={"cube": 0.5, "vector": 0.1},
                            slo_ms=15.0, batch_size=1, frontier_version=1, confidence=0.9)]
    planner = HarmonicFitPlanner(
        feasibility=FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5),
        weights=ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05),
        eta=0.30,
        hbm_high=0.85, queue_high=0.80,
    )
    snapshots = {"bert-base": _snap("bert-base", cube=0.5, vec=0.1)}
    plan = planner.plan(replicas, state=state, snapshots=snapshots, interference=InterferenceTable.from_entries([]))
    assert plan.placements["r0"] == 0  # cube-heavy slack lives on device 0

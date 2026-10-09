import time
from datetime import datetime, timezone

import pytest

from adapdomain.core.frontier import (
    ConfidenceModel,
    FrontierPoint,
    FrontierQuery,
    FrontierSnapshot,
)


def _fp(point_id: str, p95: float, batch: int, feasible: bool = True, *, conf: float = 1.0) -> FrontierPoint:
    return FrontierPoint(
        model_name="m",
        point_id=point_id,
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=0.3, norm_vector=0.3,
        batch_size=batch, slo_ms=20.0,
        p50_ms=p95 / 2, p95_ms=p95, p99_ms=p95 * 1.1,
        goodput_qps=100.0,
        feasible=feasible, confidence=conf,
    )


def test_snapshot_basic_invariants():
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=[_fp("a", 10, 1), _fp("b", 20, 1, feasible=False)],
        safe_point_id="a", dominant_domain="balanced",
        confidence_summary=0.8, source_run_id="run-0",
    )
    assert snap.safe_point().point_id == "a"
    assert len(snap.feasible_points()) == 1


def test_frontier_query_picks_lowest_p95_under_slo():
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=[_fp("a", 8, 1), _fp("b", 15, 1), _fp("c", 25, 1, feasible=False)],
        safe_point_id="a", dominant_domain="balanced",
        confidence_summary=0.9, source_run_id="r",
    )
    point = FrontierQuery(snap).pick(batch_size=1, slo_ms=20.0)
    assert point.point_id == "a"


def test_frontier_query_returns_none_when_no_feasible_match():
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=[_fp("a", 30, 1, feasible=False)],
        safe_point_id="a", dominant_domain="balanced",
        confidence_summary=0.5, source_run_id="r",
    )
    assert FrontierQuery(snap).pick(batch_size=1, slo_ms=20.0) is None


def test_query_under_one_ms_on_100_points():
    pts = [_fp(f"p{i}", float(i), 1) for i in range(100)]
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=pts, safe_point_id="p0", dominant_domain="balanced",
        confidence_summary=1.0, source_run_id="r",
    )
    q = FrontierQuery(snap)
    start = time.perf_counter()
    for _ in range(50):
        q.pick(batch_size=1, slo_ms=50.0)
    elapsed = (time.perf_counter() - start) / 50
    assert elapsed < 1e-3


def test_confidence_decays_with_staleness():
    now = datetime.now(timezone.utc)
    fresh = ConfidenceModel().score(
        neighbor_count=10, recent_deviations=[0.0], created_at=now, now=now,
    )
    stale = ConfidenceModel().score(
        neighbor_count=10, recent_deviations=[0.0],
        created_at=now.replace(year=now.year - 1), now=now,
    )
    assert fresh > stale
    assert 0.0 <= stale <= 1.0


def test_confidence_drops_when_recent_deviations_large():
    now = datetime.now(timezone.utc)
    cm = ConfidenceModel()
    low_drift = cm.score(neighbor_count=10, recent_deviations=[0.01, 0.02], created_at=now, now=now)
    high_drift = cm.score(neighbor_count=10, recent_deviations=[0.5, 0.6], created_at=now, now=now)
    assert low_drift > high_drift

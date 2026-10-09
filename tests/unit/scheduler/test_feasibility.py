import pytest

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.feasibility import FeasibilityFilter, RejectReason


def _shape(cube=0.4, vec=0.2, slo=15.0, batch=1) -> TenantShape:
    return TenantShape(model_name="bert-base", replica_id="r0",
                       domains={"cube": cube, "vector": vec},
                       slo_ms=slo, batch_size=batch,
                       frontier_version=1, confidence=0.9)


def _residual(cube=1.0, vec=1.0, hbm=0.5, queue=0.5) -> DeviceResidual:
    return DeviceResidual(device_id=0, residual={"cube": cube, "vector": vec},
                          placed_replicas=[], hbm_pressure=hbm, queue_pressure=queue)


def _snapshot_with_point(cube=0.4, vec=0.2, slo=15.0, p95=8.0, feasible=True) -> FrontierSnapshot:
    pt = FrontierPoint(
        model_name="bert-base", point_id="p0",
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=cube, norm_vector=vec,
        batch_size=1, slo_ms=slo,
        p50_ms=p95/2, p95_ms=p95, p99_ms=p95*1.1,
        goodput_qps=100.0, feasible=feasible, confidence=0.9,
    )
    return FrontierSnapshot(
        model_name="bert-base", version=1, k_domains=["cube", "vector"],
        points=[pt], safe_point_id=pt.point_id, dominant_domain="balanced",
        confidence_summary=0.9, source_run_id="r",
    )


def test_passes_when_all_thresholds_satisfied():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(), _snapshot_with_point(), _residual())
    assert ok and reason is None


def test_rejects_when_cube_residual_low():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(cube=0.6), _snapshot_with_point(), _residual(cube=0.4))
    assert not ok and reason is RejectReason.CUBE


def test_rejects_when_vector_residual_low():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(vec=0.3), _snapshot_with_point(), _residual(vec=0.2))
    assert not ok and reason is RejectReason.VECTOR


def test_rejects_when_hbm_pressure_high():
    ff = FeasibilityFilter(hbm_max=0.4, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(), _snapshot_with_point(), _residual(hbm=0.7))
    assert not ok and reason is RejectReason.HBM


def test_rejects_when_queue_pressure_high():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.3, confidence_min=0.5)
    ok, reason = ff.check(_shape(), _snapshot_with_point(), _residual(queue=0.6))
    assert not ok and reason is RejectReason.QUEUE


def test_rejects_when_confidence_too_low():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.9)
    snap = _snapshot_with_point()
    pt = snap.points[0].model_copy(update={"confidence": 0.5})
    snap = snap.model_copy(update={"points": [pt]})
    ok, reason = ff.check(_shape(), snap, _residual())
    assert not ok and reason is RejectReason.CONFIDENCE


def test_rejects_when_no_frontier_point_under_slo():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    snap = _snapshot_with_point(p95=50.0)
    ok, reason = ff.check(_shape(slo=10.0), snap, _residual())
    assert not ok and reason is RejectReason.SLO

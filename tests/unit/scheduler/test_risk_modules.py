import pytest

from adapdomain.core.interference import InterferenceEntry, InterferenceTable
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.fragmentation import fragmentation_risk
from adapdomain.scheduler.hbm_risk import hbm_risk
from adapdomain.scheduler.interference_risk import interference_risk
from adapdomain.scheduler.queue_risk import queue_risk


def _residual(cube=1.0, vec=1.0, hbm=0.0, queue=0.0) -> DeviceResidual:
    return DeviceResidual(device_id=0, residual={"cube": cube, "vector": vec},
                          placed_replicas=[], hbm_pressure=hbm, queue_pressure=queue)


def _shape(cube=0.5, vec=0.3) -> TenantShape:
    return TenantShape(model_name="m", replica_id="r0",
                       domains={"cube": cube, "vector": vec},
                       slo_ms=15.0, batch_size=1, frontier_version=1, confidence=0.9)


def test_hbm_risk_zero_when_no_pressure():
    assert hbm_risk(_residual(hbm=0.0), high_threshold=0.85) == 0.0


def test_hbm_risk_grows_with_pressure():
    assert hbm_risk(_residual(hbm=0.9), high_threshold=0.85) > 0


def test_queue_risk_capped_at_one():
    assert queue_risk(_residual(queue=2.0), high_threshold=0.5) == 1.0


def test_interference_risk_uses_max_plus_damped_sum():
    entries = [
        InterferenceEntry(model_a="X", model_b="A", direction="a_on_b", pressure_bucket="mid", slowdown=0.3, p95_delta=0, goodput_delta=0, source_run_id="r"),
        InterferenceEntry(model_a="X", model_b="B", direction="a_on_b", pressure_bucket="mid", slowdown=0.1, p95_delta=0, goodput_delta=0, source_run_id="r"),
    ]
    table = InterferenceTable.from_entries(entries)
    r = interference_risk(candidate="X", colocated=["A", "B"], bucket="mid", table=table, eta=0.30)
    assert r == pytest.approx(0.3 + 0.30 * (0.3 + 0.1))


def test_fragmentation_risk_increases_when_residual_skewed():
    balanced = fragmentation_risk(_residual(cube=0.5, vec=0.5), _shape(cube=0.4, vec=0.4))
    skewed = fragmentation_risk(_residual(cube=0.05, vec=0.95), _shape(cube=0.05, vec=0.4))
    assert skewed > balanced

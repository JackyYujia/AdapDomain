from datetime import datetime, timezone

import pytest

from adapdomain.core.telemetry import (
    DeviceTelemetry,
    TelemetryWindow,
)


def test_device_telemetry_validation():
    t = DeviceTelemetry(
        device_id=0,
        cube_util=0.5, vector_util=0.3,
        hbm_pressure=0.4, queue_depth=3,
        p50_ms=2.0, p95_ms=4.0, p99_ms=5.0,
        goodput_qps=120.0,
        timestamp=datetime.now(timezone.utc),
    )
    assert t.queue_depth == 3
    with pytest.raises(Exception):
        DeviceTelemetry(
            device_id=0,
            cube_util=-0.1, vector_util=0.0, hbm_pressure=0.0, queue_depth=0,
            p50_ms=0, p95_ms=0, p99_ms=0, goodput_qps=0,
            timestamp=datetime.now(timezone.utc),
        )


def test_ewma_converges():
    w = TelemetryWindow(alpha=0.5)
    series = [10.0] * 30
    for v in series:
        w.observe("p95", v)
    assert w.ewma("p95") == pytest.approx(10.0, abs=1e-3)


def test_residual_drift_signed():
    w = TelemetryWindow(alpha=0.5)
    for v in [0.20, 0.22, 0.18]:
        w.observe("cube_util", v)
    assert w.residual_drift("cube_util", baseline=0.20) == pytest.approx(w.ewma("cube_util") - 0.20)

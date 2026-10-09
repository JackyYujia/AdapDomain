from datetime import datetime, timezone

from adapdomain.backends.base import DeviceTelemetrySample
from adapdomain.runtime.monitor import RuntimeMonitor


def test_monitor_records_drift_against_baseline():
    mon = RuntimeMonitor(alpha=0.5)
    for cube in [0.4, 0.5, 0.6, 0.55]:
        mon.observe_sample(DeviceTelemetrySample(device_id=0, cube_util=cube, vector_util=0.2, hbm_pressure=0.3, queue_depth=1))
    drift = mon.cube_drift(device_id=0, baseline=0.4)
    assert drift > 0

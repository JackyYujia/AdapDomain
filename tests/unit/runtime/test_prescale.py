from adapdomain.core.shape import DeviceResidual
from adapdomain.runtime.prescale import PreScale


def test_prescale_picks_device_with_matching_scarce_domain():
    state = [
        DeviceResidual(device_id=0, residual={"cube": 0.7, "vector": 0.1}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
        DeviceResidual(device_id=1, residual={"cube": 0.1, "vector": 0.7}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
    ]
    ps = PreScale()
    assert ps.pick_for(scarce_domain="cube", residuals=state) == 0
    assert ps.pick_for(scarce_domain="vector", residuals=state) == 1

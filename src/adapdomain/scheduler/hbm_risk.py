"""HBM pressure -> [0,1] risk."""
from __future__ import annotations

from adapdomain.core.shape import DeviceResidual


def hbm_risk(device: DeviceResidual, *, high_threshold: float) -> float:
    if device.hbm_pressure <= 0:
        return 0.0
    return min(1.0, max(0.0, (device.hbm_pressure - high_threshold) / max(1.0 - high_threshold, 1e-6)) + 0.5 * device.hbm_pressure)

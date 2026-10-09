"""Queue pressure -> [0,1] risk."""
from __future__ import annotations

from adapdomain.core.shape import DeviceResidual


def queue_risk(device: DeviceResidual, *, high_threshold: float) -> float:
    if device.queue_pressure <= 0:
        return 0.0
    over = device.queue_pressure - high_threshold
    return min(1.0, max(0.0, over) + 0.5 * device.queue_pressure)

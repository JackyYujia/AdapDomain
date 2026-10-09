"""Edge feasibility filter for HarmonicFit planning."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from adapdomain.core.frontier import FrontierQuery, FrontierSnapshot
from adapdomain.core.shape import DeviceResidual, TenantShape


class RejectReason(str, Enum):
    SLO = "no_feasible_frontier_for_slo"
    CUBE = "insufficient_cube_residual"
    VECTOR = "insufficient_vector_residual"
    HBM = "hbm_pressure_above_threshold"
    QUEUE = "queue_pressure_above_threshold"
    CONFIDENCE = "frontier_confidence_below_threshold"


@dataclass(frozen=True)
class FeasibilityFilter:
    hbm_max: float
    queue_max: float
    confidence_min: float

    def check(
        self,
        replica: TenantShape,
        snapshot: FrontierSnapshot,
        device: DeviceResidual,
    ) -> tuple[bool, RejectReason | None]:
        point = FrontierQuery(snapshot).pick(batch_size=replica.batch_size, slo_ms=replica.slo_ms)
        if point is None:
            return False, RejectReason.SLO
        if point.confidence < self.confidence_min:
            return False, RejectReason.CONFIDENCE
        if replica.domains.get("cube", 0) > device.residual.get("cube", 0):
            return False, RejectReason.CUBE
        if replica.domains.get("vector", 0) > device.residual.get("vector", 0):
            return False, RejectReason.VECTOR
        if device.hbm_pressure > self.hbm_max:
            return False, RejectReason.HBM
        if device.queue_pressure > self.queue_max:
            return False, RejectReason.QUEUE
        return True, None

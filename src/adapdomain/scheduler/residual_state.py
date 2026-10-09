"""Thread-safe per-device residual cluster state."""
from __future__ import annotations

import threading
from typing import Iterable

from adapdomain.core.shape import DeviceResidual, TenantShape, update_residual


class ClusterState:
    def __init__(self) -> None:
        self._residuals: dict[int, DeviceResidual] = {}
        self._locks: dict[int, threading.Lock] = {}
        self._placement_log: list[tuple[str, int, TenantShape]] = []

    @classmethod
    def from_residuals(cls, residuals: Iterable[DeviceResidual]) -> "ClusterState":
        state = cls()
        for r in residuals:
            state._residuals[r.device_id] = r
            state._locks[r.device_id] = threading.Lock()
        return state

    def device_ids(self) -> list[int]:
        return list(self._residuals.keys())

    def residual_of(self, device_id: int) -> DeviceResidual:
        return self._residuals[device_id]

    def apply_placement(self, *, replica: TenantShape, device_id: int) -> None:
        with self._locks[device_id]:
            self._residuals[device_id] = update_residual(self._residuals[device_id], [replica])
            self._placement_log.append((replica.replica_id, device_id, replica))

    def revert_placement(self, *, replica_id: str, device_id: int) -> None:
        with self._locks[device_id]:
            log = [(r, d, s) for r, d, s in self._placement_log if not (r == replica_id and d == device_id)]
            self._placement_log = log
            # rebuild residual from scratch
            base = DeviceResidual.empty(device_id, list(self._residuals[device_id].residual.keys()))
            shapes = [s for r, d, s in log if d == device_id]
            self._residuals[device_id] = update_residual(base, shapes)

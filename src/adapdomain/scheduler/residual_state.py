"""Thread-safe per-device residual cluster state."""
from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping

from adapdomain.core.shape import DeviceResidual, TenantShape, update_residual


class ClusterState:
    def __init__(self) -> None:
        self._residuals: dict[int, DeviceResidual] = {}
        self._locks: dict[int, threading.Lock] = {}
        self._placement_log: list[tuple[str, int, TenantShape]] = []
        self._resident_profile_keys: dict[str, str] = {}

    @classmethod
    def from_residuals(
        cls, residuals: Iterable[DeviceResidual], *, resident_profile_keys: Mapping[str, str] | None = None,
    ) -> ClusterState:
        state = cls()
        state._resident_profile_keys.update(resident_profile_keys or {})
        for r in residuals:
            state._residuals[r.device_id] = r
            state._locks[r.device_id] = threading.Lock()
        return state

    def device_ids(self) -> list[int]:
        return list(self._residuals.keys())

    def residual_of(self, device_id: int) -> DeviceResidual:
        return self._residuals[device_id]

    def resident_profile_keys_of(self, device_id: int) -> list[str]:
        return [self._resident_profile_keys.get(replica_id, replica_id)
                for replica_id in self._residuals[device_id].placed_replicas]

    def apply_placement(self, *, replica: TenantShape, device_id: int) -> None:
        with self._locks[device_id]:
            self._residuals[device_id] = update_residual(self._residuals[device_id], [replica])
            self._placement_log.append((replica.replica_id, device_id, replica))
            self._resident_profile_keys[replica.replica_id] = replica.profile_key or replica.model_name

    def revert_placement(self, *, replica_id: str, device_id: int) -> None:
        with self._locks[device_id]:
            log = [(r, d, s) for r, d, s in self._placement_log if not (r == replica_id and d == device_id)]
            self._placement_log = log
            self._resident_profile_keys.pop(replica_id, None)
            # rebuild residual from scratch
            base = DeviceResidual.empty(device_id, list(self._residuals[device_id].residual.keys()))
            shapes = [s for r, d, s in log if d == device_id]
            self._residuals[device_id] = update_residual(base, shapes)

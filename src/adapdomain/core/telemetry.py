"""Per-device telemetry and EWMA windows."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque

from pydantic import BaseModel, ConfigDict, Field

from .types import DeviceId


class DeviceTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: DeviceId
    cube_util: float = Field(ge=0.0, le=1.0)
    vector_util: float = Field(ge=0.0, le=1.0)
    hbm_pressure: float = Field(ge=0.0, le=1.0)
    queue_depth: int = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)
    goodput_qps: float = Field(ge=0)
    timestamp: datetime


class TelemetryWindow:
    """EWMA + ring-buffer per metric."""

    def __init__(self, *, alpha: float = 0.2, window_size: int = 60) -> None:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._ewma: dict[str, float] = {}
        self._window_size = window_size
        self._history: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def observe(self, key: str, value: float) -> None:
        if key in self._ewma:
            self._ewma[key] = self._alpha * value + (1 - self._alpha) * self._ewma[key]
        else:
            self._ewma[key] = float(value)
        self._history[key].append(float(value))

    def ewma(self, key: str) -> float:
        return self._ewma.get(key, 0.0)

    def history(self, key: str) -> list[float]:
        return list(self._history.get(key, ()))

    def residual_drift(self, key: str, baseline: float) -> float:
        return self.ewma(key) - baseline

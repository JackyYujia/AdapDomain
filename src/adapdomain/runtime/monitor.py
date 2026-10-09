"""Telemetry monitor: combines per-device EWMAs with a recent ring buffer."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from adapdomain.backends.base import DeviceTelemetrySample

from .ewma import Ewma
from .ring_buffer import RingBuffer


class RuntimeMonitor:
    def __init__(self, *, alpha: float, history: int = 60) -> None:
        self._alpha = alpha
        self._history = history
        self._cube: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._vector: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._hbm: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._queue: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._samples: dict[int, RingBuffer] = defaultdict(lambda: RingBuffer(capacity=history))

    def observe_sample(self, sample: DeviceTelemetrySample) -> None:
        self._cube[sample.device_id].observe(sample.cube_util)
        self._vector[sample.device_id].observe(sample.vector_util)
        self._hbm[sample.device_id].observe(sample.hbm_pressure)
        self._queue[sample.device_id].observe(sample.queue_depth)
        self._samples[sample.device_id].push(sample)

    def cube_drift(self, *, device_id: int, baseline: float) -> float:
        return self._cube[device_id].value - baseline

    def vector_drift(self, *, device_id: int, baseline: float) -> float:
        return self._vector[device_id].value - baseline

    def hbm_value(self, device_id: int) -> float:
        return self._hbm[device_id].value

    def queue_value(self, device_id: int) -> float:
        return self._queue[device_id].value

    def history(self, device_id: int) -> list[DeviceTelemetrySample]:
        return list(self._samples[device_id].snapshot())

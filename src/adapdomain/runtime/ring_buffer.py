"""Fixed-capacity ring buffer (single producer)."""
from __future__ import annotations

from collections import deque
from typing import Iterable


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf: deque = deque(maxlen=capacity)

    def push(self, value) -> None:
        self._buf.append(value)

    def snapshot(self) -> Iterable:
        return list(self._buf)

"""Scalar exponentially weighted moving average."""
from __future__ import annotations


class Ewma:
    def __init__(self, *, alpha: float) -> None:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._value: float | None = None

    def observe(self, value: float) -> None:
        self._value = float(value) if self._value is None else self._alpha * value + (1 - self._alpha) * self._value

    @property
    def value(self) -> float:
        return 0.0 if self._value is None else self._value

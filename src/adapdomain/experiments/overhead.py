"""Controller overhead microbenchmark."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class OverheadResult:
    size: int
    mean_ms: float
    p95_ms: float
    max_ms: float


def measure(target: Callable[[], None], *, size: int, repeats: int = 5) -> OverheadResult:
    latencies: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        target()
        latencies.append((time.perf_counter() - start) * 1e3)
    latencies.sort()
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return OverheadResult(size=size, mean_ms=statistics.mean(latencies), p95_ms=p95, max_ms=latencies[-1])

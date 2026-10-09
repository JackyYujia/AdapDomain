"""SLO-aware metrics."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SeedResult:
    seed: int
    goodput: float
    violation: float


@dataclass(frozen=True)
class AggregatedSeedResult:
    n: int
    mean_goodput: float
    std_goodput: float
    mean_violation: float
    std_violation: float


def compute_goodput(records: Iterable[Mapping], *, slo_ms: float, window_seconds: float) -> float:
    served_within_slo = sum(
        1 for r in records if r.get("served") and r["latency_ms"] <= slo_ms
    )
    return served_within_slo / max(window_seconds, 1e-9)


def slo_violation_rate(records: Iterable[Mapping], *, slo_ms: float) -> float:
    served = [r for r in records if r.get("served")]
    if not served:
        return 0.0
    violations = sum(1 for r in served if r["latency_ms"] > slo_ms)
    return violations / len(served)


def latency_percentiles(values: Sequence[float], percentiles: Sequence[int]) -> dict[int, float]:
    if not values:
        return {p: 0.0 for p in percentiles}
    arr = np.asarray(values, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def aggregate_seeds(results: Sequence[SeedResult]) -> AggregatedSeedResult:
    if not results:
        return AggregatedSeedResult(n=0, mean_goodput=0.0, std_goodput=0.0, mean_violation=0.0, std_violation=0.0)
    g = [r.goodput for r in results]
    v = [r.violation for r in results]
    return AggregatedSeedResult(
        n=len(results),
        mean_goodput=statistics.mean(g),
        std_goodput=statistics.pstdev(g) if len(g) > 1 else 0.0,
        mean_violation=statistics.mean(v),
        std_violation=statistics.pstdev(v) if len(v) > 1 else 0.0,
    )

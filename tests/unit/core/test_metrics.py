import math
import statistics

import pytest

from adapdomain.core.metrics import (
    SeedResult,
    aggregate_seeds,
    compute_goodput,
    latency_percentiles,
    slo_violation_rate,
)


def _latencies(values: list[float]) -> list[float]:
    return list(values)


def test_compute_goodput_counts_under_slo():
    served = [{"latency_ms": v, "served": True} for v in [5, 5, 5, 50]]
    gp = compute_goodput(served, slo_ms=10.0, window_seconds=1.0)
    assert gp == 3.0


def test_slo_violation_rate():
    served = [{"latency_ms": v, "served": True} for v in [5, 50, 50, 50]]
    rate = slo_violation_rate(served, slo_ms=10.0)
    assert rate == pytest.approx(0.75)


def test_latency_percentiles():
    pcts = latency_percentiles(_latencies([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [50, 95, 99])
    assert pcts[50] == pytest.approx(5.5, abs=0.5)
    assert pcts[95] >= pcts[50]
    assert pcts[99] >= pcts[95]


def test_aggregate_seeds():
    results = [
        SeedResult(seed=1, goodput=100.0, violation=0.05),
        SeedResult(seed=2, goodput=120.0, violation=0.04),
        SeedResult(seed=3, goodput=110.0, violation=0.06),
    ]
    agg = aggregate_seeds(results)
    assert agg.mean_goodput == pytest.approx(statistics.mean([100, 120, 110]))
    assert agg.std_goodput >= 0

import pytest

from adapdomain.core.interference import (
    InterferenceEntry,
    InterferenceTable,
    aggregate_interference,
)


def _entries() -> list[InterferenceEntry]:
    return [
        InterferenceEntry(model_a="A", model_b="B", direction="a_on_b",
                          pressure_bucket="mid", slowdown=0.30,
                          p95_delta=0.0, goodput_delta=0.0, source_run_id="r"),
        InterferenceEntry(model_a="A", model_b="C", direction="a_on_b",
                          pressure_bucket="mid", slowdown=0.20,
                          p95_delta=0.0, goodput_delta=0.0, source_run_id="r"),
        InterferenceEntry(model_a="A", model_b="D", direction="a_on_b",
                          pressure_bucket="mid", slowdown=0.10,
                          p95_delta=0.0, goodput_delta=0.0, source_run_id="r"),
    ]


def test_aggregate_max_plus_damped_sum_matches_paper():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["B", "C", "D"], bucket="mid", eta=0.30)
    expected = 0.30 + 0.30 * (0.30 + 0.20 + 0.10)
    assert x == pytest.approx(expected, rel=1e-9)


def test_aggregate_eta_zero_reduces_to_max():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["B", "C", "D"], bucket="mid", eta=0.0)
    assert x == pytest.approx(0.30)


def test_aggregate_eta_one_equals_sum_plus_max():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["B", "C", "D"], bucket="mid", eta=1.0)
    assert x == pytest.approx(0.30 + (0.30 + 0.20 + 0.10))


def test_missing_pair_returns_zero():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["Z"], bucket="mid", eta=0.30)
    assert x == 0.0

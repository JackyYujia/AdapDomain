"""Pairwise interference collection harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from adapdomain.core.interference import InterferenceEntry, InterferenceTable


@dataclass(frozen=True)
class PairSpec:
    model_a: str
    model_b: str
    pressure_bucket: str = "mid"


def collect(
    *,
    pairs: Iterable[PairSpec],
    measure_slowdown: Callable[[str, str, str], float],
) -> InterferenceTable:
    entries: list[InterferenceEntry] = []
    for pair in pairs:
        s = measure_slowdown(pair.model_a, pair.model_b, pair.pressure_bucket)
        entries.append(InterferenceEntry(
            model_a=pair.model_a, model_b=pair.model_b, direction="a_on_b",
            pressure_bucket=pair.pressure_bucket,
            slowdown=s, p95_delta=0.0, goodput_delta=0.0, source_run_id="collect",
        ))
    return InterferenceTable.from_entries(entries)

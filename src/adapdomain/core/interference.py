"""Pairwise interference model (paper Eq. interference_aggregation)."""
from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .types import RunId


class InterferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_a: str
    model_b: str
    direction: str  # "a_on_b" or "b_on_a"
    pressure_bucket: str
    slowdown: float = Field(ge=0)
    p95_delta: float
    goodput_delta: float
    source_run_id: RunId | None = None


class InterferenceTable:
    """Indexed lookup over (candidate, colocated, bucket)."""

    def __init__(self, entries: Iterable[InterferenceEntry]) -> None:
        self._index: dict[tuple[str, str, str], float] = {}
        for e in entries:
            key = (e.model_a, e.model_b, e.pressure_bucket) if e.direction == "a_on_b" else (e.model_b, e.model_a, e.pressure_bucket)
            self._index[key] = e.slowdown

    @classmethod
    def from_entries(cls, entries: Iterable[InterferenceEntry]) -> "InterferenceTable":
        return cls(entries)

    def lookup(self, candidate: str, other: str, bucket: str) -> float:
        return self._index.get((candidate, other, bucket), 0.0)


def aggregate_interference(
    table: InterferenceTable,
    *,
    candidate: str,
    colocated: Sequence[str],
    bucket: str,
    eta: float,
) -> float:
    """X(i, k) = max_j X(i, j) + eta * sum_j X(i, j)."""
    pair_slowdowns = [table.lookup(candidate, other, bucket) for other in colocated]
    pair_slowdowns = [s for s in pair_slowdowns if s > 0]
    if not pair_slowdowns:
        return 0.0
    return max(pair_slowdowns) + eta * sum(pair_slowdowns)

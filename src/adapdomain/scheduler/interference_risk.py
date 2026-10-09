"""Adapter around core.interference.aggregate_interference."""
from __future__ import annotations

from typing import Sequence

from adapdomain.core.interference import InterferenceTable, aggregate_interference


def interference_risk(
    *,
    candidate: str,
    colocated: Sequence[str],
    bucket: str,
    table: InterferenceTable,
    eta: float,
) -> float:
    raw = aggregate_interference(table, candidate=candidate, colocated=colocated, bucket=bucket, eta=eta)
    # squash to [0, 1] via saturating tanh-like mapping
    return min(1.0, raw)

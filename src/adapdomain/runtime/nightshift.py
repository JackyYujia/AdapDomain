"""NightShift: idle-period migration picker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MigrationCandidate:
    replica_id: str
    src_device: int
    dst_device: int
    delta_ceff: float
    risk: float


class NightShift:
    def __init__(self, *, migration_budget: int, risk_max: float = 0.5) -> None:
        self._budget = migration_budget
        self._risk_max = risk_max

    def select(self, candidates: Sequence[MigrationCandidate]) -> list[MigrationCandidate]:
        safe = [c for c in candidates if c.risk <= self._risk_max]
        safe.sort(key=lambda c: c.delta_ceff, reverse=True)
        return safe[: self._budget]

"""Feasible frontier data model, query, and confidence."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .types import FrontierVersion, ModelId, RunId


class FrontierPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    point_id: str
    raw_cube: float
    raw_vector: float
    raw_hbm: float
    norm_cube: float = Field(ge=0.0, le=1.0)
    norm_vector: float = Field(ge=0.0, le=1.0)
    batch_size: int = Field(gt=0)
    slo_ms: float = Field(gt=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)
    goodput_qps: float = Field(ge=0)
    feasible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    msprof_artifact_path: str | None = None


class FrontierSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    version: FrontierVersion
    k_domains: list[str]
    points: list[FrontierPoint]
    safe_point_id: str
    dominant_domain: str
    confidence_summary: float = Field(ge=0.0, le=1.0)
    source_run_id: RunId | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def feasible_points(self) -> list[FrontierPoint]:
        return [p for p in self.points if p.feasible]

    def safe_point(self) -> FrontierPoint:
        for p in self.points:
            if p.point_id == self.safe_point_id:
                return p
        raise KeyError(f"safe_point_id {self.safe_point_id!r} not found")


class FrontierQuery:
    """Pick the best feasible point for a target (batch_size, slo_ms)."""

    def __init__(self, snapshot: FrontierSnapshot) -> None:
        self._snapshot = snapshot

    def pick(self, *, batch_size: int, slo_ms: float) -> FrontierPoint | None:
        best: FrontierPoint | None = None
        for p in self._snapshot.points:
            if not p.feasible:
                continue
            if p.batch_size != batch_size:
                continue
            if p.p95_ms > slo_ms:
                continue
            if best is None or p.p95_ms < best.p95_ms:
                best = p
        return best


class ConfidenceModel:
    """Combine neighbor density, recent deviations, and age into [0,1]."""

    def __init__(
        self,
        *,
        half_life_days: float = 60.0,
        deviation_scale: float = 0.3,
        neighbor_saturation: int = 8,
    ) -> None:
        self._half_life = timedelta(days=half_life_days)
        self._deviation_scale = deviation_scale
        self._neighbor_saturation = neighbor_saturation

    def score(
        self,
        *,
        neighbor_count: int,
        recent_deviations: Sequence[float],
        created_at: datetime,
        now: datetime | None = None,
    ) -> float:
        now = now or datetime.now(timezone.utc)
        age = max(now - created_at, timedelta(0))
        age_factor = 0.5 ** (age / self._half_life)
        density = min(1.0, neighbor_count / self._neighbor_saturation)
        mean_dev = (sum(abs(d) for d in recent_deviations) / max(1, len(recent_deviations)))
        dev_factor = math.exp(-mean_dev / max(self._deviation_scale, 1e-6))
        return float(max(0.0, min(1.0, density * dev_factor * age_factor)))


__all__ = [
    "ConfidenceModel",
    "FrontierPoint",
    "FrontierQuery",
    "FrontierSnapshot",
]

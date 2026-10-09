"""Build FrontierSnapshot from MsprofObservation lists."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot

from .confidence import summarize_confidence
from .normalization import CounterRange, normalize


@dataclass(frozen=True)
class MsprofObservation:
    model: str
    point_id: str
    raw_cube: float
    raw_vector: float
    raw_hbm: float
    batch_size: int
    slo_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    goodput_qps: float
    feasible: bool


class FrontierBuilder:
    def __init__(self, *, cube_range: tuple[float, float], vector_range: tuple[float, float]) -> None:
        self._cube = CounterRange(*cube_range)
        self._vector = CounterRange(*vector_range)

    def build(
        self,
        *,
        model_name: str,
        version: int,
        observations: Iterable[MsprofObservation],
        source_run_id: str,
    ) -> FrontierSnapshot:
        obs = list(observations)
        if not obs:
            raise ValueError("no observations")
        points = [
            FrontierPoint(
                model_name=model_name,
                point_id=o.point_id,
                raw_cube=o.raw_cube, raw_vector=o.raw_vector, raw_hbm=o.raw_hbm,
                norm_cube=normalize(o.raw_cube, self._cube),
                norm_vector=normalize(o.raw_vector, self._vector),
                batch_size=o.batch_size, slo_ms=o.slo_ms,
                p50_ms=o.p50_ms, p95_ms=o.p95_ms, p99_ms=o.p99_ms,
                goodput_qps=o.goodput_qps,
                feasible=o.feasible,
                confidence=summarize_confidence(created_at=datetime.now(timezone.utc), point_count=len(obs)),
            )
            for o in obs
        ]
        feasible_points = [p for p in points if p.feasible]
        safe = min(feasible_points, key=lambda p: p.p95_ms) if feasible_points else points[0]
        mean_cube = sum(p.norm_cube for p in points) / len(points)
        mean_vec = sum(p.norm_vector for p in points) / len(points)
        if mean_cube > mean_vec * 1.5:
            dominant = "matrix"
        elif mean_vec > mean_cube * 1.5:
            dominant = "vector"
        else:
            dominant = "balanced"
        return FrontierSnapshot(
            model_name=model_name,
            version=version,
            k_domains=["cube", "vector"],
            points=points,
            safe_point_id=safe.point_id,
            dominant_domain=dominant,
            confidence_summary=float(sum(p.confidence for p in points) / len(points)),
            source_run_id=source_run_id,
        )

"""Wrapper around core.frontier.ConfidenceModel keyed to msprof inputs."""
from __future__ import annotations

from datetime import datetime

from adapdomain.core.frontier import ConfidenceModel


def summarize_confidence(*, created_at: datetime, point_count: int, deviations: list[float] | None = None) -> float:
    return ConfidenceModel().score(
        neighbor_count=point_count,
        recent_deviations=deviations or [0.0],
        created_at=created_at,
    )

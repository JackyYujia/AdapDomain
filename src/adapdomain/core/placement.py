"""Placement plan and score breakdown."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .types import DeviceId, FrontierVersion, ModelId, ReplicaId


@dataclass(frozen=True)
class ReplicaPlacement:
    replica_id: ReplicaId
    device_id: DeviceId
    frontier_point_id: str


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ceff: float
    x: float
    h: float
    q: float
    f: float
    u: float
    final: float


class PlacementPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    created_at: datetime
    input_replicas: list[ReplicaId]
    placements: dict[ReplicaId, DeviceId]
    rejected: dict[ReplicaId, str]
    score_breakdown: Mapping[tuple[ReplicaId, DeviceId], ScoreBreakdown]
    frontier_versions: dict[ModelId, FrontierVersion]

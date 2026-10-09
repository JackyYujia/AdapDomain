"""Workload model used by the replay engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import ReplicaId


class Replica(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    replica_id: ReplicaId
    model: str
    slo_ms: float = Field(gt=0)
    batch_size: int = Field(gt=0)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    model: str
    arrival_ms: float = Field(ge=0)
    deadline_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def _deadline_after_arrival(self) -> "Request":
        if self.deadline_ms <= self.arrival_ms:
            raise ValueError("deadline_ms must be > arrival_ms")
        return self


class WorkloadTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replicas: list[Replica]
    requests: list[Request]

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_type": "header", "replicas": [r.model_dump() for r in self.replicas]}) + "\n")
            for req in self.requests:
                f.write(json.dumps({"_type": "request", **req.model_dump()}) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "WorkloadTrace":
        p = Path(path)
        replicas: list[Replica] = []
        requests: list[Request] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                kind = rec.pop("_type")
                if kind == "header":
                    replicas = [Replica.model_validate(r) for r in rec.get("replicas", [])]
                elif kind == "request":
                    requests.append(Request.model_validate(rec))
        return cls(replicas=replicas, requests=requests)

    def slice(self, *, start_ms: float, end_ms: float) -> "WorkloadTrace":
        keep = [r for r in self.requests if start_ms < r.arrival_ms <= end_ms]
        return WorkloadTrace(replicas=list(self.replicas), requests=keep)

    @classmethod
    def from_requests(cls, replicas: Iterable[Replica], requests: Iterable[Request]) -> "WorkloadTrace":
        return cls(replicas=list(replicas), requests=list(requests))

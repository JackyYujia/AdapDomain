"""Audit logging for runtime repair actions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapdomain.backends.base import ResourceQuota
from adapdomain.storage.jsonl import JsonlWriter


class RepairLogger:
    def __init__(self, path: str | Path) -> None:
        self._writer = JsonlWriter(path)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        self._writer.write({"type": event_type, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload})


def perform_quota_repair(*, logger: RepairLogger, device_id: int, replica_id: str, new_quota: ResourceQuota, reason: str) -> None:
    logger.log("quota_repair", {
        "device_id": device_id,
        "replica_id": replica_id,
        "cube_units": new_quota.cube_units,
        "vector_units": new_quota.vector_units,
        "reason": reason,
    })


def perform_migration(*, logger: RepairLogger, replica_id: str, src_device: int, dst_device: int, reason: str) -> None:
    logger.log("migrate", {
        "replica_id": replica_id, "src": src_device, "dst": dst_device, "reason": reason,
    })


def perform_one_card_rebalance(*, logger: RepairLogger, device_id: int, plan: dict[str, int], reason: str) -> None:
    logger.log("one_card_rebalance", {"device_id": device_id, "plan": plan, "reason": reason})

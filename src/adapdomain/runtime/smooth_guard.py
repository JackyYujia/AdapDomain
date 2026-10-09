"""SmoothGuard: persistent-drift detector with cooldown."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GuardAction(str, Enum):
    NONE = "none"
    QUOTA = "quota_repair"
    MIGRATE = "local_migration"
    REBALANCE = "one_card_rebalance"
    REPROFILE = "trigger_reprofile"


@dataclass(frozen=True)
class GuardThresholds:
    p95_drift: float
    residual_drift: float
    hbm_high: float
    queue_high: float


@dataclass
class GuardEvaluation:
    action: GuardAction
    reasons: list[str] = field(default_factory=list)


class SmoothGuard:
    def __init__(self, *, thresholds: GuardThresholds, consecutive: int, cooldown_s: float) -> None:
        self._thresholds = thresholds
        self._consecutive = consecutive
        self._cooldown_s = cooldown_s
        self._counters: dict[int, int] = {}
        self._last_fire_at: dict[int, float] = {}

    def evaluate(
        self,
        *,
        device_id: int,
        p95_drift: float,
        residual_drift: float,
        hbm: float,
        queue: float,
        now: float,
    ) -> GuardEvaluation:
        reasons: list[str] = []
        if abs(p95_drift) > self._thresholds.p95_drift:
            reasons.append("p95_drift")
        if abs(residual_drift) > self._thresholds.residual_drift:
            reasons.append("residual_drift")
        if hbm > self._thresholds.hbm_high:
            reasons.append("hbm_high")
        if queue > self._thresholds.queue_high:
            reasons.append("queue_high")

        if not reasons:
            self._counters[device_id] = 0
            return GuardEvaluation(action=GuardAction.NONE)

        last = self._last_fire_at.get(device_id, float("-inf"))
        if now - last < self._cooldown_s:
            return GuardEvaluation(action=GuardAction.NONE, reasons=reasons + ["cooldown"])

        self._counters[device_id] = self._counters.get(device_id, 0) + 1
        if self._counters[device_id] < self._consecutive:
            return GuardEvaluation(action=GuardAction.NONE, reasons=reasons)

        self._counters[device_id] = 0
        self._last_fire_at[device_id] = now
        # Pick action by severity
        if "hbm_high" in reasons or "queue_high" in reasons:
            return GuardEvaluation(action=GuardAction.MIGRATE, reasons=reasons)
        return GuardEvaluation(action=GuardAction.QUOTA, reasons=reasons)

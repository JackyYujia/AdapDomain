import pytest

from adapdomain.runtime.smooth_guard import GuardAction, GuardThresholds, SmoothGuard


def test_guard_triggers_only_after_two_crossings():
    guard = SmoothGuard(thresholds=GuardThresholds(p95_drift=0.2, residual_drift=0.2, hbm_high=0.85, queue_high=0.8),
                        consecutive=2, cooldown_s=30)
    sample = lambda: guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=0.0)
    assert sample().action is GuardAction.NONE
    next_action = guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=1.0)
    assert next_action.action is GuardAction.QUOTA


def test_guard_respects_cooldown():
    guard = SmoothGuard(thresholds=GuardThresholds(p95_drift=0.2, residual_drift=0.2, hbm_high=0.85, queue_high=0.8),
                        consecutive=2, cooldown_s=30)
    guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=0.0)
    second = guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=1.0)
    third = guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=2.0)
    assert second.action is GuardAction.QUOTA
    assert third.action is GuardAction.NONE

import pytest

from adapdomain.core.errors import (
    AdapDomainError,
    AclDeviceError,
    AclExecuteError,
    AclInitError,
    AclMemoryError,
    AclModelError,
    BackendError,
    ConfigError,
    EmptyCandidateError,
    InfeasiblePlacementError,
    MsprofParseError,
    MsprofRunError,
    ProfilerError,
    RepairBudgetExceededError,
    RuntimeAdapterError,
    SchedulerError,
)
from adapdomain.core.types import DEFAULT_K, EPSILON_DOMAIN_FIT


def test_error_hierarchy():
    assert issubclass(ConfigError, AdapDomainError)
    assert issubclass(BackendError, AdapDomainError)
    for cls in [AclInitError, AclDeviceError, AclModelError, AclMemoryError, AclExecuteError]:
        assert issubclass(cls, BackendError)
    assert issubclass(ProfilerError, AdapDomainError)
    for cls in [MsprofRunError, MsprofParseError]:
        assert issubclass(cls, ProfilerError)
    assert issubclass(SchedulerError, AdapDomainError)
    for cls in [InfeasiblePlacementError, EmptyCandidateError]:
        assert issubclass(cls, SchedulerError)
    assert issubclass(RuntimeAdapterError, AdapDomainError)
    assert issubclass(RepairBudgetExceededError, RuntimeAdapterError)


def test_constants():
    assert EPSILON_DOMAIN_FIT == pytest.approx(1e-6)
    assert DEFAULT_K == 2


def test_acl_error_carries_ret_code():
    err = AclInitError("init failed", ret=507000)
    assert err.ret == 507000
    assert "507000" in str(err)

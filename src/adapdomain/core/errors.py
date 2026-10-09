"""Error hierarchy for AdapDomain."""
from __future__ import annotations


class AdapDomainError(Exception):
    """Base class for all AdapDomain errors."""


class ConfigError(AdapDomainError):
    """Invalid or missing configuration."""


class BackendError(AdapDomainError):
    """Errors raised by an inference backend (e.g. ACL)."""

    def __init__(self, message: str, *, ret: int | None = None, operation: str | None = None) -> None:
        if ret is not None:
            message = f"{message} (ret={ret}, op={operation})"
        super().__init__(message)
        self.ret = ret
        self.operation = operation


class AclInitError(BackendError): ...
class AclDeviceError(BackendError): ...
class AclModelError(BackendError): ...
class AclMemoryError(BackendError): ...
class AclExecuteError(BackendError): ...


class ProfilerError(AdapDomainError):
    """Errors raised by the profiling layer."""


class MsprofRunError(ProfilerError): ...
class MsprofParseError(ProfilerError): ...


class SchedulerError(AdapDomainError):
    """Errors raised by the scheduler."""


class InfeasiblePlacementError(SchedulerError): ...
class EmptyCandidateError(SchedulerError): ...


class RuntimeAdapterError(AdapDomainError):
    """Errors raised by runtime guards (SmoothGuard / PreScale / NightShift)."""


class RepairBudgetExceededError(RuntimeAdapterError): ...

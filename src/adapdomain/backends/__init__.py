"""Backend abstraction package."""
from .base import (
    BackendRegistry,
    DeviceInfo,
    DeviceTelemetrySample,
    ExecutionResult,
    InferenceBackend,
    MockBackend,
    ModelHandle,
    ResourceQuota,
)

__all__ = [
    "BackendRegistry",
    "DeviceInfo",
    "DeviceTelemetrySample",
    "ExecutionResult",
    "InferenceBackend",
    "MockBackend",
    "ModelHandle",
    "ResourceQuota",
]

"""Inference backend abstraction. Only `backends/ascend/` may import `acl`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from adapdomain.core.errors import BackendError


@dataclass(frozen=True)
class DeviceInfo:
    device_id: int
    name: str
    cube_capacity: int
    vector_capacity: int


@dataclass(frozen=True)
class ResourceQuota:
    device_id: int
    cube_units: int | None = None
    vector_units: int | None = None
    hbm_bytes: int | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelHandle:
    backend_name: str
    model_id: str
    device_id: int
    internal: Any


@dataclass(frozen=True)
class ExecutionResult:
    latencies_ms: "np.ndarray"
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceTelemetrySample:
    device_id: int
    cube_util: float
    vector_util: float
    hbm_pressure: float
    queue_depth: int


@runtime_checkable
class InferenceBackend(Protocol):
    name: str

    def initialize(self) -> None: ...
    def finalize(self) -> None: ...
    def list_devices(self) -> list[DeviceInfo]: ...
    def load_model(self, *, model_id: str, path: str, device_id: int) -> ModelHandle: ...
    def unload_model(self, handle: ModelHandle) -> None: ...
    def execute(self, handle: ModelHandle, *, inputs: Mapping[str, Any], repeat: int = 1) -> ExecutionResult: ...
    def apply_resource_quota(self, *, device_id: int, quota: ResourceQuota) -> None: ...
    def collect_telemetry(self, *, device_id: int) -> DeviceTelemetrySample: ...


class MockBackend:
    name = "mock"

    def __init__(self, devices: list[DeviceInfo]) -> None:
        self._devices = devices
        self._initialized = False
        self._loaded: dict[str, ModelHandle] = {}

    def initialize(self) -> None:
        self._initialized = True

    def finalize(self) -> None:
        self._initialized = False
        self._loaded.clear()

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def load_model(self, *, model_id: str, path: str, device_id: int) -> ModelHandle:
        if not self._initialized:
            raise BackendError("MockBackend not initialized")
        handle = ModelHandle(backend_name=self.name, model_id=model_id, device_id=device_id, internal={"path": path})
        self._loaded[model_id] = handle
        return handle

    def unload_model(self, handle: ModelHandle) -> None:
        self._loaded.pop(handle.model_id, None)

    def execute(self, handle: ModelHandle, *, inputs, repeat: int = 1) -> ExecutionResult:
        rng = np.random.default_rng(seed=hash(handle.model_id) & 0xFFFFFFFF)
        return ExecutionResult(latencies_ms=rng.uniform(2.0, 8.0, size=repeat))

    def apply_resource_quota(self, *, device_id: int, quota: ResourceQuota) -> None:
        pass

    def collect_telemetry(self, *, device_id: int) -> DeviceTelemetrySample:
        return DeviceTelemetrySample(device_id=device_id, cube_util=0.4, vector_util=0.3, hbm_pressure=0.3, queue_depth=1)


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, callable] = {}

    def register(self, name: str, factory) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> InferenceBackend:
        if name not in self._factories:
            raise BackendError(f"unknown backend {name!r}")
        return self._factories[name](**kwargs)

    @classmethod
    def default(cls) -> "BackendRegistry":
        reg = cls()
        reg.register("mock", lambda: MockBackend(devices=[
            DeviceInfo(device_id=i, name=f"mock-{i}", cube_capacity=20, vector_capacity=40) for i in range(4)
        ]))
        return reg

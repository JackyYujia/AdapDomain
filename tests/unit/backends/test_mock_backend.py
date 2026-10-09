import numpy as np

from adapdomain.backends.base import (
    BackendRegistry,
    DeviceInfo,
    ExecutionResult,
    InferenceBackend,
    MockBackend,
    ResourceQuota,
)


def test_mock_backend_basic_lifecycle():
    backend: InferenceBackend = MockBackend(
        devices=[DeviceInfo(device_id=0, name="mock-0", cube_capacity=20, vector_capacity=40)]
    )
    backend.initialize()
    handle = backend.load_model(model_id="bert-base", path="/tmp/fake.om", device_id=0)
    res: ExecutionResult = backend.execute(handle, inputs={"x": np.zeros(1)}, repeat=3)
    assert res.latencies_ms.shape == (3,)
    backend.apply_resource_quota(device_id=0, quota=ResourceQuota(device_id=0, cube_units=10, vector_units=20))
    backend.unload_model(handle)
    backend.finalize()


def test_registry_returns_named_backend():
    reg = BackendRegistry.default()
    assert isinstance(reg.create("mock"), InferenceBackend)

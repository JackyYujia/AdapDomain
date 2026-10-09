import pytest

from adapdomain.backends.ascend.resource_limit import (
    ACL_RT_DEV_RES_CUBE_CORE,
    ACL_RT_DEV_RES_VECTOR_CORE,
    AscendResourceLimiter,
    ResourceLimitConfig,
)
from adapdomain.backends.base import ResourceQuota


def test_huawei_runtime_limit_type_constants():
    assert ACL_RT_DEV_RES_CUBE_CORE == 0
    assert ACL_RT_DEV_RES_VECTOR_CORE == 1


def test_no_op_when_types_unset(caplog):
    cfg = ResourceLimitConfig(cube_limit_type=None, vector_limit_type=None, hbm_limit_type=None)
    limiter = AscendResourceLimiter(cfg)
    limiter.apply(ResourceQuota(device_id=0, cube_units=10, vector_units=20))
    assert any("no-op" in r.message for r in caplog.records)


def test_rejects_negative_units():
    cfg = ResourceLimitConfig(cube_limit_type=1, vector_limit_type=2)
    limiter = AscendResourceLimiter(cfg)
    with pytest.raises(ValueError):
        limiter.apply(ResourceQuota(device_id=0, cube_units=-1, vector_units=20))


def test_apply_calls_acl_with_configured_cube_vector_types(monkeypatch):
    calls = []

    class FakeRt:
        @staticmethod
        def set_device_res_limit(device_id, limit_type, value):
            calls.append((device_id, limit_type, value))
            return 0

    class FakeAcl:
        rt = FakeRt()

    monkeypatch.setattr("adapdomain.backends.ascend.resource_limit._import_acl", lambda: FakeAcl())

    cfg = ResourceLimitConfig(
        cube_limit_type=ACL_RT_DEV_RES_CUBE_CORE,
        vector_limit_type=ACL_RT_DEV_RES_VECTOR_CORE,
    )
    limiter = AscendResourceLimiter(cfg)
    limiter.apply(ResourceQuota(device_id=2, cube_units=12, vector_units=24))

    assert calls == [(2, 0, 12), (2, 1, 24)]

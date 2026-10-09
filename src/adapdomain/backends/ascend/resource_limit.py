"""Wrapper around acl.rt.set_device_res_limit.

Huawei CANN exposes Cube and Vector resource-limit domains as
ACL_RT_DEV_RES_CUBE_CORE = 0 and ACL_RT_DEV_RES_VECTOR_CORE = 1. The values are
still loaded from hardware YAML so a different CANN target can override them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from adapdomain.backends.base import ResourceQuota

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclDeviceError

_log = logging.getLogger(__name__)

ACL_RT_DEV_RES_CUBE_CORE = 0
ACL_RT_DEV_RES_VECTOR_CORE = 1


@dataclass(frozen=True)
class ResourceLimitConfig:
    cube_limit_type: int | None
    vector_limit_type: int | None
    hbm_limit_type: int | None = None


class AscendResourceLimiter:
    """`set_device_res_limit` must be called after set_device and before any operator execution."""

    def __init__(self, config: ResourceLimitConfig) -> None:
        self._config = config

    def apply(self, quota: ResourceQuota) -> None:
        if quota.cube_units is not None and quota.cube_units < 0:
            raise ValueError("cube_units must be >= 0")
        if quota.vector_units is not None and quota.vector_units < 0:
            raise ValueError("vector_units must be >= 0")

        if (
            self._config.cube_limit_type is None
            and self._config.vector_limit_type is None
            and self._config.hbm_limit_type is None
        ):
            _log.warning(
                "AscendResourceLimiter no-op: set_device_res_limit type enums not configured for device %d",
                quota.device_id,
            )
            return

        acl = _import_acl()
        if self._config.cube_limit_type is not None and quota.cube_units is not None:
            ret = acl.rt.set_device_res_limit(quota.device_id, self._config.cube_limit_type, quota.cube_units)
            if ret != 0:
                raise AclDeviceError("set_device_res_limit cube failed", ret=ret, operation="set_device_res_limit(cube)")
        if self._config.vector_limit_type is not None and quota.vector_units is not None:
            ret = acl.rt.set_device_res_limit(quota.device_id, self._config.vector_limit_type, quota.vector_units)
            if ret != 0:
                raise AclDeviceError("set_device_res_limit vector failed", ret=ret, operation="set_device_res_limit(vector)")
        if self._config.hbm_limit_type is not None and quota.hbm_bytes is not None:
            ret = acl.rt.set_device_res_limit(quota.device_id, self._config.hbm_limit_type, quota.hbm_bytes)
            if ret != 0:
                raise AclDeviceError("set_device_res_limit hbm failed", ret=ret, operation="set_device_res_limit(hbm)")

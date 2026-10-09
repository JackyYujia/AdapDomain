"""Ascend device lifecycle helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from adapdomain.utils.logging import get_logger

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclDeviceError

_log = get_logger(__name__)


class AscendDevice:
    def __init__(self, device_id: int) -> None:
        self._device_id = device_id
        self._set = False

    @property
    def device_id(self) -> int:
        return self._device_id

    def set(self) -> None:
        acl = _import_acl()
        ret = acl.rt.set_device(self._device_id)
        AclRuntime.check_ret(ret, operation=f"acl.rt.set_device({self._device_id})")
        self._set = True

    def reset(self) -> None:
        if not self._set:
            return
        acl = _import_acl()
        ret = acl.rt.reset_device(self._device_id)
        AclRuntime.check_ret(ret, operation=f"acl.rt.reset_device({self._device_id})")
        self._set = False

    @staticmethod
    def get_count() -> int:
        acl = _import_acl()
        count, ret = acl.rt.get_device_count()
        if ret != 0:
            raise AclDeviceError("acl.rt.get_device_count failed", ret=ret, operation="get_device_count")
        return int(count)

    @contextmanager
    def bind(self) -> Iterator["AscendDevice"]:
        self.set()
        try:
            yield self
        finally:
            self.reset()

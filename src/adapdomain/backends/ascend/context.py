"""Ascend execution context (acl.rt.create_context / destroy_context)."""
from __future__ import annotations

from typing import Any

from .acl_runtime import AclRuntime, _import_acl


class AscendContext:
    def __init__(self, device_id: int) -> None:
        self._device_id = device_id
        self._handle: Any = None

    def create(self) -> None:
        acl = _import_acl()
        handle, ret = acl.rt.create_context(self._device_id)
        AclRuntime.check_ret(ret, operation="acl.rt.create_context")
        self._handle = handle

    def destroy(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        ret = acl.rt.destroy_context(self._handle)
        AclRuntime.check_ret(ret, operation="acl.rt.destroy_context")
        self._handle = None

    def __enter__(self) -> "AscendContext":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

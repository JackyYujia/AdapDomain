"""Ascend stream (acl.rt.create_stream / synchronize_stream / destroy_stream)."""
from __future__ import annotations

from typing import Any

from .acl_runtime import AclRuntime, _import_acl


class AscendStream:
    def __init__(self) -> None:
        self._handle: Any = None

    def create(self) -> None:
        acl = _import_acl()
        handle, ret = acl.rt.create_stream()
        AclRuntime.check_ret(ret, operation="acl.rt.create_stream")
        self._handle = handle

    @property
    def handle(self) -> Any:
        if self._handle is None:
            raise RuntimeError("AscendStream not created")
        return self._handle

    def synchronize(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        ret = acl.rt.synchronize_stream(self._handle)
        AclRuntime.check_ret(ret, operation="acl.rt.synchronize_stream")

    def destroy(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        ret = acl.rt.destroy_stream(self._handle)
        AclRuntime.check_ret(ret, operation="acl.rt.destroy_stream")
        self._handle = None

    def __enter__(self) -> "AscendStream":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

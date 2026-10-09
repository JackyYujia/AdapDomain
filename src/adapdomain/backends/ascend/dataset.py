"""Ascend aclmdlDataset wrapper."""
from __future__ import annotations

from typing import Any, Sequence

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclModelError


class AscendDataset:
    def __init__(self) -> None:
        self._handle: Any = None
        self._buffers: list[Any] = []

    def create(self) -> None:
        acl = _import_acl()
        self._handle = acl.mdl.create_dataset()
        if self._handle is None:
            raise AclModelError("create_dataset returned None", operation="mdl.create_dataset")

    @property
    def handle(self) -> Any:
        if self._handle is None:
            raise RuntimeError("dataset not created")
        return self._handle

    def add_device_buffer(self, ptr: Any, size: int) -> None:
        acl = _import_acl()
        buf = acl.create_data_buffer(ptr, size)
        if buf is None:
            raise AclModelError("create_data_buffer returned None", operation="create_data_buffer")
        ret = acl.mdl.add_dataset_buffer(self._handle, buf)
        AclRuntime.check_ret(ret, operation="mdl.add_dataset_buffer")
        self._buffers.append(buf)

    def destroy(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        for buf in self._buffers:
            ret = acl.destroy_data_buffer(buf)
            AclRuntime.check_ret(ret, operation="destroy_data_buffer")
        self._buffers.clear()
        ret = acl.mdl.destroy_dataset(self._handle)
        AclRuntime.check_ret(ret, operation="mdl.destroy_dataset")
        self._handle = None

    def __enter__(self) -> "AscendDataset":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

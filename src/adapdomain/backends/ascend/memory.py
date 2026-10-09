"""Ascend memory helpers.

Wraps acl.rt.malloc / malloc_host / free / free_host / memcpy.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from adapdomain.utils.logging import get_logger

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclMemoryError

_log = get_logger(__name__)


@dataclass
class DeviceBuffer:
    ptr: Any
    size: int

    def free(self) -> None:
        acl = _import_acl()
        ret = acl.rt.free(self.ptr)
        AclRuntime.check_ret(ret, operation="acl.rt.free")


@dataclass
class HostBuffer:
    ptr: Any
    size: int

    def free(self) -> None:
        acl = _import_acl()
        ret = acl.rt.free_host(self.ptr)
        AclRuntime.check_ret(ret, operation="acl.rt.free_host")


def malloc_device(size: int) -> DeviceBuffer:
    acl = _import_acl()
    ptr, ret = acl.rt.malloc(size, acl.rt.ACL_MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        raise AclMemoryError(f"malloc({size}) failed", ret=ret, operation="acl.rt.malloc")
    return DeviceBuffer(ptr=ptr, size=size)


def malloc_host(size: int) -> HostBuffer:
    acl = _import_acl()
    ptr, ret = acl.rt.malloc_host(size)
    if ret != 0:
        raise AclMemoryError(f"malloc_host({size}) failed", ret=ret, operation="acl.rt.malloc_host")
    return HostBuffer(ptr=ptr, size=size)


def memcpy(dst: Any, dst_size: int, src: Any, src_size: int, kind: str) -> None:
    acl = _import_acl()
    kind_enum = {
        "host_to_device": acl.rt.ACL_MEMCPY_HOST_TO_DEVICE,
        "device_to_host": acl.rt.ACL_MEMCPY_DEVICE_TO_HOST,
        "device_to_device": acl.rt.ACL_MEMCPY_DEVICE_TO_DEVICE,
    }[kind]
    ret = acl.rt.memcpy(dst, dst_size, src, src_size, kind_enum)
    AclRuntime.check_ret(ret, operation=f"acl.rt.memcpy:{kind}")


@contextmanager
def device_buffer(size: int) -> Iterator[DeviceBuffer]:
    buf = malloc_device(size)
    try:
        yield buf
    finally:
        buf.free()


@contextmanager
def host_buffer(size: int) -> Iterator[HostBuffer]:
    buf = malloc_host(size)
    try:
        yield buf
    finally:
        buf.free()

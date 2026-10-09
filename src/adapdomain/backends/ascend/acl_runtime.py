"""ACL Python runtime lifecycle wrapper.

CANN docs: acl.init, acl.finalize, acl.rt.set_device, acl.rt.reset_device,
acl.rt.set_device_res_limit (referenced by the user).
"""
from __future__ import annotations

import threading
from typing import Any

from adapdomain.utils.logging import get_logger

from .errors import AclInitError

_log = get_logger(__name__)


def _import_acl() -> Any:
    try:
        import acl  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise AclInitError(f"acl python module not importable: {exc}")
    return acl


class AclRuntime:
    """One-shot lifecycle. init() must be paired with finalize()."""

    _lock = threading.Lock()
    _initialized: bool = False

    def init(self, config_path: str | None = None) -> None:
        with AclRuntime._lock:
            if AclRuntime._initialized:
                _log.debug("AclRuntime.init: already initialized")
                return
            acl = _import_acl()
            ret = acl.init(config_path) if config_path else acl.init()
            self.check_ret(ret, operation="acl.init")
            AclRuntime._initialized = True
            _log.info("acl initialized")

    def finalize(self) -> None:
        with AclRuntime._lock:
            if not AclRuntime._initialized:
                return
            acl = _import_acl()
            ret = acl.finalize()
            self.check_ret(ret, operation="acl.finalize")
            AclRuntime._initialized = False
            _log.info("acl finalized")

    @staticmethod
    def check_ret(ret: int, operation: str) -> None:
        if ret != 0:
            raise AclInitError(f"acl op failed: {operation}", ret=ret, operation=operation)

"""ACL/CANN healthcheck used by `adapdomain verify-remote`."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapdomain.utils.logging import get_logger
from adapdomain.utils.subprocess import SubprocessError, run_checked

_log = get_logger(__name__)


@dataclass
class HealthcheckReport:
    python_version: str
    cann_env: dict[str, str | None]
    acl_importable: bool
    msprof_available: bool
    npu_smi_available: bool
    device_count: int | None
    smoke_inference_ok: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "cann_env": self.cann_env,
            "acl_importable": self.acl_importable,
            "msprof_available": self.msprof_available,
            "npu_smi_available": self.npu_smi_available,
            "device_count": self.device_count,
            "smoke_inference_ok": self.smoke_inference_ok,
            "notes": self.notes,
        }


def run_healthcheck(*, smoke_om_path: str | None = None) -> HealthcheckReport:
    rep = HealthcheckReport(
        python_version=sys.version.split()[0],
        cann_env={
            "ASCEND_TOOLKIT_HOME": os.environ.get("ASCEND_TOOLKIT_HOME"),
            "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        acl_importable=False,
        msprof_available=shutil.which("msprof") is not None,
        npu_smi_available=shutil.which("npu-smi") is not None,
        device_count=None,
        smoke_inference_ok=False,
    )

    try:
        import acl  # type: ignore
        rep.acl_importable = True
    except Exception as exc:  # noqa: BLE001
        rep.notes.append(f"acl not importable: {exc}")
        return rep

    from .acl_runtime import AclRuntime
    from .device import AscendDevice

    rt = AclRuntime()
    rt.init()
    try:
        rep.device_count = AscendDevice.get_count()
        if smoke_om_path and Path(smoke_om_path).exists():
            try:
                from .context import AscendContext
                from .dataset import AscendDataset
                from .executor import AscendExecutor
                from .model import AscendModel
                from .stream import AscendStream

                device = AscendDevice(0)
                with device.bind(), AscendContext(0), AscendStream():
                    with AscendModel(smoke_om_path) as loaded:
                        with AscendDataset() as inputs, AscendDataset() as outputs:
                            AscendExecutor().warmup(model_id=loaded.model_id, input_dataset=inputs.handle, output_dataset=outputs.handle, runs=1)
                            rep.smoke_inference_ok = True
            except Exception as exc:  # noqa: BLE001
                rep.notes.append(f"smoke inference failed: {exc}")
        else:
            rep.notes.append("smoke om not provided; skipping smoke inference")
    finally:
        rt.finalize()

    if rep.msprof_available:
        try:
            run_checked(["msprof", "--version"], timeout=10)
        except SubprocessError as exc:
            rep.notes.append(f"msprof --version returned non-zero: {exc}")
    return rep

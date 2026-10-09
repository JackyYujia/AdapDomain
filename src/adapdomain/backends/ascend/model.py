"""Ascend .om model load/unload."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclModelError


@dataclass
class LoadedModel:
    model_id: int
    desc: Any
    path: str


class AscendModel:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._loaded: LoadedModel | None = None

    def load(self) -> LoadedModel:
        acl = _import_acl()
        model_id, ret = acl.mdl.load_from_file(self._path)
        if ret != 0:
            raise AclModelError(f"load_from_file failed: {self._path}", ret=ret, operation="mdl.load_from_file")
        desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(desc, model_id)
        if ret != 0:
            raise AclModelError("get_desc failed", ret=ret, operation="mdl.get_desc")
        self._loaded = LoadedModel(model_id=model_id, desc=desc, path=self._path)
        return self._loaded

    def unload(self) -> None:
        if self._loaded is None:
            return
        acl = _import_acl()
        ret = acl.mdl.unload(self._loaded.model_id)
        AclRuntime.check_ret(ret, operation="mdl.unload")
        ret = acl.mdl.destroy_desc(self._loaded.desc)
        AclRuntime.check_ret(ret, operation="mdl.destroy_desc")
        self._loaded = None

    def __enter__(self) -> LoadedModel:
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

"""Versioned, atomic frontier snapshot store."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from adapdomain.core.frontier import FrontierSnapshot


class FrontierStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        d = self._root / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, snap: FrontierSnapshot) -> Path:
        d = self._model_dir(snap.model_name)
        path = d / f"{snap.version}.json"
        payload = snap.model_dump(mode="json")
        with tempfile.NamedTemporaryFile("w", delete=False, dir=d, suffix=".tmp", encoding="utf-8") as tmp:
            json.dump(payload, tmp, sort_keys=True)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        return path

    def load_version(self, model_name: str, version: int) -> FrontierSnapshot:
        path = self._model_dir(model_name) / f"{version}.json"
        return FrontierSnapshot.model_validate(json.loads(path.read_text()))

    def load_latest(self, model_name: str) -> FrontierSnapshot:
        files = sorted(self._model_dir(model_name).glob("*.json"), key=lambda p: int(p.stem))
        if not files:
            raise FileNotFoundError(model_name)
        return self.load_version(model_name, int(files[-1].stem))

"""Append-only JSONL helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


class JsonlReader:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read_all(self, *, tolerant: bool = False) -> Iterator[dict[str, Any]]:
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if tolerant:
                        continue
                    raise

    def filter(self, **conditions: Any) -> Iterator[dict[str, Any]]:
        for rec in self.read_all():
            if all(rec.get(k) == v for k, v in conditions.items()):
                yield rec

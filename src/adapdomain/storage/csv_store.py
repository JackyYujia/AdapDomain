"""Schema-checked CSV append store."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


class CsvStore:
    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        self._path = Path(path)
        self._columns = list(columns)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._columns)

    def append(self, row: dict) -> None:
        missing = [c for c in self._columns if c not in row]
        if missing:
            raise ValueError(f"missing columns for csv {self._path.name}: {missing}")
        with self._path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[c] for c in self._columns])

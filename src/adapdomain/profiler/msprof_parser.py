"""msprof CSV parser tolerant to per-CANN field-name drift."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MsprofSchema:
    cann_version: str
    op_name_keys: list[str]
    duration_keys: list[str]
    cube_keys: list[str]
    vector_keys: list[str]
    hbm_keys: list[str]


class MsprofParser:
    def __init__(self, *, schema: MsprofSchema) -> None:
        self._schema = schema

    def _first_present(self, row: dict[str, str], keys: list[str], cast=float):
        for k in keys:
            if k in row:
                try:
                    return cast(row[k])
                except (TypeError, ValueError):
                    return None
        _log.warning("msprof parser: none of %s present in row", keys)
        return None

    def parse_csv(self, path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out: list[dict[str, Any]] = []
            for row in reader:
                out.append({
                    "op_name": self._first_present(row, self._schema.op_name_keys, cast=str),
                    "duration": self._first_present(row, self._schema.duration_keys),
                    "cube": self._first_present(row, self._schema.cube_keys),
                    "vector": self._first_present(row, self._schema.vector_keys),
                    "hbm": self._first_present(row, self._schema.hbm_keys),
                })
        return out

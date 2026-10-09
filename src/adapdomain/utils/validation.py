"""Small validation helpers used across modules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_in_range(value: float, low: float, high: float, name: str) -> float:
    if not (low <= value <= high):
        raise ValueError(f"{name}={value} not in [{low}, {high}]")
    return value


def ensure_unique(items: Iterable, name: str) -> None:
    seen: set = set()
    for it in items:
        if it in seen:
            raise ValueError(f"{name}: duplicate value {it!r}")
        seen.add(it)


def ensure_path_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path

"""Time helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat(timespec="seconds")


@contextmanager
def stopwatch() -> Iterator[dict]:
    """Yield a dict that gets a 'seconds' key on context exit."""
    start = time.perf_counter()
    state: dict = {}
    try:
        yield state
    finally:
        state["seconds"] = time.perf_counter() - start

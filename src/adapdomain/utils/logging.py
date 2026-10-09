"""Structured logging for AdapDomain."""
from __future__ import annotations

import logging
import os
from typing import Literal

from rich.logging import RichHandler

_LOG_FORMAT_RICH = "%(message)s"
_LOG_FORMAT_PLAIN = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def configure_logging(
    level: int | str = "INFO",
    style: Literal["rich", "plain"] = "rich",
) -> None:
    """Configure root logger. Idempotent."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    if style == "rich" and os.isatty(2):
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_time=False)
        fmt = _LOG_FORMAT_RICH
    else:
        handler = logging.StreamHandler()
        fmt = _LOG_FORMAT_PLAIN

    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

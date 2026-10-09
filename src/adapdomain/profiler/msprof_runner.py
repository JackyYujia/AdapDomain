"""Drive the `msprof` binary; do not parse here (see M19)."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from adapdomain.core.errors import MsprofRunError
from adapdomain.utils.subprocess import SubprocessError, run_checked


@dataclass
class MsprofRunSpec:
    command: Sequence[str]
    metrics: Sequence[str]
    duration_s: float
    run_id: str
    extra_args: Sequence[str] = field(default_factory=list)


class MsprofRunner:
    def __init__(self, *, binary: str = "msprof", output_root: str | Path = "data/raw/msprof") -> None:
        self._binary = binary
        self._output_root = Path(output_root)

    def run(self, spec: MsprofRunSpec) -> Path:
        if not shutil.which(self._binary):
            raise MsprofRunError(f"{self._binary} not on PATH")
        out_dir = self._output_root / spec.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = ",".join(spec.metrics) if spec.metrics else "Default"
        cmd = [
            shutil.which(self._binary) or self._binary,
            "--output", str(out_dir),
            "--application", " ".join(spec.command),
            "--aic-metrics", metrics,
            "--duration", str(int(spec.duration_s)),
            *spec.extra_args,
        ]
        try:
            run_checked(cmd, timeout=spec.duration_s + 60)
        except SubprocessError as exc:
            raise MsprofRunError(f"msprof failed: {exc.stderr.strip()[:200]}") from exc
        return out_dir

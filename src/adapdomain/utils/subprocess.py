"""Subprocess wrappers with strict error handling."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Sequence


class SubprocessError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stdout: str, stderr: str) -> None:
        super().__init__(f"command {list(cmd)!r} failed with code {returncode}: {stderr.strip()[:200]}")
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class RunResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_checked(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: str | None = None,
    env: dict | None = None,
) -> RunResult:
    """Run cmd, capture stdout/stderr, raise SubprocessError on failure or timeout."""
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
        raise SubprocessError(cmd, -1, stdout, stderr + f"\n[timeout after {timeout}s]")

    if proc.returncode != 0:
        raise SubprocessError(cmd, proc.returncode, proc.stdout or "", proc.stderr or "")

    return RunResult(cmd=list(cmd), returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")

"""Run manifest written for every experiment / replay / verify run."""
from __future__ import annotations

import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path


class RunManifest:
    def __init__(self, *, run_id: str, command: str, out_dir: Path) -> None:
        self._run_id = run_id
        self._command = command
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._data: dict = {
            "run_id": run_id,
            "command": command,
            "git_commit_or_none": _git_commit_or_none(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ended_at": None,
            "status": "in_progress",
        }

    @classmethod
    def start(cls, *, run_id: str, command: str, out_dir: Path) -> "RunManifest":
        return cls(run_id=run_id, command=command, out_dir=out_dir)

    def add_config_hash(self, h: str) -> None:
        self._data["config_hash"] = h

    def set_hardware(self, platform_name: str, *, device_count: int, cann_version: str | None) -> None:
        self._data["device_info"] = {
            "platform": platform_name,
            "count": device_count,
            "cann_version": cann_version,
        }

    def set_status(self, status: str) -> None:
        self._data["status"] = status

    def update(self, **kwargs) -> None:
        self._data.update(kwargs)

    def finalize(self) -> Path:
        self._data["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = self._out_dir / "manifest.json"
        path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        return path


def _git_commit_or_none() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None

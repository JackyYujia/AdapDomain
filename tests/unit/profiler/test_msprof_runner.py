import shutil
import sys

import pytest

from adapdomain.profiler.msprof_runner import MsprofRunner, MsprofRunSpec
from adapdomain.core.errors import MsprofRunError


def test_runner_raises_if_binary_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    runner = MsprofRunner(binary="msprof", output_root=tmp_path)
    with pytest.raises(MsprofRunError):
        runner.run(MsprofRunSpec(command=[sys.executable, "-c", "pass"], metrics=[], duration_s=1, run_id="r0"))


def test_runner_creates_unique_output_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/msprof")
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        from adapdomain.utils.subprocess import RunResult
        return RunResult(cmd=list(cmd), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("adapdomain.profiler.msprof_runner.run_checked", fake_run)
    runner = MsprofRunner(binary="msprof", output_root=tmp_path)
    spec = MsprofRunSpec(command=["python", "-c", "pass"], metrics=["PipeUtilization"], duration_s=2, run_id="r0")
    out_dir = runner.run(spec)
    assert out_dir.exists()
    assert calls and calls[0][0] == "/usr/bin/msprof"
    assert "--output" in " ".join(calls[0])

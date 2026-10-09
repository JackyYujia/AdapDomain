import json
from pathlib import Path

import pytest

from adapdomain.backends.base import MockBackend, DeviceInfo
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual
from adapdomain.core.workload import WorkloadTrace
from adapdomain.experiments.replay import ReplayInputs, run_replay
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights


def test_replay_runs_offline(tmp_path):
    trace = WorkloadTrace.from_jsonl("tests/fixtures/sample_traces/tiny.jsonl")
    snap_dict = json.loads(Path("tests/fixtures/sample_frontiers/bert-base.v1.json").read_text())
    snapshots = {"bert-base": FrontierSnapshot.model_validate(snap_dict)}
    inputs = ReplayInputs(
        trace=trace,
        residuals=[DeviceResidual.empty(i, ["cube", "vector"]) for i in range(4)],
        snapshots=snapshots,
        interference=InterferenceTable.from_entries([]),
    )
    res = run_replay(
        inputs=inputs,
        seeds=[1, 2],
        weights=ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05),
        feasibility=FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5),
        eta=0.30,
        backend=MockBackend(devices=[DeviceInfo(device_id=i, name=f"mock-{i}", cube_capacity=20, vector_capacity=40) for i in range(4)]),
        output_root=tmp_path,
    )
    assert (res.out_dir / "summary.csv").exists()
    assert (res.out_dir / "manifest.json").exists()
    assert res.goodput_mean >= 0

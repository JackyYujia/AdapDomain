from pathlib import Path

from adapdomain.backends.base import MockBackend, DeviceInfo
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual
from adapdomain.core.workload import WorkloadTrace
from adapdomain.experiments.replay import ReplayInputs
from adapdomain.experiments.sweeps import sweep_arrival_rate
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights


def test_sweep_runs_offline(tmp_path):
    import json
    trace = WorkloadTrace.from_jsonl("tests/fixtures/sample_traces/tiny.jsonl")
    snap = {"bert-base": FrontierSnapshot.model_validate(json.loads(Path("tests/fixtures/sample_frontiers/bert-base.v1.json").read_text()))}
    base = ReplayInputs(trace=trace, residuals=[DeviceResidual.empty(i, ["cube", "vector"]) for i in range(4)], snapshots=snap, interference=InterferenceTable.from_entries([]))
    pts = sweep_arrival_rate(
        base_inputs=base, rates=[1.0, 2.0],
        build_inputs=lambda b, r: b,
        seeds=[1],
        weights=ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05),
        feasibility=FeasibilityFilter(0.9, 0.9, 0.5),
        eta=0.30,
        backend=MockBackend(devices=[DeviceInfo(device_id=i, name=f"mock-{i}", cube_capacity=20, vector_capacity=40) for i in range(4)]),
        output_root=str(tmp_path),
    )
    assert len(pts) == 2

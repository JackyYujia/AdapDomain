"""`adapdomain replay`."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from adapdomain.backends.base import BackendRegistry
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual
from adapdomain.core.workload import WorkloadTrace
from adapdomain.experiments.replay import ReplayInputs, run_replay
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights
from adapdomain.utils.config import ExperimentReplayConfig, HarmonicFitConfig

app = typer.Typer(help="Replay a workload trace against the scheduler.")


@app.callback(invoke_without_command=True)
def replay(
    config: str = typer.Option("configs/experiments/e2e_replay.yaml"),
    scheduler: str = typer.Option("configs/scheduler/harmonicfit_default.yaml"),
    backend: str = typer.Option("mock", help="Inference backend (mock|ascend)"),
) -> None:
    cfg = ExperimentReplayConfig.from_yaml(config)
    sched = HarmonicFitConfig.from_yaml(scheduler)
    weights = ScoreWeights(
        lambda_c=sched.weights.lambda_c, lambda_x=sched.weights.lambda_x,
        lambda_h=sched.weights.lambda_h, lambda_q=sched.weights.lambda_q,
        lambda_f=sched.weights.lambda_f, lambda_u=sched.weights.lambda_u,
    )
    fs = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    backend_obj = BackendRegistry.default().create(backend)
    trace = WorkloadTrace.from_jsonl("tests/fixtures/sample_traces/tiny.jsonl")
    inputs = ReplayInputs(
        trace=trace,
        residuals=[DeviceResidual.empty(i, ["cube", "vector"]) for i in range(4)],
        snapshots={"bert-base": FrontierSnapshot.model_validate(json.loads(Path("tests/fixtures/sample_frontiers/bert-base.v1.json").read_text()))},
        interference=InterferenceTable.from_entries([]),
    )
    res = run_replay(
        inputs=inputs, seeds=cfg.seeds, weights=weights, feasibility=fs, eta=sched.interference.eta,
        backend=backend_obj, output_root=cfg.output_root,
    )
    typer.echo(str(res.out_dir))

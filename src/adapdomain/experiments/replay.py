"""Trace replay engine."""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from adapdomain.backends.base import InferenceBackend
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.core.workload import WorkloadTrace
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.planner import HarmonicFitPlanner
from adapdomain.scheduler.residual_state import ClusterState
from adapdomain.scheduler.scoring import ScoreWeights
from adapdomain.storage.csv_store import CsvStore
from adapdomain.storage.jsonl import JsonlWriter
from adapdomain.storage.run_manifest import RunManifest


@dataclass
class ReplayResult:
    seeds: list[int]
    goodput_mean: float
    goodput_std: float
    violation_mean: float
    violation_std: float
    out_dir: Path


@dataclass(frozen=True)
class ReplayInputs:
    trace: WorkloadTrace
    residuals: list[DeviceResidual]
    snapshots: Mapping[str, FrontierSnapshot]
    interference: InterferenceTable


def run_replay(
    *,
    inputs: ReplayInputs,
    seeds: list[int],
    weights: ScoreWeights,
    feasibility: FeasibilityFilter,
    eta: float,
    backend: InferenceBackend,
    output_root: str | Path,
) -> ReplayResult:
    output_root = Path(output_root)
    run_id = f"replay-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = output_root / run_id
    manifest = RunManifest.start(run_id=run_id, command="adapdomain replay", out_dir=out_dir)
    summary = CsvStore(out_dir / "summary.csv", columns=["seed", "goodput", "violation"])
    events = JsonlWriter(out_dir / "events.jsonl")

    g_values: list[float] = []
    v_values: list[float] = []
    backend.initialize()
    try:
        for seed in seeds:
            state = ClusterState.from_residuals(inputs.residuals)
            planner = HarmonicFitPlanner(feasibility=feasibility, weights=weights, eta=eta, hbm_high=0.85, queue_high=0.80)
            plan = planner.plan(list(_replicas_from_trace(inputs.trace)), state=state, snapshots=inputs.snapshots, interference=inputs.interference)
            served = []
            for req in inputs.trace.requests:
                if any(rid for rid, did in plan.placements.items() if rid.startswith(req.model)):
                    handle = backend.load_model(model_id=req.model, path=f"/fake/{req.model}.om", device_id=0)
                    res = backend.execute(handle, inputs={}, repeat=1)
                    served.append(float(res.latencies_ms[0]))
                    backend.unload_model(handle)
            window_seconds = max(1.0, max(r.arrival_ms for r in inputs.trace.requests) / 1000.0)
            goodput = len([t for t in served if t <= 15.0]) / window_seconds
            violation = (1 - len([t for t in served if t <= 15.0]) / max(1, len(served))) if served else 0.0
            summary.append({"seed": seed, "goodput": goodput, "violation": violation})
            events.write({"type": "seed_done", "seed": seed, "goodput": goodput, "violation": violation})
            g_values.append(goodput)
            v_values.append(violation)
    finally:
        backend.finalize()

    res = ReplayResult(
        seeds=list(seeds),
        goodput_mean=statistics.mean(g_values) if g_values else 0.0,
        goodput_std=statistics.pstdev(g_values) if len(g_values) > 1 else 0.0,
        violation_mean=statistics.mean(v_values) if v_values else 0.0,
        violation_std=statistics.pstdev(v_values) if len(v_values) > 1 else 0.0,
        out_dir=out_dir,
    )
    manifest.set_status("done")
    manifest.update(metrics={
        "goodput_mean": res.goodput_mean,
        "goodput_std": res.goodput_std,
        "violation_mean": res.violation_mean,
        "violation_std": res.violation_std,
    })
    manifest.finalize()
    return res


def _replicas_from_trace(trace: WorkloadTrace) -> Iterable[TenantShape]:
    seen: set[str] = set()
    for r in trace.replicas:
        if r.replica_id in seen:
            continue
        seen.add(r.replica_id)
        yield TenantShape(
            model_name=r.model, replica_id=r.replica_id,
            domains={"cube": 0.40, "vector": 0.20},
            slo_ms=r.slo_ms, batch_size=r.batch_size,
            frontier_version=1, confidence=0.9,
        )

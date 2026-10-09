# AdapDomain Module Plan

This file mirrors §5 of the design spec
(`docs/superpowers/specs/2026-06-26-adapdomain-implementation-design.md`)
and adds a status column. Every session must update it.

Status values: `pending`, `in_progress`, `done`, `blocked`.

## Stage A — Skeleton and Core Data Model

| # | Module | Files | Parallel | Status | Notes |
|---|---|---|---|---|---|
| M1 | Project skeleton & configs | `README.md`, `pyproject.toml`, `.gitignore`, `cli/main.py`, all `configs/*.yaml` | no | done |  |
| M2 | Utils | `utils/{logging,config,subprocess,time,validation}.py` | no | done |  |
| M3 | Errors & types | `core/{errors,types}.py` | no | done |  |
| M4 | Domain & shape | `core/{domain,shape}.py`, `tests/unit/core/test_shape.py` | yes | done |  |
| M5 | Frontier & confidence | `core/frontier.py`, `tests/unit/core/test_frontier.py` | yes | done |  |
| M6 | Telemetry & workload | `core/{telemetry,workload}.py` | yes | done |  |
| M7 | Interference & placement | `core/{interference,placement}.py` | yes | done |  |
| M8 | Metrics & SLO | `core/metrics.py` | yes | done |  |
| M9 | Storage | `storage/{jsonl,csv_store,frontier_store,run_manifest,artifacts}.py` | yes | done |  |
| M10 | Test infra | `tests/conftest.py`, `tests/fixtures/{sample_msprof,sample_frontiers,sample_traces}/` | yes | done |  |

## Stage B — Hardware Backend and Profiling

| # | Module | Files | Parallel | Status | Notes |
|---|---|---|---|---|---|
| M11 | Backend abstraction | `backends/base.py`, `MockBackend` | no | done |  |
| M12 | ACL runtime | `backends/ascend/{acl_runtime,errors}.py` | no | done |  |
| M13 | Device/Context/Stream | `backends/ascend/{device,context,stream}.py` | yes (3 files) | done |  |
| M14 | Memory | `backends/ascend/memory.py` | yes | done |  |
| M15 | Model/Dataset/Executor | `backends/ascend/{model,dataset,executor}.py` | no | done |  |
| M16 | Resource limit | `backends/ascend/resource_limit.py`, `docs/hardware/ascend_910b3_acl_backend.md` | yes | done | Cube=0, Vector=1 from Huawei Runtime API; Python binding sanity check still recommended |
| M17 | Healthcheck | `backends/ascend/healthcheck.py`, `cli/commands/verify_remote.py` | yes | done |  |
| M18 | msprof runner | `profiler/msprof_runner.py` | no | done |  |
| M19 | msprof parser | `profiler/msprof_parser.py`, `docs/hardware/msprof_data_contract.md` | yes | done | field map pending remote confirmation |
| M20 | Frontier builder | `profiler/{frontier_builder,normalization,confidence}.py` | yes | done |  |
| M21 | ActiveVC | `profiler/{activevc,candidate_grid,acquisition}.py`, `tests/unit/profiler/test_activevc.py` | yes | done |  |
| M22 | Profiling normalization | (in M20) | yes | done | folded into M20/M21 |
| M23 | Remote scripts | `scripts/remote_ascend/*.sh`, `docs/hardware/remote_validation.md` | yes | done |  |

## Stage C — Scheduler, Runtime, and Experiments

| # | Module | Files | Parallel | Status | Notes |
|---|---|---|---|---|---|
| M24 | Feasibility | `scheduler/feasibility.py` | yes | done |  |
| M25 | Scoring & CEFF | `scheduler/scoring.py`, `tests/unit/scheduler/test_scoring.py` | yes | done |  |
| M26 | Risk modules | `scheduler/{hbm_risk,queue_risk,interference_risk,fragmentation}.py` | yes (4 agents) | done |  |
| M27 | Residual state | `scheduler/residual_state.py` | yes | done |  |
| M28 | HarmonicFit planner | `scheduler/{harmonic_fit,planner}.py` | no | done |  |
| M29 | Runtime monitor | `runtime/{monitor,ewma,ring_buffer}.py` | yes | done |  |
| M30 | SmoothGuard | `runtime/smooth_guard.py` | yes | done |  |
| M31 | PreScale | `runtime/prescale.py` | yes | done |  |
| M32 | NightShift | `runtime/nightshift.py` | yes | done |  |
| M33 | Repair actions | `runtime/repair_actions.py` | yes | done |  |
| M34 | Replay engine | `experiments/{replay,seeds}.py` | no | done |  |
| M35 | Sweeps | `experiments/sweeps.py` | yes | done |  |
| M36 | ActiveVC ablation | `experiments/ablations.py` | yes | done |  |
| M37 | HarmonicFit ablation | (in `experiments/ablations.py`) | yes | done |  |
| M38 | Overhead | `experiments/overhead.py` | yes | done |  |
| M39 | Plotting | `experiments/plotting.py` | yes | done |  |
| M40 | Tables | `experiments/tables.py` | yes | done |  |
| M41 | Paper result bundle | `cli/commands/plot.py`, `scripts/remote_ascend/run_paper_bundle.sh` | no | done |  |

## Stage Completion Markers

| Stage | Status |
|---|---|
| Stage A | done |
| Stage B | done |
| Stage C | done |

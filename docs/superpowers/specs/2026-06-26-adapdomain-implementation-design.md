---
title: AdapDomain Implementation Design
date: 2026-06-26
status: draft, pending user review
git: workspace is not a git repository; this spec is not committed
paper_source: /Users/xiumuyoukediaoye/Desktop/docPaper/SOCC_2026/adapdomain_socc2026_polished.tex
target_root: /Users/xiumuyoukediaoye/Desktop/docPaper/AdapDomain
---

# AdapDomain Implementation Design

This spec defines the implementation of the AdapDomain paper project as a real-hardware system targeting four Ascend 910B3 NPU cards (Cube20 / Vector40). It is the contract that subsequent implementation plans and code must follow.

The spec is intentionally complete and self-contained so that work can resume after context compaction. Section 5 (Module Plan) is the authoritative module table; Section 8 (Continuation Hooks) tells future sessions where to resume.

## 0. Scope and Non-Goals

In scope (Stage A–C of Section 5):

- Real-hardware inference backend on 4× Ascend 910B3 via ACL Python.
- ActiveVC feasible-frontier profiling driven by msprof / Ascend Profiler.
- HarmonicFit complementarity-aware placement with interference, HBM, queue, fragmentation, and idle-card risk terms.
- SmoothGuard / PreScale / NightShift runtime adaptation sharing a single residual-shape state.
- Experiment pipeline reproducing the paper figures and tables (`fig2`–`fig8`, `tab:e2e`, `tab:ablation`, `tab:overhead`).
- Remote validation scripts for the Ascend Linux server.

Out of scope for stage 1:

- Cluster-wide scheduler or admission controller.
- Web dashboard, persistent daemon, REST/gRPC API.
- Kubernetes / orchestration integration.
- Automatic ONNX/PyTorch → `.om` conversion (the user supplies `.om` files).
- Mock-first development. Mock backend exists only for unit tests; the ACL backend is the primary path.

## 1. Hardware and Software Assumptions

| Item | Assumption |
|---|---|
| Hardware | 4× Ascend 910B3 NPU on one Linux server |
| Resource domains | Cube20 (matrix), Vector40 (vector) per card |
| Driver/runtime | CANN Community Edition (target 8.3.RC1 family) |
| Python backend | ACL Python (`acl`, `acl.rt`, `acl.mdl`, `acl.op` as applicable) |
| Profiler | `msprof` / Ascend Profiler |
| Models | Pre-built `.om` files provided by the user |
| Dev environment | macOS (this workstation) — code only, no NPU execution |
| Verification environment | Remote Linux NPU server with full CANN toolchain |

`acl.rt.set_device_res_limit(device_id, type, value)` (referenced by the user) is the only resource-limit API treated as part of the contract. It must be called after `set_device` and before any operator execution, and last call wins per device. Huawei's Ascend Runtime API documents Cube as `0` and Vector as `1`; the values are provided through configuration so they can still be overridden for a different CANN target.

## 2. Architecture Overview

```
CLI / Experiment Driver
        │
Orchestration Layer
        │
   ┌────┴─────┐
Scheduler   Profiler
   │           │
   └─── Core ──┘
   │           │
Runtime    Hardware Backend
```

Layers:

- **CLI** (`src/adapdomain/cli/`) — single entry `adapdomain` (Typer); all commands route through configs.
- **Orchestration** (`experiments/`, `storage/`, `utils/`) — config loader, run manifest, artifact bundling, remote runner.
- **Scheduler** (`scheduler/`) — HarmonicFit planner; depends only on `core/` types and abstract backend/telemetry.
- **Profiler** (`profiler/`) — ActiveVC sampler and msprof integration; depends on abstract `InferenceBackend` and `MsprofRunner`.
- **Core** (`core/`) — pure data model, domains, shapes, frontiers, telemetry, interference, placement, metrics, errors, types. No hardware imports.
- **Runtime** (`runtime/`) — SmoothGuard, PreScale, NightShift, repair actions; reads/writes shared `ClusterState`.
- **Hardware backend** (`backends/ascend/`) — the only place that imports `acl`. Wraps device, context, stream, memory, model, dataset, executor, resource limit, healthcheck.

Layering invariants:

1. `core/` has no dependency outside the standard library + pydantic + numpy.
2. `scheduler/`, `profiler/`, `runtime/` may not import `acl` or `msprof` directly.
3. `backends/ascend/*` is the only module allowed to import `acl`.
4. `experiments/` may compose all of the above, but writes only through `storage/`.
5. Any module-to-module exchange uses `core/` data classes or `storage/` artifacts; mutable internal state is private.

## 3. Directory Layout

```text
AdapDomain/
  README.md
  pyproject.toml
  .gitignore

  configs/
    hardware/ascend_910b3_4card.yaml
    models/model_registry.example.yaml
    profiling/activevc_default.yaml
    scheduler/harmonicfit_default.yaml
    runtime/runtime_default.yaml
    experiments/e2e_replay.yaml
    experiments/ablation.yaml

  src/adapdomain/
    __init__.py
    core/        # domain, shape, frontier, telemetry, interference, placement, workload, metrics, errors, types
    backends/    # base.py, ascend/ (acl_runtime, device, context, stream, memory, model, dataset, executor, resource_limit, healthcheck, errors)
    profiler/    # activevc, candidate_grid, acquisition, msprof_runner, msprof_parser, frontier_builder, confidence, normalization
    scheduler/   # harmonic_fit, planner, feasibility, scoring, residual_state, fragmentation, queue_risk, hbm_risk, interference_risk
    runtime/     # ewma, smooth_guard, prescale, nightshift, repair_actions, monitor, ring_buffer
    experiments/ # profile_vm, collect_interference, replay, sweeps, ablations, overhead, plotting, tables, seeds
    storage/     # jsonl, csv_store, frontier_store, run_manifest, artifacts
    cli/         # main.py, commands/{profile,schedule,run_model,replay,ablate,plot,verify_remote}.py
    utils/       # logging, config, subprocess, time, validation

  scripts/remote_ascend/
    setup_env.sh, run_healthcheck.sh, run_profile.sh, run_e2e_replay.sh, collect_msprof.sh, run_paper_bundle.sh

  tests/
    unit/{core,profiler,scheduler,runtime,storage}/
    integration/{test_acl_backend_import_guard.py, test_cli_smoke.py}
    fixtures/{sample_msprof,sample_frontiers,sample_traces}/

  docs/
    superpowers/specs/2026-06-26-adapdomain-implementation-design.md  (this file)
    hardware/{ascend_910b3_acl_backend.md, msprof_data_contract.md, remote_validation.md}
    development/{module_plan.md, continuation_checklist.md}

  data/
    raw/, processed/, frontiers/, interference/, traces/, results/, figures/, tables/
```

## 4. Core Data Model

Defined under `core/`. All models are pydantic v2.

### 4.1 DomainSpec

```text
DomainSpec
  name: "cube" | "vector" | "hbm" | str
  capacity: float
  unit: str
  schedulable: bool
  telemetry_key: str
```

For 910B3 Stage 1: `cube.capacity = 20`, `vector.capacity = 40`, both `schedulable = true`; HBM enters HarmonicFit only as a risk term `H(i,k)`.

### 4.2 TenantShape

```text
TenantShape
  model_name: str
  replica_id: ReplicaId
  domains: dict[str, float]   # normalized to [0,1] per Eq. (kdomain_normalization)
  slo_ms: float
  batch_size: int
  frontier_version: FrontierVersion
  confidence: float in [0,1]
```

Normalization (paper Eq. `kdomain_normalization`):

```
r_i^j = (x_i^j - x_min^j) / (x_sat^j - x_min^j)
```

### 4.3 DeviceResidual

```text
DeviceResidual
  device_id: DeviceId
  residual: dict[str, float]  # R_k^j = max(0, 1 - Σ r_i^j)
  placed_replicas: list[ReplicaId]
  hbm_pressure: float
  queue_pressure: float
  updated_at: datetime
```

### 4.4 FrontierPoint and FrontierSnapshot

```text
FrontierPoint
  model_name, point_id
  raw_cube, raw_vector, raw_hbm
  norm_cube, norm_vector
  batch_size, slo_ms
  p50_ms, p95_ms, p99_ms
  goodput_qps
  feasible: bool
  confidence: float
  msprof_artifact_path: Path | None

FrontierSnapshot
  model_name, version: FrontierVersion
  k_domains: list[str]
  points: list[FrontierPoint]
  safe_point_id
  dominant_domain: "matrix"|"vector"|"balanced"
  confidence_summary
  created_at
  source_run_id: RunId
```

### 4.5 InterferenceEntry

```text
InterferenceEntry
  model_a, model_b
  direction: "a_on_b" | "b_on_a"
  pressure_bucket: str
  slowdown, p95_delta, goodput_delta
  source_run_id
```

Aggregation (paper Eq. `interference_aggregation`):

```
X(i, k) = max_j X_pair(i, j) + η · Σ_j X_pair(i, j),  η = 0.30
```

### 4.6 PlacementPlan

```text
PlacementPlan
  plan_id, created_at
  input_replicas
  placements: dict[ReplicaId, DeviceId]
  rejected:   dict[ReplicaId, str]
  residuals_after: list[DeviceResidual]
  score_breakdown: dict[(ReplicaId, DeviceId), ScoreBreakdown]
  frontier_versions: dict[ModelId, FrontierVersion]

ScoreBreakdown
  ceff, x, h, q, f, u, final
```

### 4.7 Error Hierarchy

```text
AdapDomainError
├── ConfigError
├── BackendError
│   ├── AclInitError
│   ├── AclDeviceError
│   ├── AclModelError
│   ├── AclMemoryError
│   └── AclExecuteError
├── ProfilerError
│   ├── MsprofRunError
│   └── MsprofParseError
├── SchedulerError
│   ├── InfeasiblePlacementError
│   └── EmptyCandidateError
└── RuntimeAdapterError
    └── RepairBudgetExceededError
```

No bare `Exception` is allowed outside `core/errors.py`.

## 5. Module Plan (Authoritative)

Modules are grouped into three stages. Each module lists files, intent, parallelism, and completion criteria. The full table is the contract for implementation.

### Stage A — Skeleton and Core Data Model

| # | Module | Files | Parallel | Completion |
|---|---|---|---|---|
| M1 | Project skeleton & configs | `README.md`, `pyproject.toml`, `.gitignore`, `cli/main.py`, all `configs/*.yaml` | no | `pip install -e .` works; `adapdomain --help` lists placeholders; all default YAML loads via pydantic |
| M2 | Utils | `utils/{logging,config,subprocess,time,validation}.py` | no | pydantic config models cover hardware/profiling/scheduler/runtime/experiments; `tests/unit/utils/` green |
| M3 | Errors & types | `core/{errors,types}.py` | no | Error hierarchy from §4.7 in place; constants `EPSILON_DOMAIN_FIT=1e-6`, `DEFAULT_K=2` |
| M4 | Domain & shape | `core/{domain,shape}.py`, `tests/unit/core/test_shape.py` | yes | Normalization correct; harmonic case `(0.9,0.1) < (0.5,0.5)` enforced |
| M5 | Frontier & confidence | `core/frontier.py`, `tests/unit/core/test_frontier.py` | yes | Frontier query < 1 ms over 100 points; stale frontier confidence decays monotonically |
| M6 | Telemetry & workload | `core/{telemetry,workload}.py` | yes | EWMA convergence, JSONL roundtrip, trace slicing tested |
| M7 | Interference & placement | `core/{interference,placement}.py` | yes | max-only / sum-only / damped-sum comparisons; `apply_plan` invariants |
| M8 | Metrics & SLO | `core/metrics.py` | yes | `compute_goodput`, `slo_violation_rate`, multi-seed aggregator deterministic on fixtures |
| M9 | Storage | `storage/{jsonl,csv_store,frontier_store,run_manifest,artifacts}.py` | yes | Versioned frontier rollback; atomic write; corrupted JSONL tolerated |
| M10 | Test infra | `tests/conftest.py`, `tests/fixtures/{sample_msprof,sample_frontiers,sample_traces}/` | yes | `pytest -q tests/unit/` green without NPU; ACL tests use `pytest.importorskip("acl")` |

Stage A completion: `pip install -e .` works; all unit tests in `core/` and `storage/` pass; `continuation_checklist.md` marks `STAGE A DONE`.

### Stage B — Hardware Backend and Profiling

| # | Module | Files | Parallel | Completion |
|---|---|---|---|---|
| M11 | Backend abstraction | `backends/base.py`, `MockBackend` | no | Mock backend smoke OK; no upstream module imports `acl` |
| M12 | ACL runtime | `backends/ascend/{acl_runtime,errors}.py` | no | Remote `init`/`finalize` returns 0 |
| M13 | Device/Context/Stream | `backends/ascend/{device,context,stream}.py` | yes (3 files) | 4-card `set_device` smoke; resource cleanup on error |
| M14 | Memory | `backends/ascend/memory.py` | yes | 1 MB / 16 MB / 256 MB malloc/free loop ×100 no leak |
| M15 | Model/Dataset/Executor | `backends/ascend/{model,dataset,executor}.py` | no (depends on M13/M14) | ResNet-152 100× inference on real hardware; resources released on error |
| M16 | Resource limit | `backends/ascend/resource_limit.py`, `docs/hardware/ascend_910b3_acl_backend.md` | yes | Cube=10 / Vector=20 on device 0 verified via msprof |
| M17 | Healthcheck | `backends/ascend/healthcheck.py`, `cli/commands/verify_remote.py` | yes | `adapdomain verify-remote` outputs full JSON report |
| M18 | msprof runner | `profiler/msprof_runner.py` | no | Real msprof directory generated for one model |
| M19 | msprof parser | `profiler/msprof_parser.py`, `docs/hardware/msprof_data_contract.md` | yes | Fixtures cover ≥ 2 field variants; remote sample parses without warnings |
| M20 | Frontier builder | `profiler/{frontier_builder,normalization,confidence}.py` | yes | Approximates paper `tab:model_profiles` on the sample data |
| M21 | ActiveVC | `profiler/{activevc,candidate_grid,acquisition}.py`, `tests/unit/profiler/test_activevc.py` | yes | Synthetic study: at `B=24` boundary error below random/binary/BO; budget sweep monotonic |
| M22 | Profiling normalization | (in M20) | yes | Capacities 20/40 reflected; fallback decreases confidence |
| M23 | Remote scripts | `scripts/remote_ascend/*.sh`, `docs/hardware/remote_validation.md` | yes | All six scripts (setup, healthcheck, profile, msprof, e2e replay, paper bundle) exit 0 on a clean remote machine |

Stage B completion: `adapdomain run-model`, `adapdomain profile activevc`, `adapdomain verify-remote` all pass on the remote 910B3; `tests/unit/profiler/` green; `STAGE B DONE` in checklist.

### Stage C — Scheduler, Runtime, and Experiments

| # | Module | Files | Parallel | Completion |
|---|---|---|---|---|
| M24 | Feasibility | `scheduler/feasibility.py` | yes | 6 reject reasons covered; empty pool raises `EmptyCandidateError` |
| M25 | Scoring & CEFF | `scheduler/scoring.py`, `tests/unit/scheduler/test_scoring.py` | yes | `(0.9,0.1) < (0.5,0.5)` enforced in tests |
| M26 | Risk modules | `scheduler/{hbm_risk,queue_risk,interference_risk,fragmentation}.py` | yes (4 agents) | Risk values in `[0,1]`; deterministic on fixtures |
| M27 | Residual state | `scheduler/residual_state.py` | yes | Per-device locking; transactional apply/revert |
| M28 | HarmonicFit planner | `scheduler/{harmonic_fit,planner}.py` | no (integrates M24–M27) | Reproduces `tab:overhead` planning time order of magnitude; ablation flags reproduce `tab:ablation` trends |
| M29 | Runtime monitor | `runtime/{monitor,ewma,ring_buffer}.py` | yes | 1 Hz sampling; lock-free buffer; EWMA tested |
| M30 | SmoothGuard | `runtime/smooth_guard.py` | yes | Two-tick trigger; cooldown ≥ 30 s; false-alarm reduction matches paper trend |
| M31 | PreScale | `runtime/prescale.py` | yes | Picks device by matching scarce-domain residual, not by scalar free capacity |
| M32 | NightShift | `runtime/nightshift.py` | yes | Migrates by ΔCEFF under `migration_budget` |
| M33 | Repair actions | `runtime/repair_actions.py` | yes | Quota / migration / one-card rebalance audit logged |
| M34 | Replay engine | `experiments/{replay,seeds}.py` | no | Reproduces `tab:e2e` ordering: AdapDomain > Dilu-style > BFD-2D > Exclusive |
| M35 | Sweeps | `experiments/sweeps.py` | yes | arrival-rate, SLO, mix, churn sweeps with 5 seeds each |
| M36 | ActiveVC ablation | `experiments/ablations.py` (shared with M37) | yes | Comparison vs Random / Binary / BO at fixed budget; budget sweep |
| M37 | HarmonicFit ablation | (in `experiments/ablations.py`) | yes | Full / No-X / No-CEFF / No-HBM / No-F / No-U; weight heatmap |
| M38 | Overhead | `experiments/overhead.py` | yes | replicas ∈ {10, 100, 400, 800, 1000}; mean/p95/max recorded |
| M39 | Plotting | `experiments/plotting.py` | yes | Reproduces fig2–fig8 |
| M40 | Tables | `experiments/tables.py` | yes | Outputs `tab:e2e`, `tab:ablation`, `tab:overhead`, `tab:model_profiles`, `tab:protocol`, `tab:default_params` in LaTeX + markdown |
| M41 | Paper result bundle | `cli/commands/plot.py`, `scripts/remote_ascend/run_paper_bundle.sh` | no | `adapdomain plot paper` produces a complete `data/figures/` + `data/tables/` bundle |

Stage C completion: end-to-end replay, ablations, overhead study, plot bundle all succeed on the remote 910B3; all unit + integration tests green; `STAGE C DONE` in checklist.

## 6. Default Configuration Values

These values are the paper defaults and the starting point for `configs/` files. Sensitivity ranges follow `tab:default_params` in the paper.

ActiveVC:

- `B = 24` total probes; sweep range `8..48`
- `alpha = 0.40`, `beta = 0.40`, `gamma = 0.20`, `tau = 0.15`
- Seed grid `4 × 4 × 3` (V × M × HBM); boundary refinement `8 × 8` (V × M)
- HBM held near the safe median observed during seed phase

HarmonicFit:

- `lambda_c = 0.40`, `lambda_x = 0.25`, `lambda_h = 0.15`, `lambda_q = 0.10`, `lambda_f = 0.07`, `lambda_u = 0.05`
- Interference damping `eta = 0.30`
- `EPSILON_DOMAIN_FIT = 1e-6`
- Default `K = 2` (cube, vector)
- Hard feasibility filter precedes scoring; no soft-filtering via score

Runtime:

- EWMA `alpha_ewma = 0.20`
- SmoothGuard requires two consecutive EWMA crossings; cooldown ≥ 30 s
- Quota repair → local migration → one-card rebalance → trigger re-profiling
- NightShift migration budget per low-load window: ≤ 2 replicas
- PreScale prefers the device whose residual is largest in the predicted scarce domain (not largest in scalar free capacity)

Hardware:

- 4× Ascend 910B3, device ids `0..3`
- `cube_capacity = 20`, `vector_capacity = 40`
- `acl.rt.set_device_res_limit` type enum is provided via `configs/hardware/ascend_910b3_4card.yaml`; Cube is `0`, Vector is `1`, and the remote machine should still sanity-check the Python binding

Experiments:

- E2E replay: 5 random seeds, 2400 scheduling windows
- Sweep arrival rates: `0.5×, 0.75×, 1.0×, 1.25×, 1.5×, 1.75×, 2.0×`
- Overhead replica sizes: `{10, 100, 400, 800, 1000}`

## 7. Verification and Testing Strategy

Local (macOS, no NPU):

- `pytest -q tests/unit/` runs on `core/`, `scheduler/`, `profiler/` (algorithm only), `runtime/`, `storage/`, `utils/`.
- ACL backend tests use `pytest.importorskip("acl")` so absence of CANN does not break the suite.
- `tests/integration/test_acl_backend_import_guard.py` asserts that no non-backend module imports `acl`.
- `tests/integration/test_cli_smoke.py` invokes `adapdomain --help` and validates each subcommand placeholder loads its config.

Remote (Linux 910B3):

- `scripts/remote_ascend/setup_env.sh` then `run_healthcheck.sh` must exit 0.
- `run_profile.sh` runs ActiveVC for each model; produces `data/frontiers/<model>/<version>.json`.
- `collect_msprof.sh` produces a parseable msprof directory.
- `run_e2e_replay.sh` runs the full replay; produces `data/results/replay/<run_id>/summary.csv`.
- `run_paper_bundle.sh` produces all figures and tables under `data/figures/` and `data/tables/`.

Every run writes `manifest.json` and `events.jsonl` under its result directory (Section 8.2 of the paper-aligned design). A failed run is reproducible from the manifest alone.

## 8. Continuation Hooks (Resume After Context Compaction)

Two files are the source of truth across sessions:

- `docs/development/module_plan.md` — full module table copied from §5, with status column.
- `docs/development/continuation_checklist.md` — append-only log: one line per module start / module done / verification result.

Rules for every session:

1. Before any code change, read both files.
2. Before starting module `Mxx`, append `Mxx START YYYY-MM-DD expected files: …` to `continuation_checklist.md`.
3. On completion, append `Mxx DONE YYYY-MM-DD <verification command output summary>` and update the status column in `module_plan.md`.
4. Subagents must be given (a) the spec path, (b) the checklist path, (c) the target module IDs, (d) the strict file boundary; they may not edit outside their module files.
5. Cross-module shared state lives only in `core/` types or `storage/` artifacts. No subagent introduces hidden globals.

This is the protocol that lets compacted sessions resume work without re-deriving design context.

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| ACL enum values for `set_device_res_limit` may vary across CANN targets | Default 910B3 config uses Huawei-documented Cube `0` and Vector `1`; resource limiter falls back to no-op with a warning when values are `null` |
| `msprof` schema differs by CANN version | `MsprofSchema` YAML; parser logs warnings instead of crashing; tests include ≥ 2 fixture variants |
| Frontier becomes stale after model update | Confidence model decays; SmoothGuard escalates to targeted ActiveVC refresh, not full rescan |
| Long-running NPU leaks | Strict context-manager cleanup in every backend object; M14 leak test loops malloc/free; M15 forces unload on error |
| Subagent edits stomp each other | Each subagent receives a closed list of files; integration step verifies no out-of-bounds edits |
| Workspace is not a git repo | Spec and code are written but never committed in this workspace; if the user later initializes git, history starts there |

## 10. Open Items (Tracked in module_plan.md)

- Sanity-check the Python ACL binding for `acl.rt.set_device_res_limit` on the deployed CANN version; the default 910B3 YAML already uses Cube `0` and Vector `1`.
- Confirmed msprof field names for Cube/Vector/HBM utilization on the deployed CANN version — fill in `configs/profiling/activevc_default.yaml` / `MsprofSchema`.
- Exact pressure-bucket thresholds for `InterferenceTable` — calibrate from the first interference-collection run.


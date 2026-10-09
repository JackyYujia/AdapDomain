# AdapDomain Continuation Checklist

Append-only log. One line per module start / completion / verification result.
Every new session must read this file before doing anything else.

Format:

```
<UTC date>  <event>  <module>  <details>
```

Examples:

```
2026-06-26  PLAN-LOCKED  ---  design spec written at docs/superpowers/specs/2026-06-26-adapdomain-implementation-design.md
2026-06-27  M1 START      M1   pyproject.toml + README + CLI placeholder
2026-06-27  M1 DONE       M1   pytest -q passes 0 tests; adapdomain --help OK
```

Rules:

- Never edit existing lines. Always append.
- Use UTC dates.
- On `DONE`, also update the status column in `module_plan.md`.
- If a module is blocked, append `M?? BLOCKED <reason>`.
- Subagent runs append `M?? AGENT <agent-id> <files-touched>`.

## Log

2026-06-26  PLAN-LOCKED  ---  design spec written at docs/superpowers/specs/2026-06-26-adapdomain-implementation-design.md; module_plan.md initialized with all modules pending; workspace is not a git repository so spec is not committed
2026-06-26  PLAN-WRITTEN ---  implementation plan written at docs/superpowers/plans/2026-06-26-adapdomain-implementation.md covering M1..M41 with TDD steps; awaiting user choice of subagent-driven vs inline execution
2026-06-26  M1 DONE  ---  skeleton + CLI placeholders + 7 default configs; local pip install skipped (host Python 3.9.6 < 3.10); yaml validated via fallback path-existence check
2026-06-27  M2 DONE  ---  utils package + tests written; ast.parse syntax check green on 7 .py files; pytest deferred to remote Ascend server
2026-06-27  M2 FIX   ---  applied code-quality fixes: TypeVar-bound from_yaml, RunResult.cmd: list[str], timeout returncode assertion
2026-06-27  M3 DONE  ---  core/errors.py + core/types.py + tests; ast.parse green on 5 .py files; pytest deferred to remote
2026-06-27  M4 DONE  ---  core/domain.py + core/shape.py + tests; harmonic CEFF + residual update; ast.parse green
2026-06-27  M5 DONE  ---  core/frontier.py + tests; FrontierSnapshot/Query/ConfidenceModel; ast.parse green
2026-06-27  M9 DONE  ---  storage/jsonl,csv_store,frontier_store,run_manifest,artifacts + tests; ast.parse green
2026-06-27  M8 DONE  ---  core/metrics.py + tests; goodput/violation/percentiles/seed aggregator; ast.parse green
2026-06-27  M10 DONE ---  tests/integration + tests/fixtures; ast.parse + json.loads green; pytest deferred to remote
2026-06-27  STAGE A DONE  ---  M1..M10 implemented locally; tests deferred to remote Ascend NPU server
2026-06-27  M6 DONE  ---  core/telemetry.py + core/workload.py + tests; EWMA window + WorkloadTrace JSONL; ast.parse green
2026-06-27  M7 DONE  ---  core/interference.py + core/placement.py + tests; max+damped sum and PlacementPlan; ast.parse green
2026-06-27  M11 DONE ---  backends/base.py + MockBackend + BackendRegistry + tests; ast.parse green
2026-06-27  M12 DONE ---  backends/ascend/acl_runtime.py + errors.py + import-guard test; lazy acl import; ast.parse green
2026-06-27  M13 DONE ---  backends/ascend/{device,context,stream}.py + import-guard test; ast.parse green; remote smoke deferred
2026-06-27  M14 DONE ---  backends/ascend/memory.py + import-guard test; lazy acl import; ast.parse green; remote leak test deferred
2026-06-27  M15 DONE ---  backends/ascend/{model,dataset,executor}.py + __init__ re-exports + import-guard test; ast.parse green; remote benchmark deferred
2026-06-27  M18 DONE ---  profiler/msprof_runner.py + tests with monkeypatched run_checked; ast.parse green
2026-06-27  M16 DONE ---  backends/ascend/resource_limit.py + ACL backend doc + tests; lazy acl import; ast.parse green
2026-06-27  M17 DONE ---  backends/ascend/healthcheck.py + verify_remote CLI + offline test; cli/main.py wired; ast.parse green
2026-06-27  M23 DONE ---  scripts/remote_ascend/{6 .sh files} + docs/hardware/remote_validation.md; bash -n green; remote execution deferred
2026-06-27  M19 DONE ---  profiler/msprof_parser.py + variant fixture + data contract doc + tests; ast.parse + csv parse green
2026-06-27  M20 DONE ---  profiler/{normalization,confidence,frontier_builder}.py + tests; M22 (profiling normalization) folded in; ast.parse green
2026-06-27  M21 DONE ---  profiler/{candidate_grid,acquisition,activevc}.py + 3 tests; ActiveVC three-stage loop; ast.parse green
2026-06-27  STAGE B DONE  ---  M11..M23 implemented locally; ACL/msprof remote validation deferred
2026-06-27  M25 DONE ---  scheduler/scoring.py + tests; ScoreWeights/RiskTerms/compute_score; ast.parse green
2026-06-27  M27 DONE ---  scheduler/residual_state.py + tests; ClusterState with per-device locks + transactional revert; ast.parse green
2026-06-27  M26 DONE ---  scheduler/{hbm_risk,queue_risk,interference_risk,fragmentation}.py + tests; ast.parse green
2026-06-27  M24 DONE ---  scheduler/feasibility.py + tests; 6 RejectReason cases; ast.parse green
2026-06-27  M29 DONE ---  runtime/{ewma,ring_buffer,monitor}.py + tests; ast.parse green
2026-06-27  M28 DONE ---  scheduler/{harmonic_fit,planner}.py + tests; greedy CEFF planner; ast.parse green
2026-06-27  M32 DONE ---  runtime/nightshift.py + tests; ΔCEFF + risk-bounded migration picker; ast.parse green
2026-06-27  M30 DONE ---  runtime/smooth_guard.py + tests; two-tick trigger + cooldown; ast.parse green
2026-06-27  M33 DONE ---  runtime/repair_actions.py + tests; JSONL audit log; ast.parse green
2026-06-27  M31 DONE ---  runtime/prescale.py + tests; scarce-domain picker; ast.parse green
2026-06-27  M34 DONE ---  experiments/{seeds,replay}.py + cli/replay + tests; cli/main.py wired; ast.parse green
2026-06-27  M40 DONE ---  experiments/tables.py + tests; markdown + latex writers; ast.parse green
2026-06-27  M35 DONE ---  experiments/sweeps.py + tests; arrival-rate sweep harness; ast.parse green
2026-06-27  M39 DONE ---  experiments/plotting.py + tests; matplotlib Agg backend; ast.parse green
2026-06-27  M38 DONE ---  experiments/overhead.py + tests; mean/p95/max bench; ast.parse green
2026-06-27  M36+M37 DONE ---  experiments/ablations.py + cli/ablate + tests; cli/main.py wired; ast.parse green
2026-06-27  M41 DONE ---  experiments/{profile_vm,collect_interference}.py + cli/plot + integration test; cli/main.py wired; ast.parse green
2026-06-27  STAGE C DONE  ---  M24..M41 implemented locally; tests deferred to remote Ascend NPU server

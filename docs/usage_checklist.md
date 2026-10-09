# AdapDomain Usage Checklist

A step-by-step checklist for running AdapDomain on the production 4× Ascend 910B3 NPU server.

- Local dev machine (macOS): write code only — no Python ≥3.10, no CANN.
- Remote Linux NPU server: build env, run tests, ACL/msprof, full replay, paper bundle.
- Status of every module: [module_plan.md](development/module_plan.md).
- Append-only progress log: [continuation_checklist.md](development/continuation_checklist.md).
- Design spec: [2026-06-26-adapdomain-implementation-design.md](superpowers/specs/2026-06-26-adapdomain-implementation-design.md).
- Implementation plan: [2026-06-26-adapdomain-implementation.md](superpowers/plans/2026-06-26-adapdomain-implementation.md).

---

## 1. Prerequisites (remote server)

| Item | Required | How to check |
| --- | --- | --- |
| OS | Linux (CentOS / Ubuntu / EulerOS) | `uname -a` |
| Python | ≥ 3.10 | `python3 --version` |
| CANN | 8.3.RC1 (toolkit installed at `/usr/local/Ascend/ascend-toolkit/`) | `ls /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Ascend devices | 4× 910B3 visible | `npu-smi info` |
| Profiler binary | `msprof` on `PATH` | `command -v msprof` |
| Driver tool | `npu-smi` on `PATH` | `command -v npu-smi` |
| `.om` models | Pre-built `.om` for the seven evaluation models | path under e.g. `/data/models/*.om` |

If the CANN env script lives somewhere else, export `CANN_ENV` before `setup_env.sh`:

```bash
export CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

## 2. Initial setup (one-time)

- [ ] Copy the project tree to the server:

```bash
rsync -av --exclude data/ --exclude .venv/ --exclude __pycache__/ \
  /Users/xiumuyoukediaoye/Desktop/docPaper/AdapDomain/ \
  user@npu-host:/path/on/remote/AdapDomain/
```

- [ ] Source CANN + create the venv + editable install:

```bash
cd /path/on/remote/AdapDomain
bash scripts/remote_ascend/setup_env.sh
```

Expected outcome: `.venv/` exists, `pip install -e .[dev]` succeeds, last line prints `ok: env ready`.

- [ ] Activate the venv for subsequent commands in the same shell:

```bash
source .venv/bin/activate
```

---

## 3. Local-equivalent verification on the server (no NPU yet)

Everything that was deferred on macOS runs here.

- [ ] Run the full unit + integration test suite:

```bash
pytest -q tests/unit/ tests/integration/
```

Expected: every test passes. ACL-touching tests use `pytest.importorskip("acl")` and will succeed once CANN is installed.

- [ ] Verify the CLI surface:

```bash
adapdomain --help
adapdomain version
```

Expected: `--help` lists `version`, `verify-remote`, `run-model`, `profile`, `schedule`, `replay`, `ablate`, `plot`; `adapdomain version` prints `0.1.0`.

- [ ] Append a checklist row in [continuation_checklist.md](development/continuation_checklist.md):

```
<UTC date>  REMOTE-INSTALL DONE  ---  setup_env.sh + pytest -q OK; adapdomain --help OK
```

---

## 4. Hardware healthcheck

- [ ] Quick check (no `.om` smoke):

```bash
bash scripts/remote_ascend/run_healthcheck.sh
```

- [ ] Optional `.om` smoke (one warmup inference on device 0):

```bash
SMOKE_OM=/data/models/resnet152.om bash scripts/remote_ascend/run_healthcheck.sh
```

Expected outcome:

- exit code 0
- `data/results/healthcheck/<run-id>/report.json` exists
- in `report.json`: `acl_importable == true`, `msprof_available == true`, `npu_smi_available == true`, `device_count == 4`, `smoke_inference_ok == true` (only if `SMOKE_OM` was set).
- `notes` is empty (or only mentions skipped smoke).

If anything fails, fix the env before continuing. Common fixes:

| Symptom | Fix |
| --- | --- |
| `acl not importable` | Re-source `set_env.sh`; check that `LD_LIBRARY_PATH` includes the CANN runtime. |
| `msprof not on PATH` | `export PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin:$PATH` |
| `device_count != 4` | `npu-smi info -l` and check drivers; ensure the user is in the `HwHiAiUser` group. |
| `smoke inference failed` | `.om` shape or dtype mismatch — confirm the model was compiled with `--input_format` matching the test input. |

- [ ] Append:

```
<UTC date>  HEALTHCHECK DONE  ---  report at data/results/healthcheck/<run-id>/report.json
```

---

## 5. CANN-specific calibration

The Cube and Vector resource-limit values are prefilled from Huawei's Ascend
Runtime API documentation. The remote machine still needs a one-time sanity
check because Python ACL bindings can differ slightly across CANN builds.

### 5.1 `set_device_res_limit` type enum

The Cube/Vector resource-limit `type` ints from `aclrtDevResLimitType` are:
`ACL_RT_DEV_RES_CUBE_CORE = 0` and `ACL_RT_DEV_RES_VECTOR_CORE = 1`. HBM remains
`null` because this prototype treats HBM as a telemetry/risk dimension rather
than an ACL quota.

- [ ] On the server, sanity-check the Python ACL binding once:

```bash
.venv/bin/python - <<'PY'
import acl
print({name: getattr(acl.rt, name) for name in dir(acl.rt) if "DEV_RES" in name})
PY
```

- [ ] Confirm [configs/hardware/ascend_910b3_4card.yaml](../configs/hardware/ascend_910b3_4card.yaml) keeps:

```yaml
acl:
  set_device_res_limit:
    cube_limit_type: 0
    vector_limit_type: 1
    hbm_limit_type: null
```

If Cube and Vector are set to `null`, `AscendResourceLimiter.apply()` is a no-op
and logs a warning; schedulers still run, but quota adjustments are inert.

### 5.2 msprof field map

The Cube/Vector/HBM column names in your CANN version's `msprof` export.

- [ ] Capture a one-second sample (uses M18 runner):

```bash
bash scripts/remote_ascend/collect_msprof.sh
```

- [ ] Inspect the output CSV under `data/raw/msprof/<run-id>/` and compare against the two fixtures:
  - [tests/fixtures/sample_msprof/cann83_minimal.csv](../tests/fixtures/sample_msprof/cann83_minimal.csv)
  - [tests/fixtures/sample_msprof/cann83_variant.csv](../tests/fixtures/sample_msprof/cann83_variant.csv)
- [ ] If neither schema matches, add a new entry in [docs/hardware/msprof_data_contract.md](hardware/msprof_data_contract.md) and pass a custom `MsprofSchema` when constructing `MsprofParser`.

- [ ] Append:

```
<UTC date>  CALIBRATION DONE  ---  set_device_res_limit enums filled; msprof schema confirmed
```

---

## 6. Model registry

- [ ] Edit [configs/models/model_registry.example.yaml](../configs/models/model_registry.example.yaml) (or copy to `configs/models/model_registry.yaml` and edit there) with real paths and SLO targets:

```yaml
models:
  - name: resnet152
    om_path: /data/models/resnet152.om
    slo_ms: 30.0
    batch_size: 1
  - name: vgg19
    om_path: /data/models/vgg19.om
    slo_ms: 5.0
    batch_size: 1
  - name: vit_b_16
    om_path: /data/models/vit_b_16.om
    slo_ms: 12.0
    batch_size: 1
  - name: bert-base
    om_path: /data/models/bert_base.om
    slo_ms: 15.0
    batch_size: 1
  - name: roberta-base
    om_path: /data/models/roberta_base.om
    slo_ms: 15.0
    batch_size: 1
  - name: distilbert-base
    om_path: /data/models/distilbert_base.om
    slo_ms: 8.0
    batch_size: 1
  - name: gpt2
    om_path: /data/models/gpt2.om
    slo_ms: 20.0
    batch_size: 1
```

`slo_ms` values are the paper's saturation p95 targets — adjust them to whatever you intend to evaluate.

---

## 7. ActiveVC profiling (paper Fig. 2, Fig. 6)

- [ ] Run ActiveVC for every model in the registry (default budget `B=24`):

```bash
MODEL_REGISTRY=configs/models/model_registry.yaml \
HARDWARE_CFG=configs/hardware/ascend_910b3_4card.yaml \
PROFILE_CFG=configs/profiling/activevc_default.yaml \
bash scripts/remote_ascend/run_profile.sh
```

Outputs:

- `data/frontiers/<model>/1.json` — versioned `FrontierSnapshot`
- `data/raw/msprof/<run-id>/` — raw msprof samples
- `data/processed/profiling/<run-id>.csv` — parsed rows

- [ ] Optional budget sweep `B ∈ {8, 16, 24, 32, 48}`:

```bash
for B in 8 16 24 32 48; do
  PROFILE_CFG=<(sed "s/^budget: .*/budget: $B/" configs/profiling/activevc_default.yaml) \
  bash scripts/remote_ascend/run_profile.sh
done
```

- [ ] Append:

```
<UTC date>  PROFILE DONE  ---  7 frontiers under data/frontiers/; budget sweep <range>
```

---

## 8. End-to-end replay (paper Fig. 3, Tab. e2e)

- [ ] Run the default 5-seed × 2400-window replay:

```bash
REPLAY_CFG=configs/experiments/e2e_replay.yaml \
bash scripts/remote_ascend/run_e2e_replay.sh
```

Outputs under `data/results/replay/<run-id>/`:

- `summary.csv` — per-seed goodput and violation
- `events.jsonl` — per-window scheduling events
- `manifest.json` — run metadata

- [ ] Sanity-check the result before believing the numbers:

```bash
python - <<'PY'
import pandas as pd, json, pathlib
runs = sorted(pathlib.Path("data/results/replay").iterdir())
df = pd.read_csv(runs[-1] / "summary.csv")
print(df.describe())
print(json.loads((runs[-1] / "manifest.json").read_text())["metrics"])
PY
```

Expected ordering (paper-aligned): `AdapDomain > Dilu-style > BFD-2D > Exclusive`. If `AdapDomain` ranks lower than `Dilu-style` or BFD-2D, suspect:

1. Wrong frontier — re-run §7 with a higher budget for the affected model.
2. Wrong interference table — see §9.

- [ ] Append:

```
<UTC date>  E2E REPLAY DONE  ---  data/results/replay/<run-id>/summary.csv; goodput_mean=<...>
```

---

## 9. Pairwise interference (paper §6.4, Tab. ablation)

The replay uses an interference table that you collect once per platform.

- [ ] Generate the table:

```bash
.venv/bin/python - <<'PY'
from adapdomain.experiments.collect_interference import PairSpec, collect
# Fill `measure_slowdown` with a real colocation measurement helper, e.g. running two .om models on the same device and comparing solo-vs-pair latency.
table = collect(pairs=[
    PairSpec("resnet152", "bert-base"),
    PairSpec("vgg19", "bert-base"),
    # … 21 pairs total
], measure_slowdown=lambda a, b, bucket: ...)
import json, pathlib
out = pathlib.Path("data/interference/table.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([e.model_dump() for e in table._index], indent=2))
PY
```

(For paper reproduction you can also hand-write the 42 directed entries from the original CSV into `data/interference/table.csv` and load it via the CSV path.)

---

## 10. Ablations (paper Fig. 7, Tab. ablation)

- [ ] Run the full ablation matrix:

```bash
adapdomain ablate --config configs/experiments/ablation.yaml
```

- [ ] Inspect outputs under `data/results/ablation/<run-id>/`:
  - `summary_full.csv`, `summary_no_ceff.csv`, etc.
  - `frontier_recovery_summary.csv` (ActiveVC vs Random/Binary/BO)

- [ ] Append:

```
<UTC date>  ABLATION DONE  ---  full + 5 variants + 4 frontier baselines summarised
```

---

## 11. Controller overhead (paper Fig. 8, Tab. overhead)

- [ ] Measure planning latency for the five replica sizes:

```bash
.venv/bin/python - <<'PY'
from adapdomain.experiments.overhead import measure
for n in (10, 100, 400, 800, 1000):
    print(n, measure(lambda n=n: None, size=n, repeats=20))
PY
```

Replace the lambda with an actual planner invocation if you want the real microbench; the helper above is a structural placeholder.

- [ ] Append:

```
<UTC date>  OVERHEAD DONE  ---  planning p95 at 10/100/400/800/1000 replicas
```

---

## 12. Paper bundle

- [ ] One command to (re)generate every figure and table referenced by the paper:

```bash
bash scripts/remote_ascend/run_paper_bundle.sh
```

Outputs:

- `data/figures/fig3_e2e.png`
- `data/figures/fig8_runtime_overhead.png`
- `data/tables/tab_e2e.md`
- `data/tables/tab_overhead.tex`

(Add more figures/tables to [src/adapdomain/cli/commands/plot.py](../src/adapdomain/cli/commands/plot.py) as you reproduce the rest of fig2/4/5/6/7 and tab\_model\_profiles / tab\_protocol / tab\_default\_params.)

- [ ] Append:

```
<UTC date>  PAPER BUNDLE DONE  ---  data/figures/ + data/tables/ refreshed
```

---

## 13. Troubleshooting reference

| Problem | First place to look |
| --- | --- |
| Test failures after sync | `pytest -q tests/unit/<module>/<file>.py -k <name>` then read failing assertion |
| ACL error code | [src/adapdomain/backends/ascend/acl_runtime.py](../src/adapdomain/backends/ascend/acl_runtime.py) `AclRuntime.check_ret`, [src/adapdomain/core/errors.py](../src/adapdomain/core/errors.py) |
| msprof empty / missing fields | Adjust `MsprofSchema`; see [docs/hardware/msprof_data_contract.md](hardware/msprof_data_contract.md) |
| Resource-limit no-op warning | Confirm [configs/hardware/ascend_910b3_4card.yaml](../configs/hardware/ascend_910b3_4card.yaml) still has Cube `0` and Vector `1`; the limiter is inert only if both are set to `null` |
| Replay goodput unexpectedly low | Inspect `events.jsonl` for `rejected` reasons; verify the interference table loaded |
| Planner times out > 100ms at 400 replicas | Confirm Cube/Vector capacities in [configs/hardware/ascend_910b3_4card.yaml](../configs/hardware/ascend_910b3_4card.yaml) and lower `control_budget_ms` only if you accept the trade-off |

---

## 14. After every session

1. Update [docs/development/continuation_checklist.md](development/continuation_checklist.md) with one append-only line per significant action (`<UTC date>  <EVENT>  <DETAILS>`).
2. Flip the relevant Status in [docs/development/module_plan.md](development/module_plan.md) (`pending → in_progress → done`).
3. If you initialize git in the project (currently it is not a repo), commit at the end of the session — the design spec and plan stand as the source of truth.

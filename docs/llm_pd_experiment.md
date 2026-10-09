# LLM Prefill/Decode Experiment

This experiment is the runnable environment for the LLM reviewer request. It
compares four deployment policies for LLaMA-7B-style prefill/decode serving on a
four-card Ascend NPU node.

The experiment targets the cloud-serving bottleneck raised by the reviewer:
LLM serving mixes Cube-heavy prefill work with decode work that can become
relatively Vector-heavy under low batch size, activation-heavy kernels, sampling,
and KV-Cache pressure. AdapDomain evaluates whether colocating prefill and
decode replicas with phase-aware Cube/Vector partitions improves goodput over
whole-card serving, static PD colocation, and PD-separated pools.

## Policies

- `no_partition`: whole-card LLM replicas. This is the no-core-partition baseline.
- `static_pd_colocation`: prefill and decode share each card with fixed equal
  partitions. This is the static PD mixed-colocation baseline.
- `pd_separate`: cards are split into prefill-only and decode-only pools. This
  represents PD separation.
- `pd_dynamic_partition`: each card has phase-aware partitions derived from the
  prefill/decode resource profile. This is the AdapDomain-style dynamic mixed
  deployment.

## Local smoke run

The local run uses deterministic LLaMA-7B-like profiles. These values are for
environment validation only and must not be reported as paper measurements.

```bash
pip install -e .[dev]
adapdomain llm-pd --out data/results/llm_pd_smoke
```

Outputs:

- `data/results/llm_pd_smoke/llm_pd_summary.csv`
- `data/results/llm_pd_smoke/llm_pd_summary.md`

## Measured CSV mode

Once Ascend measurements are ready, pass the measured phase profile and trace:

```bash
adapdomain llm-pd \
  --profiles-csv /path/to/llm_phase_profiles.csv \
  --trace-csv /path/to/llm_phase_trace.csv \
  --out data/results/llm_pd_measured
```

Required profile CSV columns:

```text
phase,cube_util,vector_util,hbm_gb,p95_ms,slo_ms,batch_size,sampling_policy
```

Optional profile CSV columns:

```text
hbm_capacity_gb,kv_cache_tokens
```

Required trace CSV columns:

```text
phase,arrival_ms,count
```

Use `phase=prefill` or `phase=decode`. The same trace should be used for all
policies so that goodput and SLO violation are comparable.

## Ascend server quick start

Run these commands from the repository root on the NPU server:

```bash
export CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/remote_ascend/setup_env.sh
bash scripts/remote_ascend/run_healthcheck.sh
adapdomain llm-pd --out data/results/llm_pd_smoke
```

The hardware config uses Huawei's documented Runtime API resource-limit enum:
Cube core `0` and Vector core `1`. Confirm the Python ACL binding once on a new
CANN build:

```bash
.venv/bin/python - <<'PY'
import acl
print({name: getattr(acl.rt, name) for name in dir(acl.rt) if "DEV_RES" in name})
PY
```

Then collect measured prefill/decode profiles and rerun:

```bash
adapdomain llm-pd \
  --profiles-csv data/raw/llm/llama7b_phase_profiles.csv \
  --trace-csv data/raw/llm/llama7b_phase_trace.csv \
  --out data/results/llm_pd_measured
```

The measured profile should separate prefill and decode rows. Prefill rows
should be collected from prompt-processing runs, while decode rows should use
low-batch token generation with the sampling policy and KV-Cache size reported
in the CSV. The resulting `llm_pd_summary.csv` is the table source for the paper.

## Paper claim rule

Only measured CSV mode should be used for paper numbers. The smoke run exists to
test the scheduling harness, output format, and plotting pipeline before the NPU
server finishes profiling.

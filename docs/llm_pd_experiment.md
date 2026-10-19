# LLM Prefill/Decode Experiment

## Execution boundary

`llm-pd` is an offline scheduling model. Supplying measured phase profiles
does not make its output a live hardware measurement. It uses modeled service
times and a phase-arrival trace, so its `goodput_qps` is not the paper's
request-level goodput. The official runtime source is vendored at
`third_party/vllm-ascend`; deploy and validate it on the Ascend host
following upstream's release-specific instructions.

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
phase,cube_util,vector_util,hbm_bw_util,p95_ms,slo_ms,batch_size,sampling_policy
```

Optional profile CSV columns:

```text
precision,prompt_tokens,kv_cache_tokens,kv_cache_gb
```

Required trace CSV columns:

```text
phase,arrival_ms,count
```

Use `phase=prefill` or `phase=decode`. The same trace should be used for all
policies so that goodput and SLO violation are comparable.
`hbm_bw_util` is the measured HBM-bandwidth share, not HBM memory in GB.
The loader averages repetitions of one operating point per phase; it rejects
mixed batch/prompt/KV settings and mixed precisions unless a precision is
selected with `--precision`. Keep one phase setting per precision-specific CSV.

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
in the CSV. The resulting `llm_pd_summary.csv` is an offline model output,
not the table source for a live-serving result.

## Live request-level evaluation

Export one request per row from the vLLM-Ascend serving run. Required columns:

```text
request_id,policy,trace,seed,arrival_ms,first_token_ms,completion_ms,output_tokens,status
```

All timestamps use one monotonic millisecond clock. `arrival_ms` is when the
request is offered; `first_token_ms` is when its first output token is observed;
`completion_ms` is when its final output token is observed, not a later logging
or connection-close event. `output_tokens` counts generated tokens. Failed or
timed-out requests have `status` other than `ok` and may leave timing fields
empty. The log must include every offered request, including failures.

For each completed request, TTFT is `first_token_ms - arrival_ms` and mean ITL
is `(completion_ms - first_token_ms) / (output_tokens - 1)`. A one-token
response has mean ITL zero. Goodput counts requests that meet *both* TTFT
`<= 500 ms` and mean ITL `<= 40 ms`, divided by the offered-arrival window;
the violation rate counts all other requests over all offered requests. The
same offered-arrival duration and trace must be used for policy comparisons.

```bash
adapdomain llm-measure data/raw/llm/requests.csv \
  --duration-s 60 \
  --out data/results/llm_request_summary.csv
```

This command summarizes policy/trace/seed rows. It does not launch vLLM,
apply ACL partitions, or claim that an offline model run produced hardware
measurements. Retain the run configuration and model/precision identifiers
with each raw log.

## Paper claim rule

Only request-level logs from an actual server run should be used for the paper's
live goodput and violation numbers. The smoke run tests the offline scheduling
harness and output format.

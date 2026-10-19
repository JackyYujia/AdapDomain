# AdapDomain

Resource-shape-aware node-local inference scheduling for Ascend 910B3 NPUs.

See `docs/superpowers/specs/2026-06-26-adapdomain-implementation-design.md` for the design.

Quick start:

```bash
pip install -e .[dev]
adapdomain --help
```

Ascend NPU quick start:

```bash
export CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/remote_ascend/setup_env.sh
bash scripts/remote_ascend/run_healthcheck.sh
adapdomain llm-pd --out data/results/llm_pd_smoke
```

The 910B3 hardware config already maps `acl.rt.set_device_res_limit` domains to
Huawei's documented values: Cube core `0`, Vector core `1`, and no HBM quota.
Use the smoke LLM PD run only to verify the offline harness. The paper's live
goodput requires request-level logs collected on the NPU server.

The official vLLM-Ascend v0.23.0.post1 source is vendored directly under
`third_party/vllm-ascend` so a normal clone includes its files. See
`third_party/VLLM_ASCEND_VENDOR.md` for the pinned commit, license, and its
separate `catlass` build dependency. Install vLLM-Ascend using its
release-specific upstream instructions on the Ascend host; this repository
does not install CANN or bundle model weights. Validate the installed
vLLM/CANN/torch-npu stack and the selected FP16/W8A8/W4A16 kernels there
before collecting results.

For measured request logs, `adapdomain llm-measure` reports SLO-satisfying
goodput using both TTFT and mean inter-token latency. The older `llm-pd`
command is an offline scheduling model, not live vLLM serving. See
`docs/llm_pd_experiment.md` for the request-log contract and command.

The planner accepts an optional `TenantShape.profile_key` for a specific
model/version, precision, phase, batch, and SLO configuration. When supplied,
frontier and interference lookups use that key, preventing a W8A8 decode
profile from being substituted for an FP16 prefill profile. Existing two-domain
model inputs can omit the key and retain their model-name lookup.

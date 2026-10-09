# AdapDomain

Resource-shape-aware multi-tenant inference scheduling for Ascend 910B3 NPUs.

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
Use the smoke LLM PD run only to verify the harness. Paper numbers should come
from measured prefill/decode CSVs collected on the NPU server.

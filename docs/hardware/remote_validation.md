# Remote 910B3 validation

Run the scripts in order on the remote Linux NPU server:

1. `bash scripts/remote_ascend/setup_env.sh`
2. `bash scripts/remote_ascend/run_healthcheck.sh`  (optionally `SMOKE_OM=/path/to/resnet152.om`)
3. `bash scripts/remote_ascend/collect_msprof.sh`
4. `bash scripts/remote_ascend/run_profile.sh`
5. `bash scripts/remote_ascend/run_e2e_replay.sh`
6. `bash scripts/remote_ascend/run_paper_bundle.sh`

Each step must exit with code 0. Outputs land under `data/`:

- `data/results/healthcheck/<run-id>/report.json`
- `data/raw/msprof/<run-id>/`
- `data/frontiers/<model>/<version>.json`
- `data/results/replay/<run-id>/summary.csv`
- `data/results/ablation/<run-id>/`
- `data/figures/`, `data/tables/`

Append a line per script run to `docs/development/continuation_checklist.md`,
including the run id and a one-line outcome.

## Resource-limit enum sanity check

The Ascend Runtime API documents the resource-limit type enum used by
`acl.rt.set_device_res_limit` as `ACL_RT_DEV_RES_CUBE_CORE = 0` and
`ACL_RT_DEV_RES_VECTOR_CORE = 1`. These values are already filled in
`configs/hardware/ascend_910b3_4card.yaml`.

On a new CANN build, confirm the Python binding exposes compatible constants:

```bash
.venv/bin/python - <<'PY'
import acl
print({name: getattr(acl.rt, name) for name in dir(acl.rt) if "DEV_RES" in name})
PY
```

If the Python binding does not expose named constants, keep the YAML numeric
values and validate the call path with `bash scripts/remote_ascend/run_healthcheck.sh`
before running profiling workloads.

#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
adapdomain profile --registry configs/models/model_registry.example.yaml --hardware configs/hardware/ascend_910b3_4card.yaml --profile configs/profiling/activevc_default.yaml
adapdomain replay --config configs/experiments/e2e_replay.yaml
adapdomain ablate --config configs/experiments/ablation.yaml
adapdomain plot paper

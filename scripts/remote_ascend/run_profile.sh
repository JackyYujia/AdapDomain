#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
REGISTRY="${MODEL_REGISTRY:-configs/models/model_registry.example.yaml}"
HARDWARE="${HARDWARE_CFG:-configs/hardware/ascend_910b3_4card.yaml}"
PROFCFG="${PROFILE_CFG:-configs/profiling/activevc_default.yaml}"
adapdomain profile --registry "$REGISTRY" --hardware "$HARDWARE" --profile "$PROFCFG"

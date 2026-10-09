#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
CONFIG="${REPLAY_CFG:-configs/experiments/e2e_replay.yaml}"
adapdomain replay --config "$CONFIG"

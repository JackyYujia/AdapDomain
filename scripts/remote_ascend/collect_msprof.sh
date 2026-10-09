#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
RUN_ID="${RUN_ID:-msprof-$(date -u +%Y%m%dT%H%M%SZ)}"
APP=("python" "-c" "print('placeholder; replace with model command')")
msprof --output "data/raw/msprof/${RUN_ID}" --application "${APP[*]}" --aic-metrics PipeUtilization,ArithmeticUtilization,HBM --duration 30
echo "ok: msprof output at data/raw/msprof/${RUN_ID}"

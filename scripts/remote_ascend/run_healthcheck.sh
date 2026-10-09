#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
SMOKE_OM="${SMOKE_OM:-}"
if [ -n "$SMOKE_OM" ]; then
  adapdomain verify-remote --smoke-om "$SMOKE_OM"
else
  adapdomain verify-remote
fi

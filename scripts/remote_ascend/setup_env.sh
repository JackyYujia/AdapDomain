#!/usr/bin/env bash
set -euo pipefail

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ ! -f "$CANN_ENV" ]; then
  echo "CANN env script not found at $CANN_ENV" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CANN_ENV"

VENV="${VENV:-.venv}"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip
pip install -e .[dev]

echo "ok: env ready"

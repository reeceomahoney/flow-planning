#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -f "$SCRIPT_DIR/.venv/.ready" ]]; then
    bash "$SCRIPT_DIR/setup.sh"
fi

exec "$PYTHON" "$SCRIPT_DIR/calibrate_cameras.py" "$@"

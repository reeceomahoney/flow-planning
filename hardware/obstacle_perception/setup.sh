#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$SCRIPT_DIR/.venv"

mkdir -p "$SCRIPT_DIR/calibration" "$SCRIPT_DIR/runs"

"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "Install python3-venv, then rerun this script"
    exit 1
fi

PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "$PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
    "$PYTHON" -m pip install torch torchvision
fi

"$PYTHON" -m pip install \
    "accelerate>=1.1" \
    "numpy>=1.26" \
    "opencv-python>=4.10" \
    "pillow>=10" \
    "pyrealsense2>=2.55" \
    "rerun-sdk>=0.26,<0.29" \
    "transformers>=4.57"

"$PYTHON" "$SCRIPT_DIR/preflight.py" \
    --download-model \
    --skip-camera \
    --skip-display

touch "$VENV_DIR/.ready"

echo "Setup complete"
echo "Run: $SCRIPT_DIR/track_obstacle.sh"

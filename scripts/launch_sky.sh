#!/usr/bin/env bash
set -euo pipefail

# usage: launch_sky.sh [cluster_id] — exec on existing cluster, else launch a new one
if [[ $# -gt 0 ]]; then
  sky exec "$1" configs/sky.yaml --env HF_TOKEN --env WANDB_API_KEY "${@:2}"
else
  sky launch configs/sky.yaml -c flow -y --env HF_TOKEN --env WANDB_API_KEY
fi

#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Override these for a different Sirius allocation.
WORKERS="${WORKERS:-8}"
GPUS="${GPUS:-0,1}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}bidirectional-contract"

if [[ ! -f bidirectional-contract/artifacts/recovery_windows.npz ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" pixi run python -m \
    bidirectional_contract.recovery_data
fi

if [[ ! -d bidirectional-contract/artifacts/recovery_policy/checkpoints/step_010000 ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" pixi run python -m \
    bidirectional_contract.train_recovery
fi

pixi run python bidirectional-contract/benchmark_fixed_scene.py \
  --method base --episodes 50 --workers "${WORKERS}" --gpus "${GPUS}" \
  --output bidirectional-contract/results/reproduction/base_50

pixi run python bidirectional-contract/benchmark_fixed_scene.py \
  --method aegis --episodes 50 --workers "${WORKERS}" --gpus "${GPUS}" \
  --safety-margin 0 \
  --output bidirectional-contract/results/reproduction/aegis_released_50

pixi run python bidirectional-contract/benchmark_fixed_scene.py \
  --method aegis --episodes 25 --workers "${WORKERS}" --gpus "${GPUS}" \
  --safety-margin 0.02 \
  --output bidirectional-contract/results/reproduction/aegis_2cm_25

pixi run python bidirectional-contract/benchmark_fixed_scene.py \
  --method ours --episodes 50 --workers "${WORKERS}" --gpus "${GPUS}" \
  --max-frames 300 \
  --recovery-checkpoint \
    bidirectional-contract/artifacts/recovery_policy/checkpoints/step_010000 \
  --output bidirectional-contract/results/reproduction/ours_untuned_50

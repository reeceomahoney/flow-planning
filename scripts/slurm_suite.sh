#!/usr/bin/env bash
# usage: scripts/slurm_suite.sh SUITE "0 1 2 3" [LEVELS]
set -e
SUITE=$1
TASKS=${2:-"0 1 2 3"}
LEVELS=${3:-"I II"}
for T in $TASKS; do
  pixi run slurm run --cluster civo --name "${SUITE#safelibero_}_$T" --command \
    "export PATH=\$HOME/.pixi/bin:\$PATH; cd /users/kebl6123/flow-planning; LEVELS='$LEVELS' bash scripts/pipeline_libero.sh $SUITE $T 2>&1 | grep -vE 'it/s\]|Warning|WARNING|macro'"
done

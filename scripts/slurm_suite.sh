#!/usr/bin/env bash
# usage: scripts/slurm_suite.sh SUITE "0 1 2 3" [LEVELS]
set -e
SUITE=$1
TASKS=${2:-"0 1 2 3"}
LEVELS=${3:-"I II"}
for T in $TASKS; do
  pixi run slurm run --cluster civo --name "${SUITE#safelibero_}_$T" --command \
    "export PATH=\$HOME/.pixi/bin:\$PATH PYTHONUNBUFFERED=1; cd /users/kebl6123/flow-planning; cp scripts/pipeline_libero.sh /tmp/pipe_\$SLURM_JOB_ID.sh; TRAIN=${TRAIN:-1} EPISODES=${EPISODES:-10} TAG=${TAG:-aug} COPIES=${COPIES:-6} NCOND=${NCOND:-8} AUG_ARGS='${AUG_ARGS:-}' EVAL_ARGS='${EVAL_ARGS:-}' LEVELS='$LEVELS' bash /tmp/pipe_\$SLURM_JOB_ID.sh $SUITE $T 2>&1 | grep --line-buffered -vE 'it/s\]|Warning|WARNING|macro'"
done

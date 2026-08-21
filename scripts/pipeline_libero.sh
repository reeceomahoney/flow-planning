#!/usr/bin/env bash
set -e

SUITE=${1:-safelibero_goal}
TASK=${2:-0}
REPO=reece-omahoney/${SUITE}_${TASK}
ENV="--env.type libero --env.suite $SUITE --env.task_id $TASK"
CACHE=~/.cache/huggingface/lerobot/$REPO

[ -d $CACHE ] || pixi run python scripts/convert_libero.py $ENV --repo_id $REPO

[ "${TRAIN:-1}" = 0 ] || pixi run python scripts/train.py $ENV --repo_id $REPO \
  --num_iters 20000 --eval_episodes 50

EVAL="pixi run python scripts/eval.py $ENV --repo_id $REPO --episodes 10"

echo "=== free space ==="
$EVAL --cond zero 2>&1 | grep -v "it/s"

for L in I II; do
  echo "=== SafeLIBERO level $L ==="
  $EVAL --env.obstacle true --env.level $L --cond sample --n_cond 8 2>&1 | grep -v "it/s"
done

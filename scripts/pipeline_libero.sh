#!/usr/bin/env bash
set -e

SUITE=${1:-safelibero_goal}
TASK=${2:-0}
REPO=reece-omahoney/${SUITE}_${TASK}_v2
ENV="--env.type libero --env.suite $SUITE --env.task_id $TASK"
CACHE=~/.cache/huggingface/lerobot/$REPO

[ -d $CACHE ] || pixi run python scripts/convert_libero.py $ENV --repo_id $REPO

COND="sample"
if [ "${AUG:-1}" = 1 ]; then
  [ -d ${CACHE}_bend ] || pixi run python scripts/augment.py $ENV \
    --src_repo $REPO --dst_repo ${REPO}_bend --bend_margin 0.3
  REPO=${REPO}_bend
  COND="search"
fi

[ "${TRAIN:-1}" = 0 ] || pixi run python scripts/train.py $ENV --repo_id $REPO \
  --num_iters 20000 --eval_episodes 50

EVAL="pixi run python scripts/eval.py $ENV --repo_id $REPO --episodes 10"

if [ "${VIDEO:-0}" = 1 ]; then
  V="$EVAL --env.world_count 1 --env.render true --episodes 5"
  $V --cond zero --video vid_free.mp4 2>&1 | grep -v "it/s"
  for L in I II; do
    $V --env.obstacle true --env.level $L --cond $COND --n_cond 8 \
      --video vid_level_$L.mp4 2>&1 | grep -v "it/s"
  done
  exit 0
fi

echo "=== free space ==="
$EVAL --cond zero 2>&1 | grep -v "it/s"

for L in I II; do
  echo "=== SafeLIBERO level $L ==="
  $EVAL --env.obstacle true --env.level $L --cond $COND --n_cond 8 2>&1 | grep -v "it/s"
done

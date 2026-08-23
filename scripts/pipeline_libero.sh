#!/usr/bin/env bash
set -e

SUITE=${1:-safelibero_goal}
TASK=${2:-0}
LEVELS=${LEVELS:-"I II"}
LEVEL=${LEVELS%% *}
REPO=reece-omahoney/${SUITE}_${TASK}_aug
[ "$SUITE $TASK" = "safelibero_goal 3" ] && REPO=${REPO}_$LEVEL
ENV="--env.type libero --env.suite $SUITE --env.task_id $TASK --env.level $LEVEL"
CACHE=~/.cache/huggingface/lerobot/$REPO
RUN=outputs/libero/${REPO##*/}

pixi run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; D('$REPO')" \
  2>/dev/null || (rm -rf $CACHE && pixi run python scripts/augment.py $ENV \
  --dst_repo $REPO --bend_margin 0.3 --copies ${COPIES:-6})

[ "${TRAIN:-1}" = 0 ] || pixi run python scripts/train.py $ENV --repo_id $REPO \
  --num_iters ${ITERS:-20000} --eval_episodes 20 --run_dir $RUN

EVAL="pixi run python scripts/eval.py $ENV --repo_id $REPO --episodes ${EPISODES:-10} --checkpoint $RUN"

echo "=== free space ==="
$EVAL --cond zero 2>&1 | grep --line-buffered -v "it/s"

for L in $LEVELS; do
  echo "=== SafeLIBERO level $L ==="
  $EVAL --env.obstacle true --env.level $L --cond search --n_cond 8 2>&1 | grep --line-buffered -v "it/s"
done

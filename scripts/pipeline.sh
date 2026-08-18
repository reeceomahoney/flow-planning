#!/usr/bin/env bash
set -e

CACHE=~/.cache/huggingface/lerobot/reece-omahoney

[ -d $CACHE/franka-src ] || pixi run python scripts/record.py --env.type franka \
  --env.world_count 256 --episodes 768 --repo_id reece-omahoney/franka-src

rm -rf $CACHE/franka
pixi run python scripts/augment.py \
  --src_repo reece-omahoney/franka-src --dst_repo reece-omahoney/franka

pixi run python scripts/train.py \
  --env.type franka --repo_id reece-omahoney/franka

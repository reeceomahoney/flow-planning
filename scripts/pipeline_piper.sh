#!/usr/bin/env bash
set -e

SRC=reece-omahoney/pick-and-place
REPO=${SRC}-aug
RUN=outputs/piper/${REPO##*/}

rm -rf ~/.cache/huggingface/lerobot/$REPO
pixi run python scripts/augment.py --env.type piper \
  --src_repo $SRC --dst_repo $REPO --copies ${COPIES:-2} --bend_max 0.5 ${AUG_ARGS:-}

pixi run python scripts/train.py --env.type piper --repo_id $REPO \
  --num_iters ${ITERS:-75000} --eval_every 0 --run_dir $RUN

pixi run hf upload reece-omahoney/piper-pick-and-place-aug $RUN

#!/usr/bin/env bash
set -euo pipefail

# DAPO + Multi-Objective (7B, local paths, no envs)
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"

# Ensure labels exist (generate 500 random if missing)
if [ ! -f "./data/labels_multiobj.json" ]; then
  echo "[train] ./data/labels_multiobj.json not found. Generating 500 random labels..."
  python ./multiobj_labelgen.py --random 500 --out ./data/labels_multiobj.json
fi

python ./main_dapo.py \
  model.pretrained_model_name_or_path="$MODEL_ID" \
  custom_reward_function.path=./my_multiobj_reward_fixed.py \
  custom_reward_function.name=compute_score \
  reward_model.enable=false \
  algorithm.filter_groups.enable=true \
  algorithm.filter_groups.metric=seq_final_reward \
  actor.loss_agg_mode=token-mean \
  algorithm.clip_higher.enable=true \
  algorithm.clip_higher.ratio_high=0.4 \
  algorithm.clip_higher.ratio_low=0.0 \
  reward_model.overlong_shaping.enable=true \
  +logging.project="dapo-multiobj" \
  +logging.run_name="7b-$(date +%y%m%d_%H%M%S)"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess, sys, shlex, os

# auto-create labels if missing
if not os.path.exists("./data/labels_multiobj.json"):
    print("[run] ./data/labels_multiobj.json not found. Generating 500 random labels...")
    subprocess.check_call([sys.executable, "./multiobj_labelgen.py", "--random", "500", "--out", "./data/labels_multiobj.json"])

CMD = """
python ./main_dapo.py
  model.pretrained_model_name_or_path=Qwen/Qwen2.5-7B-Instruct
  custom_reward_function.path=./my_multiobj_reward_fixed.py
  custom_reward_function.name=compute_score
  reward_model.enable=false
  algorithm.filter_groups.enable=true
  algorithm.filter_groups.metric=seq_final_reward
  actor.loss_agg_mode=token-mean
  algorithm.clip_higher.enable=true
  algorithm.clip_higher.ratio_high=0.4
  algorithm.clip_higher.ratio_low=0.0
  reward_model.overlong_shaping.enable=true
  +logging.project=dapo-multiobj
  +logging.run_name=7b-run
""".strip()

print("[run_dapo_multiobj_7b.py] Launching:", CMD)
args = shlex.split(CMD)
sys.exit(subprocess.call(args))

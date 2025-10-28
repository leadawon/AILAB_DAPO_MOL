
# DAPO + Multi-Objective (7B, Local Paths, No Envs)

## Files
- `my_multiobj_reward_fixed.py` — Reward plug-in, reads `./data/labels_multiobj.json`
- `multiobj_labelgen.py` — Generate labels (defaults: 500 random → `./data/labels_multiobj.json`)
- `train_dapo_multiobj_7b.sh` — Run training (Hydra overrides inside)
- `run_dapo_multiobj_7b.py` — Single-command Python runner (no args)

## Steps
1) Put your repo here (same folder as `main_dapo.py`). Ensure a `./data` folder exists.
2) (Optional) Generate labels yourself:
```bash
python ./multiobj_labelgen.py --random 500 --out ./data/labels_multiobj.json
```
3) Train (pick one):
```bash
bash ./train_dapo_multiobj_7b.sh
# or
python ./run_dapo_multiobj_7b.py
```
Notes:
- If labels are missing, both scripts will auto-generate 500 random labels into `./data/labels_multiobj.json`.
- DAPO core knobs are inside the scripts (`clip_higher`, `token-mean`, `filter_groups` on seq_final_reward, and `overlong_shaping`).
- If any Hydra key mismatches your recipe version, rename the keys in the script to the closest equivalents from your config.

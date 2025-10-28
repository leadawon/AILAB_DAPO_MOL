#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_dapo.py
------------
Lightweight launcher for DAPO-style training that accepts Hydra-like CLI overrides
without requiring Hydra. It expects a local `dapo_ray_trainer.py` providing:

    RayDAPOTrainer(config: dict, reward_fn)

and a reward plugin module exposing:

    compute_score(data_source, solution_str, ground_truth, extra_info) -> dict

Usage examples:
    python ./main_dapo.py
    python ./main_dapo.py model.pretrained_model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
    python ./main_dapo.py custom_reward_function.path=./my_multiobj_reward_fixed.py

Note:
- Do NOT set GPU env vars inside this file. If needed, prefer shell:
  CUDA_VISIBLE_DEVICES=0 python ./main_dapo.py ...
"""
import sys, os, importlib.util, traceback, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" 
from typing import Any, Dict, List

# ---------- Default config (extended) ----------
DEFAULT_CFG = {
    "model": {
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "custom_reward_function": {
        "path": "./my_multiobj_reward_fixed.py",
        "name": "compute_score",
    },
    "reward_model": {
        "enable": False,
        "overlong_shaping": {
            "enable": True,
            # optional shaping knobs (can override via CLI)
            # "max_tokens": 1024,
            # "penalty": 0.0,
        },
    },
    "algorithm": {
        "filter_groups": {
            "enable": True,
            "metric": "seq_final_reward",
            "group_size": 4,
        },
        "clip_higher": {
            "enable": True,
            "ratio_high": 0.4,
            "ratio_low": 0.0,
        },
    },
    "actor": {
        "loss_agg_mode": "token-mean",
    },
    "ppo": {
        "batch_size": 1,
        "mini_batch_size": 1,
        "learning_rate": 5e-6,
    },
    "sampling": {
        "max_new_tokens": 32,
        "temperature": 1.0,
        "top_p": 0.7,
        "use_chat_template": True,   # Instruct 모델 기본 ON
        "context_max_len": 128,
    },
    "data": {
        "train_path": "./data/train.jsonl",
    },
    "train": {
        "total_steps": 1000,
        "save_path": "./checkpoints/dapo-multiobj-7b",
    },
    "logging": {
        "project": "dapo-multiobj",
        "run_name": "7b-local",
    },
}

def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _parse_cli_overrides(argv: List[str]) -> Dict[str, Any]:
    cfg = {}
    for tok in argv:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        vv: Any = v
        if v.lower() in ("true", "false"):
            vv = (v.lower() == "true")
        else:
            try:
                if "." in v:
                    vv = float(v)
                else:
                    vv = int(v)
            except Exception:
                try:
                    vv = json.loads(v)  # list/dict literal 허용
                except Exception:
                    vv = v
        _deep_set(cfg, k, vv)
    return cfg

def _merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out

def _load_reward_fn(module_path: str, fn_name: str):
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Reward module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("custom_reward_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, fn_name):
        raise AttributeError(f"Reward module {module_path} has no function '{fn_name}'")
    return getattr(mod, fn_name)

def _load_trainer():
    # Expect local file dapo_ray_trainer.py with RayDAPOTrainer
    try:
        import importlib
        trainer_mod = importlib.import_module("dapo_ray_trainer")
        if not hasattr(trainer_mod, "RayDAPOTrainer"):
            raise AttributeError("dapo_ray_trainer.py does not define RayDAPOTrainer")
        return trainer_mod.RayDAPOTrainer
    except Exception as e:
        raise ImportError("Cannot import RayDAPOTrainer from dapo_ray_trainer.py in current directory") from e

def main(argv: List[str]) -> int:
    # 1) parse overrides
    overrides = _parse_cli_overrides(argv)
    cfg = _merge(DEFAULT_CFG, overrides)

    # 2) load reward function
    reward_path = cfg["custom_reward_function"]["path"]
    reward_name = cfg["custom_reward_function"]["name"]
    compute_score = _load_reward_fn(reward_path, reward_name)

    # 3) import trainer
    RayDAPOTrainer = _load_trainer()

    # 4) launch training
    print("[main_dapo] Config:")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    trainer = RayDAPOTrainer(config=cfg, reward_fn=compute_score)
    trainer.fit()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        print("[main_dapo] ERROR:", e)
        traceback.print_exc()
        sys.exit(1)

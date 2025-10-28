#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dapo_ray_trainer.py
-------------------
Lightweight DAPO-style RL (approx) using TRL PPO for small models (1.5B~7B).

- 포함한 DAPO 근사 아이디어:
  * Dynamic Sampling / Group Filtering (보상이 전부 동일한 그룹 드랍)
  * Token-level PG (TRL 내부 token-mean)
  * Overlong Shaping (응답 과도 길이 패널티)
  * Clip-Higher 근사: PPO cliprange를 상대적으로 크게 (비대칭은 미포함)

Dataset (jsonl, per line):
  {"qid": "...", "prompt": "...", "ground_truth": "..."}

주의: 단일 노드 간이 구현입니다. 대규모/분산은 VERL 레시피로 전환 권장.
"""

import os, json, random
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class RayDAPOTrainer:
    def __init__(self, config: Dict[str, Any], reward_fn):
        self.cfg = config
        self.reward_fn = reward_fn

        # ------- Model / Sampling config -------
        model_cfg = self.cfg.get("model", {}) or {}
        model_id = model_cfg.get("pretrained_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        force_float32 = bool(model_cfg.get("force_float32", False))  # fp32 강제 옵션(안정성↑)

        sampling = self.cfg.get("sampling", {}) or {}
        self.max_new_tokens    = int(sampling.get("max_new_tokens", 64))
        self.use_chat_template = bool(sampling.get("use_chat_template", True))
        self.context_max_len   = int(sampling.get("context_max_len", 512))

        # ------- PPO config (최소 인자만) -------
        ppo_cfg = PPOConfig(
            batch_size=int(self.cfg.get("ppo", {}).get("batch_size", 1)),
            mini_batch_size=int(self.cfg.get("ppo", {}).get("mini_batch_size", 1)),
            learning_rate=float(self.cfg.get("ppo", {}).get("learning_rate", 5e-6)),
            cliprange=float(self.cfg.get("algorithm", {}).get("clip_higher", {}).get("ratio_high", 0.4)),
        )

        print(f"[RayDAPOTrainer] Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # dtype / device
        use_cuda = torch.cuda.is_available()
        if force_float32:
            dtype = torch.float32
        else:
            dtype = torch.float16 if use_cuda else torch.float32

        # 단일 프로세스 모델 병렬 자동 샤딩
        device_map = "auto" if use_cuda else None
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

        # 메모리 완화 & use_cache 충돌 방지
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            if hasattr(self.model, "pretrained_model") and hasattr(self.model.pretrained_model.config, "use_cache"):
                self.model.pretrained_model.config.use_cache = False
            elif hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
        except Exception:
            pass

        # PPOTrainer
        try:
            self.ppo_trainer = PPOTrainer(
                ppo_cfg,
                self.model,
                None,
                self.tokenizer,
            )
        except TypeError:
            self.ppo_trainer = PPOTrainer(ppo_cfg, self.model)
            if not hasattr(self.ppo_trainer, "tokenizer") or self.ppo_trainer.tokenizer is None:
                self.ppo_trainer.tokenizer = self.tokenizer

        # ------- Generation kwargs: 항상 그리디 -------
        # do_sample=False, num_beams=1 → Greedy
        self.gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # ------- DAPO-ish knobs -------
        algo = self.cfg.get("algorithm", {}) or {}
        self.group_filter_enable = bool(algo.get("filter_groups", {}).get("enable", True))
        self.group_filter_metric = str(algo.get("filter_groups", {}).get("metric", "seq_final_reward"))
        self.group_size = int(algo.get("filter_groups", {}).get("group_size", 4))

        over = self.cfg.get("reward_model", {}).get("overlong_shaping", {}) or {}
        self.overlong_enable  = bool(over.get("enable", True))
        self.overlong_max     = int(over.get("max_tokens", 1024))
        self.overlong_penalty = float(over.get("penalty", 0.0))

        # ------- Data / Train -------
        data_path = self.cfg.get("data", {}).get("train_path", "./data/train.jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training dataset not found: {data_path}")
        self.dataset = read_jsonl(data_path)

        self.total_steps = int(self.cfg.get("train", {}).get("total_steps", 1000))
        self.save_path = self.cfg.get("train", {}).get("save_path", "./checkpoints/dapo-multiobj")
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def _drop_all_identical_groups(indices: List[int], rewards: List[float], group_size: int) -> List[int]:
        kept = []
        for g in chunks(indices, group_size):
            group_rewards = [rewards[i] for i in g]
            if len(set([round(r, 6) for r in group_rewards])) == 1:
                continue
            kept.extend(g)
        return kept if kept else indices

    def _generate_one(self, qt: torch.Tensor) -> str:
        """
        단일 쿼리 greedy 생성.
        """
        with torch.no_grad():
            # TRL PPOTrainer.generate는 query_tensor 단일만 보장적으로 받도록 사용
            out = self.ppo_trainer.generate(query_tensor=qt, **self.gen_kwargs)

        # out 타입 정규화 → 텍스트
        if isinstance(out, torch.Tensor):
            if out.dim() == 2:
                out = out[0]
            ids = out.detach().cpu().tolist()
            return self.tokenizer.decode(ids, skip_special_tokens=True)

        if isinstance(out, list):
            first = out[0]
            if isinstance(first, torch.Tensor):
                ids = first.detach().cpu().tolist()
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            if isinstance(first, (list, int)):
                ids = first if isinstance(first, list) else out
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            return str(out)

        if isinstance(out, dict):
            seq = out.get("sequences") or out.get("generated_tokens") or out.get("output_ids")
            if isinstance(seq, torch.Tensor):
                if seq.dim() == 2:
                    seq = seq[0]
                ids = seq.detach().cpu().tolist()
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            if isinstance(seq, list):
                s0 = seq[0]
                if isinstance(s0, torch.Tensor):
                    ids = s0.detach().cpu().tolist()
                elif isinstance(s0, list):
                    ids = s0
                else:
                    return str(out)
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            return str(out)

        if isinstance(out, str):
            return out
        return str(out)

    def fit(self):
        print("[RayDAPOTrainer] Starting training...")
        step = 0
        ds = self.dataset
        random.shuffle(ds)

        # 장치
        if hasattr(self.ppo_trainer, "accelerator") and hasattr(self.ppo_trainer.accelerator, "device"):
            dev = self.ppo_trainer.accelerator.device
        else:
            dev = next(self.model.parameters()).device

        while step < self.total_steps:
            bs = getattr(self.ppo_trainer.config, "batch_size", 1)
            start = (step * self.group_size) % max(1, len(ds))
            batch = ds[start : start + bs]
            if not batch:
                random.shuffle(ds)
                continue

            queries = [ex["prompt"] for ex in batch]
            qids    = [ex.get("qid") for ex in batch]
            gts     = [ex.get("ground_truth", "") for ex in batch]

            # --- Instruct 템플릿 (padding=False: 패딩 없이 토크나이즈) ---
            if self.use_chat_template:
                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for q in queries
                ]
            else:
                prompts = queries

            # 각 샘플 개별 토크나이즈 (pad 없음 → 확률 NaN 방지에 유리)
            query_tensors = []
            for p in prompts:
                ids = self.tokenizer(
                    p,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.context_max_len,
                ).input_ids.squeeze(0).to(dev)
                query_tensors.append(ids)

            # --- Greedy 생성(순차) ---
            responses = [self._generate_one(qt) for qt in query_tensors]

            # --- 보상 ---
            rewards = []
            for resp, gt, qid, prompt in zip(responses, gts, qids, queries):
                out = self.reward_fn("train", resp, gt, {"qid": qid, "prompt": prompt})
                r = float(out.get("score", 0.0))
                if self.overlong_enable and self.overlong_penalty > 0.0 and len(resp) > self.overlong_max:
                    overflow = len(resp) - self.overlong_max
                    r -= self.overlong_penalty * (overflow / self.overlong_max)
                rewards.append(r)

            # --- 그룹 필터 ---
            indices = list(range(len(queries)))
            if self.group_filter_enable:
                indices = self._drop_all_identical_groups(indices, rewards, group_size=self.group_size)
            if not indices:
                step += 1
                continue

            # === PPOTrainer.step 입력: 텐서 리스트 ===
            queries_kept_ids = [query_tensors[i].to(dev) for i in indices]
            responses_kept_ids = [
                self.tokenizer(
                    responses[i],
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max(8, self.max_new_tokens + 8),
                ).input_ids.squeeze(0).to(dev)
                for i in indices
            ]
            scores_kept_tensors = [torch.tensor(rewards[i], dtype=torch.float32, device=dev) for i in indices]

            # --- PPO 업데이트 ---
            stats = self.ppo_trainer.step(queries_kept_ids, responses_kept_ids, scores_kept_tensors)
            step += 1

            if step % 10 == 0:
                mean_r = float(torch.stack(scores_kept_tensors).mean().item())
                print(f"[step {step}] reward(mean)={mean_r:.4f} | kl={stats.get('ppo/kl', 0):.4f}")

            if step % 200 == 0:
                save_dir = os.path.join(self.save_path, f"step_{step}")
                os.makedirs(save_dir, exist_ok=True)
                self.ppo_trainer.save_pretrained(save_dir)
                print(f"[save] {save_dir}")

        self.ppo_trainer.save_pretrained(self.save_path)
        print(f"[done] saved to {self.save_path}")

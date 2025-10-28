# DAPO + Multi-Objective RL (Lightweight TRL/PPO Approximation)

> **Status:** Runnable prototype on single node with **greedy decoding** and a **custom multi-objective reward plugin**. Distributed VERL features and fully decoupled-clip dynamics are **not included** in this lightweight trainer.

---

## 1) Overview

**Goal.** Fine-tune an instruct LLM (e.g., Qwen2.5-1.5B/7B-Instruct) with **DAPO-style RL** while using a **multi-objective reward** (e.g., correctness, clarity, detail, creativity). The scalar reward is produced by your Python plugin (or hard-coded JSON labels).

**Approach.** We built a compact trainer around **TRL’s `PPOTrainer`** and approximated several DAPO ideas:

- **Dynamic Sampling / Group Filtering (approx):** drop degenerate groups whose rewards are all identical.
- **Token-level PG aggregation:** via TRL’s value-head PPO (token-mean).
- **Overlong shaping:** optional penalty for overly long responses.
- **Instruct chat template:** use the tokenizer’s chat template for Qwen-Instruct prompts.
- **Greedy decoding only:** `do_sample=False`, `num_beams=1` to avoid CUDA sampling asserts.

**Out of scope vs. full DAPO/VERL (confirmed missing):** VERL Ray pipeline, true **Decoupled Clip** (asymmetric high/low clipping), and full multi-candidate dynamic sampling.  

**Empirical effects (current run):** End-to-end training loop executes with logging and periodic saves. Quantitative benchmark gains (e.g., AIME-2024) = **Unknown**. Long-horizon stability at large batch sizes/mixed precision = **Unverified**.

---

## 2) Expected Repository Layout

```
.
├─ main_dapo.py                     # Launcher (Hydra-like CLI overrides; no Hydra)
├─ dapo_ray_trainer.py              # Lightweight DAPO-ish PPO trainer (greedy only)
├─ my_multiobj_reward_fixed.py      # Your multi-objective reward (compute_score)
├─ data/
│  └─ train.jsonl                   # {"qid": "...", "prompt": "...", "ground_truth": "..."} per line
├─ train_dapo_multiobj_7b.sh        # Example run script (no env vars)
└─ checkpoints/                     # Saved models
```

**`data/train.jsonl` line example**
```json
{"qid": "aime24_001", "prompt": "Solve ... (math problem) ...", "ground_truth": "540"}
```

---

## 3) How to Run (Greedy Only)

Minimal example (adjust paths/model as needed):
```bash
python ./main_dapo.py   model.pretrained_model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct   model.force_float32=true   sampling.max_new_tokens=16 sampling.context_max_len=64   ppo.batch_size=1 ppo.mini_batch_size=1   data.train_path=./data/train.jsonl   train.total_steps=50
```

- `model.force_float32=true` can improve stability on small GPUs (set `false` once stable).
- If multiple GPUs are visible, `device_map="auto"` **shards the model** across GPUs (**model parallel**, not data parallel).
- To select GPUs: `CUDA_VISIBLE_DEVICES=0,1 python ...` (no code change needed).

---

## 4) Multi-Objective Reward (Plugin & Worked Examples)

Implement `compute_score` in `my_multiobj_reward_fixed.py`:

```python
def compute_score(split: str, solution_str: str, ground_truth: str, info: dict) -> dict:
    """
    Return a scalar reward for PPO and (optionally) a breakdown dict.
    Implement any scalarization you want (weighted sum, etc.).
    """
    # --- toy heuristics (replace with your own logic or JSON lookups) ---
    correctness = 1.0 if ground_truth and ground_truth.strip() in solution_str else 0.0
    clarity     = min(1.0, solution_str.count("\n") / 8.0 + 0.2)  # crude: more structure → clearer, capped at 1.0
    detail      = min(1.0, len(solution_str) / 1200.0)            # crude: longer → more detail, capped at 1.0
    creativity  = 0.5 if ("alternative" in solution_str.lower() or "another approach" in solution_str.lower()) else 0.2

    # weighted sum (edit weights as you like)
    w = {"correctness": 0.5, "clarity": 0.25, "detail": 0.15, "creativity": 0.10}
    score = (
        w["correctness"] * correctness +
        w["clarity"]     * clarity +
        w["detail"]      * detail +
        w["creativity"]  * creativity
    )

    return {
        "score": float(score),
        "breakdown": {
            "correctness": float(correctness),
            "clarity": float(clarity),
            "detail": float(detail),
            "creativity": float(creativity),
        }
    }
```

### Example A — correct & well-structured
- **Inputs (hypothetical):**
  - `ground_truth = "540"`
  - `solution_str` contains `"Answer: 540"` and 10 newline-separated steps, one short “Alternative approach” section, ~900 chars.
- **Intermediate measures (toy):**  
  `correctness=1.0`, `clarity=min(1, 10/8+0.2)=1.0`, `detail=min(1, 900/1200)=0.75`, `creativity=0.5`.
- **Weights:** `0.5/0.25/0.15/0.10`.
- **Reward:**  
  `score = 0.5*1.0 + 0.25*1.0 + 0.15*0.75 + 0.10*0.5`  
  `= 0.5 + 0.25 + 0.1125 + 0.05 = 0.9125`.

### Example B — incorrect but very clear and detailed
- **Inputs:** wrong final answer; 12 well-formatted steps; ~1300 chars; no “alternative approach”.
- **Intermediates:** `correctness=0.0`, `clarity=min(1, 12/8+0.2)=1.0`, `detail=min(1, 1300/1200)=1.0`, `creativity=0.2`.
- **Reward:**  
  `score = 0.5*0.0 + 0.25*1.0 + 0.15*1.0 + 0.10*0.2`  
  `= 0 + 0.25 + 0.15 + 0.02 = 0.42`.

> **Note (confirmed):** The trainer expects a **scalar** reward; your multi-objective logic can be any scalarization (weighted sum, Chebyshev, etc.).  
> **Unknown:** Which scalarization best correlates with downstream exam accuracy remains **unverified** here.

---

## 5) Code Highlights

**`main_dapo.py`**
- Hydra-like CLI overrides (e.g., `a.b.c=value`), but no Hydra dependency.
- Loads your reward plugin (`compute_score`) and instantiates the trainer.
- Prints resolved config then runs `trainer.fit()`.

**`dapo_ray_trainer.py`**
- **Tokenizer/Model**
  - `AutoTokenizer.from_pretrained(model_id)` and `AutoModelForCausalLMWithValueHead.from_pretrained(...)`.
  - `device_map="auto"` → **model parallel** across visible GPUs.
  - `eval()`, optional `gradient_checkpointing_enable()`, and `use_cache=False` to avoid conflicts.
- **Generation: greedy only**
  - `do_sample=False`, `num_beams=1`.
  - Qwen-Instruct **chat template** via `apply_chat_template([...], add_generation_prompt=True)`.
  - **Per-sample tokenization** (`padding=False`) to sidestep padding-related sampling issues.
  - Generate **one sample at a time** (`_generate_one`) to avoid TRL API ambiguity.
- **PPO update**
  - TRL expects **lists of tensors**: queries, responses, scores.
  - Responses are **re-tokenized** before `ppo_trainer.step(...)`.
  - **Group filter** drops all-equal-reward groups.
  - Periodic `save_pretrained(...)`.

---

## 6) Process & Results (What we verified)

- **Verified (sure):**
  - End-to-end loop runs: dataset → prompt templating → greedy generation → multi-objective scalar reward → PPO step → logging/saving.
  - Model parallel via `device_map="auto"` works when multiple GPUs are visible.
  - Errors previously seen with multinomial sampling are eliminated by **greedy decoding**.

- **Unverified / Unknown:**
  - Absolute performance improvement on math leaderboards (e.g., AIME-2024) — **Unknown**.
  - Long-run stability at larger batch sizes and mixed precision — **Unknown**.
  - Exact parity with DAPO’s **Decoupled Clip** dynamics — **Not implemented**.

---

## 7) Troubleshooting

- **CUDA device-side assert (was due to sampling):**  
  Keep **greedy** (`do_sample=False`, `num_beams=1`). Optionally set `model.force_float32=true` for stability.

- **OOM:**  
  Reduce `sampling.context_max_len`, `sampling.max_new_tokens`, and `ppo.batch_size`. Keep gradient checkpointing enabled; limit visible GPUs.

- **“Elements in queries/scores must be tensors”:**  
  Ensure lists of **tensors** are passed to `ppo_trainer.step`. The provided code already does this.

- **Multi-GPU:**  
  Current trainer uses **model parallel (sharding)** via `device_map="auto"`.  
  For **data parallel**, migrate to 🤗 Accelerate/DDP or VERL.

---

## 8) Limitations vs. Full DAPO & Migration Path

- **Missing (confirmed):** True **Decoupled Clip** (asymmetric high/low), multi-candidate dynamic sampling, and VERL/Ray distributed orchestration.  
- **Recommended path:**  
  1) Validate your multi-objective reward and loop here on small models.  
  2) Port the reward logic to **VERL’s DAPO recipe** (hook your scalarization).  
  3) Re-enable decoupled clip and multi-candidate sampling within VERL to scale.

---

---

# DAPO + 다목적 RL (경량 TRL/PPO 근사)

> **상태:** 단일 노드에서 **그리디 디코딩**과 **다목적 보상 플러그인**으로 동작하는 프로토타입. VERL 분산 기능과 완전한 decoupled-clip 동작은 **미포함**.

---

## 1) 개요

**목표.** Instruct LLM(Qwen2.5-1.5B/7B-Instruct 등)에 **DAPO-스타일 RL**을 적용하되, **정확성·명확성·자세함·창의성** 등 **다목적 보상**을 하나의 스칼라로 합성하여 학습.

**접근.** **TRL `PPOTrainer`** 기반 경량 트레이너에 DAPO 아이디어를 일부 근사:

- **Dynamic Sampling/Group Filtering 근사:** 보상이 전부 같은 그룹은 드랍.
- **토큰 단위 PG:** TRL의 value-head PPO(token-mean).
- **Overlong shaping:** 응답 과도 길이에 패널티(옵션).
- **Instruct chat template:** Qwen-Instruct용 템플릿 적용.
- **샘플링 금지:** `do_sample=False`, `num_beams=1`로 **그리디**만 사용.

**정식 DAPO/VERL 대비 (확인됨):** VERL Ray 파이프라인, 진짜 **Decoupled Clip**, 다중 후보 기반 Dynamic Sampling은 **미포함**.  

**경험적 상태:** 엔드투엔드 파이프라인은 동작 확인. 벤치마크 정량 개선(AIME-2024 등) = **알 수 없음**. 대배치/혼합정밀 장시간 안정성 = **미확인**.

---

## 2) 디렉터리 구조(예시)

```
.
├─ main_dapo.py
├─ dapo_ray_trainer.py
├─ my_multiobj_reward_fixed.py
├─ data/
│  └─ train.jsonl
├─ train_dapo_multiobj_7b.sh
└─ checkpoints/
```

**`data/train.jsonl` 한 줄 예시**
```json
{"qid": "aime24_001", "prompt": "Solve ... (math problem) ...", "ground_truth": "540"}
```

---

## 3) 실행 방법(그리디)

```bash
python ./main_dapo.py   model.pretrained_model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct   model.force_float32=true   sampling.max_new_tokens=16 sampling.context_max_len=64   ppo.batch_size=1 ppo.mini_batch_size=1   data.train_path=./data/train.jsonl   train.total_steps=50
```

- `model.force_float32=true`는 작은 GPU에서 안정성↑(안정화 후 false 가능).
- 다중 GPU가 보이면 `device_map="auto"`가 **모델 병렬(샤딩)** 로 분산합니다(데이터 병렬 아님).
- GPU 선택: `CUDA_VISIBLE_DEVICES=0,1 python ...`.

---

## 4) 다목적 보상 (플러그인 & 예시 계산)

`my_multiobj_reward_fixed.py`의 `compute_score`에서 스칼라 보상을 반환하세요:

```python
def compute_score(split: str, solution_str: str, ground_truth: str, info: dict) -> dict:
    # 예시 휴리스틱(임의): 정확성/명확성/자세함/창의성을 가중합
    correctness = 1.0 if ground_truth and ground_truth.strip() in solution_str else 0.0
    clarity     = min(1.0, solution_str.count("\n") / 8.0 + 0.2)
    detail      = min(1.0, len(solution_str) / 1200.0)
    creativity  = 0.5 if ("alternative" in solution_str.lower() or "another approach" in solution_str.lower()) else 0.2

    w = {"correctness": 0.5, "clarity": 0.25, "detail": 0.15, "creativity": 0.10}
    score = (
        w["correctness"] * correctness +
        w["clarity"]     * clarity +
        w["detail"]      * detail +
        w["creativity"]  * creativity
    )

    return {"score": float(score),
            "breakdown": {"correctness": float(correctness),
                          "clarity": float(clarity),
                          "detail": float(detail),
                          "creativity": float(creativity)}}
```

### 예시 A — 정답이면서 구성도 좋음
- **입력(가정):**  
  `ground_truth="540"`, `solution_str`에 `"Answer: 540"` 포함, 단계별 10줄, “Alternative approach” 1개, ~900자.
- **중간값:** `correctness=1.0`, `clarity=min(1, 10/8+0.2)=1.0`, `detail=min(1, 900/1200)=0.75`, `creativity=0.5`.
- **가중치:** `0.5/0.25/0.15/0.10`.
- **보상:**  
  `score = 0.5*1.0 + 0.25*1.0 + 0.15*0.75 + 0.10*0.5 = 0.9125`.

### 예시 B — 오답이지만 매우 명확/자세함
- **입력(가정):** 최종 답 오답, 12줄로 잘 정리, ~1300자, 대안 접근 없음.
- **중간값:** `correctness=0.0`, `clarity=1.0`, `detail=1.0`, `creativity=0.2`.
- **보상:**  
  `score = 0.5*0.0 + 0.25*1.0 + 0.15*1.0 + 0.10*0.2 = 0.42`.

> **메모(확실):** 트레이너는 **스칼라 보상**을 기대합니다. 가중합·Chebyshev 등 어떤 스칼라화도 가능.  
> **불확실:** 어떤 스칼라화가 실제 시험 성능과 가장 상관이 높은지는 **검증되지 않음**.

---

## 5) 코드 핵심

**`main_dapo.py`**
- Hydra-유사 오버라이드 지원(의존성 없음).
- 보상 플러그인 로드 → 트레이너 생성 → `fit()` 실행.

**`dapo_ray_trainer.py`**
- **토크나이저/모델**
  - `device_map="auto"`로 **모델 병렬**(가시 GPU 샤딩).
  - `eval()`, `gradient_checkpointing_enable()`(옵션), `use_cache=False`.
- **생성: 그리디**
  - `do_sample=False`, `num_beams=1`.
  - Qwen-Instruct **chat template** 적용.
  - **샘플별 토크나이즈**(padding=False).
  - TRL API 차이를 피하려 **단일 샘플씩** 생성.
- **PPO 업데이트**
  - 입력은 **텐서 리스트**(queries/responses/scores).
  - 응답을 **재-토크나이즈** 후 `ppo_trainer.step(...)`.
  - 보상 동일 그룹 드랍, 주기 저장.

---

## 6) 과정 & 결과(확인/미확인)

- **확인됨:** 데이터셋 → 템플릿 → 그리디 생성 → 다목적 보상 스칼라화 → PPO 업데이트 → 로깅/세이브가 **엔드투엔드로 동작**. 여러 GPU가 보이면 **모델 병렬** 동작.
- **알 수 없음/미확인:** AIME-2024 등 정량 개선, 대배치/혼합정밀 장시간 안정성, **Decoupled Clip**과의 완전 동등성.

---

## 7) 트러블슈팅

- **CUDA device-side assert(샘플링 기인):**  
  **그리디 유지**. 필요 시 `model.force_float32=true`.
- **OOM:**  
  `sampling.context_max_len`, `sampling.max_new_tokens`, `ppo.batch_size`를 더 낮추세요.
- **“queries/scores must be tensors” 오류:**  
  TRL은 **텐서 리스트**를 요구(본 코드 반영).
- **멀티 GPU:**  
  현재는 **모델 병렬(샤딩)**. **데이터 병렬**은 🤗 Accelerate/DDP 또는 VERL로 이전 필요.

---

## 8) 한계 & 이행 경로

- **미포함(확정):** 진짜 **Decoupled Clip**, 다중 후보 Dynamic Sampling, VERL/Ray 분산 오케스트레이션.  
- **권장 이행:**  
  1) 본 경량 트레이너로 다목적 보상/루프를 소형 모델에서 검증 →  
  2) 보상 로직을 **VERL DAPO 레시피**의 훅에 이식 →  
  3) VERL에서 decoupled clip 및 다중 후보 샘플링을 복원하며 스케일업.

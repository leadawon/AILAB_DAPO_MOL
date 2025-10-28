
"""
my_multiobj_reward_fixed.py  (path-relative, no envs)
----------------------------------------------------
- Expects labels at ./data/labels_multiobj.json (relative to *current working directory*).
- Fixed weights: (0.7, 0.15, 0.10, 0.05) for (correct, clarity, detail, creativity).
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import json, hashlib, os

LABELS_JSON_PATH = os.path.join(".", "data", "labels_multiobj.json")
WEIGHTS = (0.7, 0.15, 0.10, 0.05)

_LABELS_DB: Optional[Dict[str, Dict[str, float]]] = None

def _lazy_load_labels() -> Dict[str, Dict[str, float]]:
    global _LABELS_DB
    if _LABELS_DB is None:
        if not os.path.exists(LABELS_JSON_PATH):
            raise FileNotFoundError(f"Labels JSON not found at {LABELS_JSON_PATH}. "
                                    "Create it or run multiobj_labelgen.py.")
        with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
            _LABELS_DB = json.load(f)
    return _LABELS_DB

def _hash_prompt(text: str) -> str:
    return "hash_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def _select_qid(extra_info: Optional[Dict[str, Any]]) -> Optional[str]:
    if not extra_info:
        return None
    for k in ("qid", "problem_id", "id", "sample_id"):
        v = extra_info.get(k)
        if v is not None and str(v).strip():
            return str(v)
    for k in ("prompt", "question", "raw_prompt", "raw_question"):
        v = extra_info.get(k)
        if isinstance(v, str) and v.strip():
            return _hash_prompt(v.strip())
    return None

def _aggregate(labels: Dict[str, float], weights: Tuple[float, float, float, float]) -> float:
    rc  = float(labels.get("r_correct", 0.0))
    rcl = float(labels.get("r_clarity", 0.0))
    rd  = float(labels.get("r_detail", 0.0))
    rcr = float(labels.get("r_creativity", 0.0))
    w1, w2, w3, w4 = weights
    R = w1*rc + w2*rcl + w3*rd + w4*rcr
    return max(-1.0, min(1.0, R))

def compute_score(data_source: str,
                  solution_str: str,
                  ground_truth: str,
                  extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    labels_db = _lazy_load_labels()
    qid = _select_qid(extra_info)
    labels = labels_db.get(qid or "", None)

    label_missing = labels is None
    if label_missing:
        labels = {"r_correct": 0.0, "r_clarity": 0.0, "r_detail": 0.0, "r_creativity": 0.0}

    score = _aggregate(labels, WEIGHTS)

    return {
        "score": float(score),
        "qid": qid,
        "r_correct": float(labels.get("r_correct", 0.0)),
        "r_clarity": float(labels.get("r_clarity", 0.0)),
        "r_detail": float(labels.get("r_detail", 0.0)),
        "r_creativity": float(labels.get("r_creativity", 0.0)),
        "w_correct": float(WEIGHTS[0]),
        "w_clarity": float(WEIGHTS[1]),
        "w_detail": float(WEIGHTS[2]),
        "w_creativity": float(WEIGHTS[3]),
        "label_missing": bool(label_missing),
        "data_source": data_source,
    }

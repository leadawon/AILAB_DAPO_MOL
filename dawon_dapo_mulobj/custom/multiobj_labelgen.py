
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sys, hashlib, random
from typing import Dict, Any

try:
    import pandas as pd
except Exception:
    pd = None

def hash_str(s: str) -> str:
    return "hash_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def gen_label(rng: random.Random, bernoulli_correct: bool = False) -> Dict[str, float]:
    if bernoulli_correct:
        rc = float(rng.random() < 0.5)
    else:
        rc = rng.random()
    return {
        "r_correct": rc,
        "r_clarity": rng.random(),
        "r_detail": rng.random(),
        "r_creativity": rng.random()
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", type=str, default=None)
    ap.add_argument("--qid-field", type=str, default="qid")
    ap.add_argument("--prompt-field", type=str, default="prompt")
    ap.add_argument("--out", type=str, default="./data/labels_multiobj.json")
    ap.add_argument("--random", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bernoulli-correct", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    labels: Dict[str, Any] = {}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.input_parquet:
        if pd is None:
            print("pandas not installed; cannot read parquet.", file=sys.stderr); sys.exit(2)
        if not os.path.exists(args.input_parquet):
            print(f"Parquet not found: {args.input_parquet}", file=sys.stderr); sys.exit(2)
        df = pd.read_parquet(args.input_parquet)
        cols = set(df.columns)
        for i, row in df.iterrows():
            qid = None
            if args.qid_field in cols and row[args.qid_field] is not None:
                qid = str(row[args.qid_field])
            elif args.prompt_field in cols and isinstance(row[args.prompt_field], str):
                qid = hash_str(row[args.prompt_field])
            else:
                qid = f"row_{i}"
            labels[qid] = gen_label(rng, bernoulli_correct=args.bernoulli_correct)
    else:
        for i in range(args.random):
            qid = f"rand_{i:06d}"
            labels[qid] = gen_label(rng, bernoulli_correct=args.bernoulli_correct)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(labels)} labels to {args.out}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_synthetic_trainset.py
-------------------------
Create a minimal ./data/train.jsonl from ./data/labels_multiobj.json.
Each label key becomes a synthetic prompt so training can run end-to-end immediately.
"""

import os, json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=str, default="./data/labels_multiobj.json")
    ap.add_argument("--out", type=str, default="./data/train.jsonl")
    args = ap.parse_args()

    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for qid in labels.keys():
            prompt = f"Solve the following problem and end with 'Answer:'. (Problem ID: {qid})\n\n[Your solution here]"
            # ground_truth can be empty since our reward is label-based
            out.write(json.dumps({"qid": qid, "prompt": prompt, "ground_truth": ""}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} examples to {args.out}")

if __name__ == "__main__":
    main()

"""Audit PSR direction and definition.

Paper §3.3 defines PSR as the fraction of base samples whose answer-correctness
indicator differs across {V1, V2, V3}, lower is better.
The current src/evaluate.py:compute_psr instead checks whether the *answer
strings* (after normalize_answer) differ across ALL six variants. This is a
broader notion that mixes position effects with conflict and distractor effects,
and it explains why GPT-5.4-medium has PSR=0.894 despite being position-stable.

This script recomputes PSR three ways for every model and writes a comparison
report so we can decide which to publish.

Outputs:
  outputs/metrics/psr_audit.csv
  paper_assets/psr_audit_report.md
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

BASE = ".."
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import load_jsonl, normalize_answer, match_answer  # noqa


def is_correct(pred, inst):
    return match_answer(pred.get("answer", ""), inst["gold_answer"], "general")


def main():
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    pred_dir = os.path.join(BASE, "outputs/parsed_predictions")

    rows = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".jsonl"):
            continue
        model = fname.replace(".jsonl", "")
        preds = list(load_jsonl(os.path.join(pred_dir, fname)))
        by_sample = defaultdict(dict)
        for p in preds:
            iid = p["instance_id"]
            if iid not in insts:
                continue
            inst = insts[iid]
            by_sample[inst["sample_id"]][inst["variant"]] = p

        # --- (a) current implementation: answer strings differ across all 6 variants
        cur_n = 0
        cur_sens = 0
        for sid, m in by_sample.items():
            cur_n += 1
            ans = {normalize_answer(p.get("answer", "")) for p in m.values()}
            if len(ans) > 1:
                cur_sens += 1
        psr_current = cur_sens / cur_n if cur_n else 0

        # --- (b) paper definition: correctness indicator differs across V1/V2/V3
        pap_n = 0
        pap_sens = 0
        for sid, m in by_sample.items():
            req = ("correct_front", "correct_middle", "correct_end")
            if not all(v in m for v in req):
                continue
            pap_n += 1
            inds = {1 if is_correct(m[v], insts[m[v]["instance_id"]]) else 0 for v in req}
            if len(inds) > 1:
                pap_sens += 1
        psr_paper = pap_sens / pap_n if pap_n else 0

        # --- (c) absolute spread: |Acc(V1) - Acc(V3)|, useful sanity check
        v1_acc, v3_acc, n13 = 0, 0, 0
        for sid, m in by_sample.items():
            if "correct_front" in m and "correct_end" in m:
                n13 += 1
                v1_acc += int(is_correct(m["correct_front"], insts[m["correct_front"]["instance_id"]]))
                v3_acc += int(is_correct(m["correct_end"], insts[m["correct_end"]["instance_id"]]))
        spread = abs(v1_acc - v3_acc) / n13 if n13 else 0

        rows.append({
            "model": model,
            "PSR_current_impl": round(psr_current, 4),
            "PSR_paper_def": round(psr_paper, 4),
            "abs_V1_minus_V3": round(spread, 4),
        })

    # Print + save
    print(f"{'model':<32s} {'PSR_current':>12s} {'PSR_paper':>10s} {'|V1-V3|':>10s}")
    print("-" * 70)
    for r in rows:
        print(f"{r['model']:<32s} {r['PSR_current_impl']:>12.4f} "
              f"{r['PSR_paper_def']:>10.4f} {r['abs_V1_minus_V3']:>10.4f}")

    out_csv = os.path.join(BASE, "outputs/metrics/psr_audit.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n[ok] wrote", out_csv)


if __name__ == "__main__":
    main()

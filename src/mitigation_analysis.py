"""Aggregate mitigation results into the paper-ready delta table.

Runs paired bootstrap on Acc / PBR / CAR / EAR / CEU between baseline and
conflict_aware per model.

Inputs:
  outputs/parsed_predictions/<model>.jsonl        (baseline per-instance)
  outputs/mitigation/phase_a__<model>.jsonl       (conflict-aware per-instance)
  data/wrong_claims.jsonl                         (planted claim candidates)

Outputs:
  outputs/mitigation/mitigation_delta.csv         (Δ table)
"""
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

BASE = os.environ.get("RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import is_valid_prediction, load_jsonl, match_answer  # noqa
from evaluate import adopts_wrong_claim, load_wrong_claims  # noqa

OUT = os.path.join(BASE, "outputs/mitigation")
SEED = 42
SAMPLE_N = 300
B_BOOT = 1000

DISPLAY = [
    ("gpt-5.4-medium",            "GPT-5.4-medium",     "#4C78A8"),
    ("deepseek-chat",             "DeepSeek-Chat",      "#72B7B2"),
    ("gpt-5.4-mini",              "GPT-5.4-mini",       "#54A24B"),
    ("claude-haiku-4-5-20251001", "Claude-Haiku-4.5",   "#B279A2"),
    ("claude-sonnet-4-6",         "Claude-Sonnet-4.6",  "#9D755D"),
    ("deepseek-reasoner",         "DeepSeek-Reasoner",  "#4C78A8"),
    ("gemini-2.5-flash",          "Gemini-2.5-Flash",   "#E45756"),
]


def parse_json(text):
    if not text: return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try: return json.loads(t)
    except: pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: return None
    return None


def select_subset(insts):
    by_sample = defaultdict(dict)
    for i in insts:
        by_sample[i["sample_id"]][i["variant"]] = i["instance_id"]
    eligible = [(sid, m["conflict_before_correct"], m["correct_before_conflict"])
                for sid, m in by_sample.items()
                if "conflict_before_correct" in m and "correct_before_conflict" in m]
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    return eligible[:SAMPLE_N]


def per_sample_indicators(records, insts, claims):
    """Return per-sample, per-variant indicators under the final metric definitions."""
    by_iid = {r["instance_id"]: r for r in records}
    out = defaultdict(lambda: {
        "acc": {}, "pbr": {}, "ear": {}, "car": {}, "ceu": {}
    })
    for iid, r in by_iid.items():
        inst = insts.get(iid)
        if not inst: continue
        d = parse_json(r["raw_output"]) or {}
        ans = (d.get("answer") or "")
        sel = d.get("selected_evidence_ids") or []
        hc = d.get("has_conflict")
        correct = match_answer(ans, inst["gold_answer"], "general")
        cor_pos = inst.get("correct_evidence_position")
        sid = inst["sample_id"]
        variant = inst["variant"]
        out[sid]["acc"][variant] = int(correct)

        if not is_valid_prediction(d):
            continue

        if variant in ("conflict_before_correct", "correct_before_conflict"):
            claim = claims.get(sid)
            if claim is not None and claim["status"] == "decided":
                adoption = int(adopts_wrong_claim(d, inst, claim["candidates"]))
                out[sid]["ear"][variant] = adoption
                if variant == "conflict_before_correct":
                    out[sid]["pbr"][variant] = adoption
            if claim is not None and claim.get("has_nonempty_wrong_evidence") is True:
                out[sid]["car"][variant] = int(hc is True and cor_pos in sel)
        if cor_pos:
            out[sid]["ceu"][variant] = int(cor_pos in sel)
    return out


def paired_values(baseline, treatment, key):
    """Average only variants observed under both conditions, then pair by sample."""
    values_b, values_t = [], []
    for sid in sorted(set(baseline) & set(treatment)):
        b = baseline[sid][key]
        t = treatment[sid][key]
        common_variants = sorted(set(b) & set(t))
        if not common_variants:
            continue
        values_b.append(float(np.mean([b[v] for v in common_variants])))
        values_t.append(float(np.mean([t[v] for v in common_variants])))
    return np.asarray(values_b), np.asarray(values_t)


def paired_bootstrap(values_a, values_b, n_boot=B_BOOT):
    """Two-sided paired bootstrap: p-value and percentile CI of (a-b) mean.
    Inputs are aligned arrays (per-sample paired observations).
    Returns (obs, p, ci_lo, ci_hi).
    """
    diffs = values_a - values_b
    obs = diffs.mean()
    rng = np.random.default_rng(SEED)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[k] = diffs[idx].mean()
    centered = boot_means - obs
    p = float((np.abs(centered) >= abs(obs)).mean())
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return obs, p, ci_lo, ci_hi


def main():
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    claims = load_wrong_claims()
    subset = select_subset(insts.values())
    subset_iids = {iv4 for _, iv4, _ in subset} | {iv5 for _, _, iv5 in subset}

    delta_rows = []

    print(f"{'model':<32s} {'metric':<7s} {'base':>7s} {'CA':>7s} {'Δ':>7s} {'p':>8s} {'sig':>4s}")
    print("-" * 80)

    for mid, _, _ in DISPLAY:
        base_path = os.path.join(BASE, f"outputs/parsed_predictions/{mid}.jsonl")
        ca_path = os.path.join(OUT, f"phase_a__{mid}.jsonl")
        if not (os.path.exists(base_path) and os.path.exists(ca_path)):
            print(f"  {mid}: missing files")
            continue

        base_recs = [{"instance_id": p["instance_id"], "raw_output": json.dumps(p)}
                     for p in load_jsonl(base_path) if p["instance_id"] in subset_iids]
        ca_recs = list(load_jsonl(ca_path))

        b = per_sample_indicators(base_recs, insts, claims)
        c = per_sample_indicators(ca_recs, insts, claims)

        for metric in ("acc", "pbr", "car", "ear", "ceu"):
            arr_b, arr_c = paired_values(b, c, metric)
            if not len(arr_b): continue
            mean_b = arr_b.mean(); mean_c = arr_c.mean()
            delta = mean_c - mean_b
            obs, p, ci_lo, ci_hi = paired_bootstrap(arr_c, arr_b)
            sig = "*" if p < 0.05 else ""
            print(f"  {mid:<30s} {metric.upper():<7s} {mean_b:>7.4f} {mean_c:>7.4f} {delta:>+7.4f} {p:>8.4f} {sig:>4s}")
            delta_rows.append({"model": mid, "metric": metric.upper(),
                               "n_pairs": len(arr_b),
                               "baseline": f"{mean_b:.4f}", "conflict_aware": f"{mean_c:.4f}",
                               "delta": f"{delta:+.4f}", "p_value": f"{p:.4f}",
                               "ci_lo": f"{ci_lo:+.4f}", "ci_hi": f"{ci_hi:+.4f}",
                               "significant": "yes" if p < 0.05 else "no"})

    # Save delta CSV
    out_delta = os.path.join(OUT, "mitigation_delta.csv")
    with open(out_delta, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(delta_rows[0].keys()), lineterminator="\n"
        )
        w.writeheader(); w.writerows(delta_rows)
    print(f"\n[ok] {out_delta}")

if __name__ == "__main__":
    main()

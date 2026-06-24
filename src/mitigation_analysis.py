"""Aggregate mitigation results into a paper-ready Δ table and a Figure 7
delta-metric bar chart. Also runs paired bootstrap on Acc / PBR / CAR / EAR / CEU
between baseline and conflict_aware per model.

Inputs:
  outputs/mitigation/mitigation_summary.csv       (already aggregated)
  outputs/parsed_predictions/<model>.jsonl        (baseline per-instance)
  outputs/mitigation/phase_a__<model>.jsonl       (conflict-aware per-instance)

Outputs:
  outputs/mitigation/mitigation_delta.csv         (Δ table)
  outputs/mitigation/mitigation_significance.csv  (paired bootstrap p-values)
  figures/fig7_mitigation_delta.{pdf,png}
"""
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np

BASE = ".."
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import load_jsonl, match_answer  # noqa

OUT = os.path.join(BASE, "outputs/mitigation")
FIG = os.path.join(BASE, "figures")
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


def per_sample_indicators(records, insts):
    """Return per-sample dict: sid -> {acc_v4, acc_v5, pbr_v4, ear, car, ceu}.
    Aggregates across V4 and V5 within one base sample.
    """
    by_iid = {r["instance_id"]: r for r in records}
    out = defaultdict(lambda: {"acc": [], "pbr": None, "ear": [], "car": [], "ceu": []})
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
        out[sid]["acc"].append(int(correct))
        if inst["variant"] == "conflict_before_correct":
            out[sid]["pbr"] = int((not correct) and ans.strip() != "")
        if inst["variant"] in ("conflict_before_correct", "correct_before_conflict"):
            out[sid]["ear"].append(int((not correct) and ans.strip() != ""))
            out[sid]["car"].append(int(hc is True and cor_pos in sel))
        if cor_pos:
            out[sid]["ceu"].append(int(cor_pos in sel))
    return out


def collapse(per_sid_indicators, key):
    """Collapse list-valued indicator into one float per sid (mean)."""
    res = {}
    for sid, d in per_sid_indicators.items():
        v = d[key]
        if isinstance(v, list):
            if v: res[sid] = float(np.mean(v))
        else:
            if v is not None: res[sid] = float(v)
    return res


def paired_bootstrap(values_a, values_b, n_boot=B_BOOT):
    """One-sided paired bootstrap p-value: prob(diff <= 0) under resample.
    Inputs are aligned arrays (per-sample paired observations).
    """
    diffs = values_a - values_b
    obs = diffs.mean()
    rng = np.random.default_rng(SEED)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[k] = diffs[idx].mean()
    if obs >= 0:
        p = float((boot_means <= 0).mean())
    else:
        p = float((boot_means >= 0).mean())
    return obs, p


def main():
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    subset = select_subset(insts.values())
    subset_iids = {iv4 for _, iv4, _ in subset} | {iv5 for _, _, iv5 in subset}
    sids_in_subset = [sid for sid, _, _ in subset]

    delta_rows = []
    sig_rows = []

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

        b = per_sample_indicators(base_recs, insts)
        c = per_sample_indicators(ca_recs, insts)

        for metric in ("acc", "pbr", "car", "ear", "ceu"):
            b_map = collapse(b, metric); c_map = collapse(c, metric)
            common = sorted(set(b_map) & set(c_map))
            if not common: continue
            arr_b = np.array([b_map[s] for s in common])
            arr_c = np.array([c_map[s] for s in common])
            mean_b = arr_b.mean(); mean_c = arr_c.mean()
            delta = mean_c - mean_b
            obs, p = paired_bootstrap(arr_c, arr_b)
            sig = "*" if p < 0.05 else ""
            print(f"  {mid:<30s} {metric.upper():<7s} {mean_b:>7.4f} {mean_c:>7.4f} {delta:>+7.4f} {p:>8.4f} {sig:>4s}")
            delta_rows.append({"model": mid, "metric": metric.upper(),
                               "baseline": f"{mean_b:.4f}", "conflict_aware": f"{mean_c:.4f}",
                               "delta": f"{delta:+.4f}", "p_value": f"{p:.4f}",
                               "significant": "yes" if p < 0.05 else "no"})

    # Save delta CSV
    out_delta = os.path.join(OUT, "mitigation_delta.csv")
    with open(out_delta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        w.writeheader(); w.writerows(delta_rows)
    print(f"\n[ok] {out_delta}")

    # ----- Figure 7: Δ-metric bar chart (rows = models, columns = 4 metrics) -----
    serif_pick = next((n for n in ("Times New Roman","Liberation Serif","DejaVu Serif")
                       if fm.findfont(fm.FontProperties(family=n), fallback_to_default=False)),
                      "DejaVu Serif")
    rcParams.update({
        "font.family":"serif", "font.serif":[serif_pick],
        "axes.titlesize":12, "axes.labelsize":10.5,
        "xtick.labelsize":9.5, "ytick.labelsize":9.5,
        "legend.fontsize":9, "axes.linewidth":0.6, "axes.edgecolor":"#444",
    })

    metrics_to_plot = ["ACC", "CAR", "PBR", "EAR"]
    invert = {"ACC": False, "CAR": False, "PBR": True, "EAR": True}
    deltas = defaultdict(dict)
    sigs = defaultdict(dict)
    for r in delta_rows:
        deltas[r["model"]][r["metric"]] = float(r["delta"])
        sigs[r["model"]][r["metric"]] = (r["significant"] == "yes")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    n_models = len(DISPLAY)
    n_metrics = len(metrics_to_plot)
    w = 0.18
    x = np.arange(n_metrics)
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#B279A2", "#9D755D", "#F2A65A", "#E45756"]

    for i, (mid, label, _) in enumerate(DISPLAY):
        # apply directional sign so "good" delta is positive on the chart
        bars = []
        for met in metrics_to_plot:
            d = deltas.get(mid, {}).get(met, 0.0)
            if invert[met]:
                d = -d  # flip so improvement is positive
            bars.append(d)
        offsets = x + (i - n_models/2 + 0.5) * w
        b = ax.bar(offsets, bars, w, color=colors[i % len(colors)], label=label,
                   edgecolor="white", linewidth=0.4)
        for bi, met in enumerate(metrics_to_plot):
            if sigs.get(mid, {}).get(met, False):
                ax.text(offsets[bi], bars[bi] + (0.01 if bars[bi] >= 0 else -0.025),
                        "*", ha="center", va="bottom" if bars[bi] >= 0 else "top",
                        color="black", fontsize=10)
    ax.axhline(0, color="#666666", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["ΔAcc ↑", "ΔCAR ↑", "ΔPBR ↓ (sign-flipped)", "ΔEAR ↓ (sign-flipped)"])
    ax.set_ylabel("Δ (conflict-aware $-$ baseline) — higher is better")
    ax.set_title("Conflict-aware prompting: per-model improvement over baseline\n"
                 "($\\star$ = paired-bootstrap significant at $p<0.05$, 300 base samples)",
                 pad=10, color="#222")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#B0B0B0", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=4,
              frameon=False, handlelength=1.4, columnspacing=1.2)
    plt.tight_layout()
    pdfp = os.path.join(FIG, "fig7_mitigation_delta.pdf")
    pngp = os.path.join(FIG, "fig7_mitigation_delta.png")
    plt.savefig(pdfp, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(pngp, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close()
    print(f"[ok] {pdfp}")


if __name__ == "__main__":
    main()

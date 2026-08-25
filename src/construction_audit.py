"""Audit released-v1 layout comparability and run a matched-content check."""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from significance import paired_bootstrap_test  # noqa: E402
from utils import load_jsonl, match_answer  # noqa: E402


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = "correct_front"
V3 = "correct_end"
V4 = "conflict_before_correct"
V5 = "correct_before_conflict"


def same_multiset(left, right):
    return Counter(left["evidences"].values()) == Counter(right["evidences"].values())


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    bases = {row["sample_id"]: row for row in load_jsonl(
        os.path.join(BASE, "data", "base_samples_with_wrong.jsonl"))}
    instances = load_jsonl(os.path.join(BASE, "data", "eval_instances.jsonl"))
    by_sample = defaultdict(dict)
    for item in instances:
        by_sample[item["sample_id"]][item["variant"]] = item

    nonempty_wrong = {
        sample_id for sample_id, row in bases.items()
        if str(row.get("wrong_evidence") or "").strip()
    }
    v1_v3_same = {
        sample_id for sample_id, layouts in by_sample.items()
        if same_multiset(layouts[V1], layouts[V3])
    }
    v4_v5_same = {
        sample_id for sample_id, layouts in by_sample.items()
        if same_multiset(layouts[V4], layouts[V5])
    }
    v4_v5_matched = v4_v5_same & nonempty_wrong
    only_key_swap = {
        sample_id for sample_id in v4_v5_matched
        if all(
            by_sample[sample_id][V4]["evidences"][slot]
            == by_sample[sample_id][V5]["evidences"][slot]
            for slot in ("E1", "E3", "E4", "E6")
        )
    }

    audit = [
        {"check": "base_samples", "count": len(bases),
         "interpretation": "all released base samples"},
        {"check": "nonempty_wrong_evidence", "count": len(nonempty_wrong),
         "interpretation": "eligible planted-conflict samples"},
        {"check": "empty_wrong_evidence", "count": len(bases) - len(nonempty_wrong),
         "interpretation": "generation failures; designated slots contain distractors"},
        {"check": "v1_v3_same_evidence_multiset", "count": len(v1_v3_same),
         "interpretation": "same six passage texts, not necessarily same distractor slots"},
        {"check": "v4_v5_same_evidence_multiset", "count": len(v4_v5_same),
         "interpretation": "includes samples without a planted wrong passage"},
        {"check": "v4_v5_matched_nonempty_wrong", "count": len(v4_v5_matched),
         "interpretation": "matched-content sensitivity subset"},
        {"check": "v4_v5_only_key_swap", "count": len(only_key_swap),
         "interpretation": "all other evidence slots identical"},
    ]
    metrics_dir = os.path.join(BASE, "outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    write_csv(os.path.join(metrics_dir, "construction_audit.csv"), audit)

    matched_rows = []
    pred_dir = os.path.join(BASE, "outputs", "parsed_predictions")
    for filename in sorted(os.listdir(pred_dir)):
        if not filename.endswith(".jsonl"):
            continue
        predictions = {row["instance_id"]: row for row in load_jsonl(
            os.path.join(pred_dir, filename))}
        scores_v4 = []
        scores_v5 = []
        for sample_id in sorted(v4_v5_matched):
            i4 = by_sample[sample_id][V4]
            i5 = by_sample[sample_id][V5]
            scores_v4.append(float(match_answer(
                predictions[i4["instance_id"]].get("answer", ""),
                i4["gold_answer"], "general")))
            scores_v5.append(float(match_answer(
                predictions[i5["instance_id"]].get("answer", ""),
                i5["gold_answer"], "general")))
        mean4, ci4, mean5, ci5, diff, ci_diff, p_value = paired_bootstrap_test(
            scores_v4, scores_v5)
        matched_rows.append({
            "model": filename[:-6],
            "n_pairs": len(v4_v5_matched),
            "mean_v4": f"{mean4:.4f}",
            "ci_v4_lo": f"{ci4[0]:.4f}",
            "ci_v4_hi": f"{ci4[1]:.4f}",
            "mean_v5": f"{mean5:.4f}",
            "ci_v5_lo": f"{ci5[0]:.4f}",
            "ci_v5_hi": f"{ci5[1]:.4f}",
            "diff_v4_minus_v5": f"{diff:.4f}",
            "ci_diff_lo": f"{ci_diff[0]:.4f}",
            "ci_diff_hi": f"{ci_diff[1]:.4f}",
            "p_value": f"{p_value:.4f}",
            "significant_at_0_01": "yes" if p_value < 0.01 else "no",
        })
    write_csv(
        os.path.join(metrics_dir, "content_matched_v4_v5.csv"), matched_rows
    )

    for row in audit:
        print(f"{row['check']}: {row['count']} ({row['interpretation']})")
    print("Matched-content V4/V5 results written for", len(matched_rows), "models")


if __name__ == "__main__":
    main()

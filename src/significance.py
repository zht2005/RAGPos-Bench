"""Statistical significance tests: bootstrap CI and paired comparisons."""
import csv
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, match_answer, is_abstention, normalize_answer

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
np.random.seed(42)


def load_data():
    instances = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl'))}
    pred_dir = os.path.join(BASE_DIR, 'outputs', 'parsed_predictions')
    all_preds = {}
    for fname in sorted(os.listdir(pred_dir)):
        if fname.endswith('.jsonl'):
            model = fname.replace('.jsonl', '')
            all_preds[model] = {p["instance_id"]: p for p in load_jsonl(os.path.join(pred_dir, fname))}
    return instances, all_preds


def is_pred_correct(pred, inst):
    return match_answer(pred["answer"], inst["gold_answer"], "general")


def bootstrap_ci(scores, n_boot=1000, ci=0.95):
    scores = np.array(scores, dtype=float)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return np.mean(scores), lo, hi


def paired_bootstrap_test(scores_a, scores_b, n_boot=1000):
    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    obs_diff = np.mean(scores_a) - np.mean(scores_b)
    count = 0
    n = len(scores_a)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        diff = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
        if diff <= 0:
            count += 1
    p_value = count / n_boot
    return obs_diff, p_value


def main():
    instances, all_preds = load_data()
    results = []

    by_sample_variant = defaultdict(lambda: defaultdict(dict))
    for inst_id, inst in instances.items():
        by_sample_variant[inst["sample_id"]][inst["variant"]] = inst_id

    for model_name, preds in all_preds.items():
        v1_scores = []
        v3_scores = []
        v4_scores = []
        for sample_id, variants in by_sample_variant.items():
            for vname, inst_id in variants.items():
                if inst_id not in preds:
                    continue
                inst = instances[inst_id]
                pred = preds[inst_id]
                correct = 1 if is_pred_correct(pred, inst) else 0
                if vname == "correct_front":
                    v1_scores.append(correct)
                elif vname == "correct_end":
                    v3_scores.append(correct)
                elif vname == "conflict_before_correct":
                    v4_scores.append(correct)

        if v1_scores and v3_scores:
            n = min(len(v1_scores), len(v3_scores))
            diff, p = paired_bootstrap_test(v1_scores[:n], v3_scores[:n])
            mean_v1, lo1, hi1 = bootstrap_ci(v1_scores)
            mean_v3, lo3, hi3 = bootstrap_ci(v3_scores)
            results.append({
                "model": model_name, "comparison": "V1_vs_V3",
                "mean_A": f"{mean_v1:.4f}", "CI_A": f"[{lo1:.4f},{hi1:.4f}]",
                "mean_B": f"{mean_v3:.4f}", "CI_B": f"[{lo3:.4f},{hi3:.4f}]",
                "diff": f"{diff:.4f}", "p_value": f"{p:.4f}",
                "significant": "yes" if p < 0.05 else "no"
            })
        if v1_scores and v4_scores:
            n = min(len(v1_scores), len(v4_scores))
            diff, p = paired_bootstrap_test(v1_scores[:n], v4_scores[:n])
            mean_v4, lo4, hi4 = bootstrap_ci(v4_scores)
            results.append({
                "model": model_name, "comparison": "V1_vs_V4",
                "mean_A": f"{mean_v1:.4f}", "CI_A": f"[{lo1:.4f},{hi1:.4f}]",
                "mean_B": f"{mean_v4:.4f}", "CI_B": f"[{lo4:.4f},{hi4:.4f}]",
                "diff": f"{diff:.4f}", "p_value": f"{p:.4f}",
                "significant": "yes" if p < 0.05 else "no"
            })

    out_path = os.path.join(BASE_DIR, 'outputs', 'metrics', 'significance_tests.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if results:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Significance tests saved to {out_path}")


if __name__ == "__main__":
    main()

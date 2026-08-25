"""Released-v1 layout comparisons: paired bootstrap over sample ids.

For each model and each contrast (V1_vs_V3, V1_vs_V4, V4_vs_V5) we pair the
per-sample correctness indicators by sample_id, compute the observed mean paired
difference D, and bootstrap-resample sample_ids (B=10,000, seed 42) to obtain
(a) a two-sided p-value with the bootstrap distribution centred at zero, and
(b) a 95% percentile CI of D. Paired resampling also yields CIs for each arm.

These full-set comparisons are paired by question, but the released v1
generator independently sampled/reordered distractors across layouts. They
must not be interpreted as content-controlled causal effects. Run
``construction_audit.py`` for the matched-content V4/V5 sensitivity analysis.
"""
import csv
import os
import shutil
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, match_answer, VARIANT_CODES

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

SEED = 42
N_BOOT = 10_000
EXPECTED_PAIRS = 2500
ALPHA = 0.01

# Contrasts expressed in paper codes; resolved to descriptive names via VARIANT_CODES.
CONTRASTS = (("V1", "V3"), ("V1", "V4"), ("V4", "V5"))
CODE_TO_NAME = {code: name for name, code in VARIANT_CODES.items()}


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


def per_sample_scores(instances, preds, variant_name):
    """Mean correctness per sample_id for one variant (one instance per sample
    in this benchmark, so the mean is the 0/1 indicator itself)."""
    acc = defaultdict(list)
    for inst_id, inst in instances.items():
        if inst["variant"] != variant_name:
            continue
        pred = preds.get(inst_id)
        if pred is None:
            continue
        acc[inst["sample_id"]].append(1.0 if is_pred_correct(pred, inst) else 0.0)
    return {sid: float(np.mean(v)) for sid, v in acc.items()}


def paired_bootstrap_test(scores_a, scores_b, n_boot=N_BOOT, seed=SEED):
    """Two-sided paired bootstrap test on aligned per-sample scores.

    Returns (mean_a, ci_a, mean_b, ci_b, D, ci_d, p_two_sided) where D is the
    observed mean paired difference and the p-value centres the bootstrap
    distribution at zero (shift by -D) before counting |.| >= |D|.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    n = len(a)
    d = a - b
    D = float(d.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_a = a[idx].mean(axis=1)
    boot_b = b[idx].mean(axis=1)
    boot_d = boot_a - boot_b

    # Centre bootstrap distribution at zero to emulate H0: E[D]=0.
    centered = boot_d - D
    p = float(np.mean(np.abs(centered) >= abs(D)))

    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return (float(a.mean()), ci(boot_a), float(b.mean()), ci(boot_b), D, ci(boot_d), p)


def main():
    instances, all_preds = load_data()
    results = []

    for model_name, preds in all_preds.items():
        # Per-sample score maps for every variant used in any contrast.
        needed_codes = sorted({c for pair in CONTRASTS for c in pair})
        scores = {code: per_sample_scores(instances, preds, CODE_TO_NAME[code])
                  for code in needed_codes}

        for code_a, code_b in CONTRASTS:
            sa, sb = scores[code_a], scores[code_b]
            common = sorted(set(sa) & set(sb))
            if len(common) != EXPECTED_PAIRS:
                print(f"WARNING: {model_name} {code_a}_vs_{code_b}: "
                      f"{len(common)} pairs (expected {EXPECTED_PAIRS})")
            if not common:
                continue
            a = [sa[sid] for sid in common]
            b = [sb[sid] for sid in common]
            mean_a, ci_a, mean_b, ci_b, D, ci_d, p = paired_bootstrap_test(a, b)
            results.append({
                "model": model_name,
                "contrast": f"{code_a}_vs_{code_b}",
                "n_pairs": len(common),
                "mean_a": f"{mean_a:.4f}",
                "ci_a_lo": f"{ci_a[0]:.4f}", "ci_a_hi": f"{ci_a[1]:.4f}",
                "mean_b": f"{mean_b:.4f}",
                "ci_b_lo": f"{ci_b[0]:.4f}", "ci_b_hi": f"{ci_b[1]:.4f}",
                "diff": f"{D:.4f}",
                "ci_diff_lo": f"{ci_d[0]:.4f}", "ci_diff_hi": f"{ci_d[1]:.4f}",
                "p_value": f"{p:.4f}",
                "significant": "yes" if p < ALPHA else "no",
            })
            print(f"{model_name:<32s} {code_a}_vs_{code_b}  n={len(common)}  "
                  f"A={mean_a:.4f}  B={mean_b:.4f}  D={D:+.4f} "
                  f"[{ci_d[0]:+.4f},{ci_d[1]:+.4f}]  p={p:.4f}")

    out_path = os.path.join(BASE_DIR, 'outputs', 'metrics', 'significance_tests.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        backup = os.path.join(os.path.dirname(out_path), 'significance_tests_OLD_BACKUP.csv')
        if not os.path.exists(backup):
            shutil.copy2(out_path, backup)
            print(f"Backed up previous results to {backup}")
    if results:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=list(results[0].keys()), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(results)
    print(f"Significance tests saved to {out_path}")


if __name__ == "__main__":
    main()

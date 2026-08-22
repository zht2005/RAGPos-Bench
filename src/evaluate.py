"""Evaluate parsed predictions and compute all metrics."""
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_jsonl, match_answer, is_abstention, normalize_answer,
                   CONFLICT_VARIANTS, is_valid_prediction)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def load_instances():
    path = os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl')
    instances = load_jsonl(path)
    return {inst["instance_id"]: inst for inst in instances}


def load_all_predictions():
    pred_dir = os.path.join(BASE_DIR, 'outputs', 'parsed_predictions')
    all_preds = {}
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith('.jsonl'):
            continue
        model_name = fname.replace('.jsonl', '')
        all_preds[model_name] = load_jsonl(os.path.join(pred_dir, fname))
    return all_preds


def is_correct(pred, instance):
    return match_answer(pred["answer"], instance["gold_answer"], "general")


def compute_psr(preds_by_sample, instances):
    """Position Sensitivity Rate (paper definition, Section 3.3):
    fraction of base samples whose answer-correctness indicator differs
    across the position-only variants {V1=correct_front, V2=correct_middle,
    V3=correct_end}. Lower is better; a perfectly position-invariant model
    has PSR=0.
    """
    POS_VARIANTS = ("correct_front", "correct_middle", "correct_end")
    sensitive = 0
    total = 0
    for sample_id, preds in preds_by_sample.items():
        by_variant = {}
        for p in preds:
            inst = instances[p["instance_id"]]
            if inst["variant"] in POS_VARIANTS:
                by_variant[inst["variant"]] = p
        if len(by_variant) != len(POS_VARIANTS):
            continue
        total += 1
        indicators = set()
        for v in POS_VARIANTS:
            p = by_variant[v]
            inst = instances[p["instance_id"]]
            indicators.add(1 if is_correct(p, inst) else 0)
        if len(indicators) > 1:
            sensitive += 1
    return sensitive / total if total > 0 else 0


def compute_pbr(preds, instances):
    """Primacy Bias Rate: in conflict-before-correct variant,
    proportion of samples where model adopts the front wrong evidence."""
    relevant = 0
    biased = 0
    for p in preds:
        inst = instances[p["instance_id"]]
        if inst["variant"] != "conflict_before_correct":
            continue
        relevant += 1
        if not is_correct(p, inst):
            biased += 1
    return biased / relevant if relevant > 0 else 0


def compute_car(preds, instances):
    """Conflict Arbitration Rate: in conflict samples,
    proportion where model BOTH identifies conflict AND selects correct answer."""
    conflict_cases = 0
    correctly_resolved = 0
    for p in preds:
        inst = instances[p["instance_id"]]
        if not inst.get("wrong_evidence_position"):
            continue
        conflict_cases += 1
        detected = p.get("has_conflict", False)
        if not detected:
            if any(k in p.get("brief_reason", "").lower() for k in ["conflict", "contradict", "inconsistent"]):
                detected = True
        if detected and is_correct(p, inst):
            correctly_resolved += 1
    return correctly_resolved / conflict_cases if conflict_cases > 0 else 0


def compute_ear(preds, instances):
    relevant = 0
    adopted = 0
    for p in preds:
        inst = instances[p["instance_id"]]
        if not inst.get("wrong_evidence_position"):
            continue
        relevant += 1
        if not is_correct(p, inst):
            adopted += 1
    return adopted / relevant if relevant > 0 else 0


def compute_ceu(preds, instances):
    relevant = 0
    correct_usage = 0
    for p in preds:
        inst = instances[p["instance_id"]]
        correct_pos = inst.get("correct_evidence_position")
        if not correct_pos:
            continue
        relevant += 1
        if correct_pos in p.get("selected_evidence_ids", []):
            correct_usage += 1
    return correct_usage / relevant if relevant > 0 else 0


# --- Fixed (v2) metric definitions --------------------------------------------
# EAR/PBR are re-defined as *adoption* of the planted wrong claim (from
# data/wrong_claims.jsonl), scoped to the conflict variants V4/V5, over valid
# predictions only. CAR/CEU keep their definitions but exclude invalid records
# (API errors, empty/unparseable outputs), matching mitigation_utility.py.

def load_wrong_claims():
    path = os.path.join(BASE_DIR, 'data', 'wrong_claims.jsonl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run src/extract_wrong_claims.py first.")
    return {c["sample_id"]: c for c in load_jsonl(path)}


def adopts_wrong_claim(pred, inst, candidates):
    """True if the (valid, non-gold) answer matches any wrong-claim candidate
    via normalized containment in either direction."""
    if is_correct(pred, inst):
        return False
    ans_n = normalize_answer(pred.get("answer"))
    if not ans_n:
        return False
    for cand in candidates:
        cand_n = normalize_answer(cand)
        if cand_n and (cand_n in ans_n or ans_n in cand_n):
            return True
    return False


def compute_ear_new(preds, instances, claims, variants=CONFLICT_VARIANTS):
    """Evidence Adoption Rate (fixed): fraction of valid predictions on the
    conflict variants whose answer adopts a planted wrong claim. Samples with
    undecidable wrong claims are excluded from numerator and denominator.
    Returns (rate, excluded_n, denom_n)."""
    denom = num = excluded = 0
    for p in preds:
        inst = instances[p["instance_id"]]
        if inst["variant"] not in variants:
            continue
        if not is_valid_prediction(p):
            continue
        claim = claims.get(inst["sample_id"])
        if claim is None or claim["status"] != "decided":
            excluded += 1
            continue
        denom += 1
        if adopts_wrong_claim(p, inst, claim["candidates"]):
            num += 1
    return (num / denom if denom else 0.0), excluded, denom


def compute_pbr_paired(preds, instances, claims):
    """Primacy Bias (fixed, paired): mean over valid pairs of
    [adoption(V4) - adoption(V5)] per sample. A pair is valid when both the
    V4 and V5 predictions are valid and the sample's wrong claim is decided.
    Returns (paired_mean, n_pairs)."""
    v4_name, v5_name = CONFLICT_VARIANTS  # conflict_before_correct, correct_before_conflict
    by_sample = {}
    for p in preds:
        inst = instances[p["instance_id"]]
        if inst["variant"] not in CONFLICT_VARIANTS:
            continue
        if not is_valid_prediction(p):
            continue
        claim = claims.get(inst["sample_id"])
        if claim is None or claim["status"] != "decided":
            continue
        adopt = adopts_wrong_claim(p, inst, claim["candidates"])
        by_sample.setdefault(inst["sample_id"], {})[inst["variant"]] = int(adopt)
    diffs = [v[v4_name] - v[v5_name] for v in by_sample.values()
             if v4_name in v and v5_name in v]
    return (sum(diffs) / len(diffs) if diffs else 0.0), len(diffs)


def compute_car_new(preds, instances, variants=CONFLICT_VARIANTS):
    """Conflict Arbitration Rate (fixed): among valid predictions on the
    conflict variants, fraction where the model flags has_conflict=True AND
    selects the correct evidence position."""
    denom = num = 0
    for p in preds:
        inst = instances[p["instance_id"]]
        if inst["variant"] not in variants:
            continue
        if not is_valid_prediction(p):
            continue
        denom += 1
        if p.get("has_conflict") is True and \
                inst.get("correct_evidence_position") in (p.get("selected_evidence_ids") or []):
            num += 1
    return num / denom if denom else 0.0


def compute_ceu_new(preds, instances):
    """Correct Evidence Usage (fixed): valid predictions only."""
    denom = num = 0
    for p in preds:
        inst = instances[p["instance_id"]]
        correct_pos = inst.get("correct_evidence_position")
        if not correct_pos:
            continue
        if not is_valid_prediction(p):
            continue
        denom += 1
        if correct_pos in (p.get("selected_evidence_ids") or []):
            num += 1
    return num / denom if denom else 0.0


def evaluate_model_v2(model_name, preds, instances, claims):
    acc = sum(1 for p in preds if is_correct(p, instances[p["instance_id"]]))
    acc = acc / len(preds) if preds else 0.0
    ear_new, ear_excl, ear_denom = compute_ear_new(preds, instances, claims)
    pbr_adopt_v4, _, _ = compute_ear_new(
        preds, instances, claims, variants=(CONFLICT_VARIANTS[0],))
    pbr_paired, _n_pairs = compute_pbr_paired(preds, instances, claims)
    return {
        "model": model_name,
        "acc": f"{acc:.4f}",
        "ear_old": f"{compute_ear(preds, instances):.4f}",
        "ear_new": f"{ear_new:.4f}",
        "ear_excluded_n": ear_excl,
        "ear_denom_n": ear_denom,
        "pbr_old": f"{compute_pbr(preds, instances):.4f}",
        "pbr_adoption_v4": f"{pbr_adopt_v4:.4f}",
        "pbr_paired": f"{pbr_paired:.4f}",
        "car_old": f"{compute_car(preds, instances):.4f}",
        "car_new": f"{compute_car_new(preds, instances):.4f}",
        "ceu_old": f"{compute_ceu(preds, instances):.4f}",
        "ceu_new": f"{compute_ceu_new(preds, instances):.4f}",
    }


def main_v2():
    instances = load_instances()
    all_preds = load_all_predictions()
    claims = load_wrong_claims()
    metrics_dir = os.path.join(BASE_DIR, 'outputs', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    rows = [evaluate_model_v2(m, preds, instances, claims)
            for m, preds in all_preds.items()]
    out_path = os.path.join(metrics_dir, 'overall_metrics_v2.csv')
    write_csv(rows, out_path)
    print(f"V2 metrics saved to {out_path}")
    for r in rows:
        print(f"  {r['model']}: acc={r['acc']} "
              f"EAR old={r['ear_old']} new={r['ear_new']} "
              f"(denom={r['ear_denom_n']}, excluded={r['ear_excluded_n']}) "
              f"PBR old={r['pbr_old']} adoptV4={r['pbr_adoption_v4']} paired={r['pbr_paired']} "
              f"CAR old={r['car_old']} new={r['car_new']} "
              f"CEU old={r['ceu_old']} new={r['ceu_new']}")
    return rows


def evaluate_model(model_name, preds, instances):
    metrics = {"model": model_name}
    acc = sum(1 for p in preds if is_correct(p, instances[p["instance_id"]]))
    metrics["accuracy"] = acc / len(preds) if preds else 0

    by_sample = defaultdict(list)
    for p in preds:
        inst = instances[p["instance_id"]]
        by_sample[inst["sample_id"]].append(p)
    metrics["PSR"] = compute_psr(by_sample, instances)
    metrics["PBR"] = compute_pbr(preds, instances)
    metrics["CAR"] = compute_car(preds, instances)
    metrics["EAR"] = compute_ear(preds, instances)
    metrics["CEU"] = compute_ceu(preds, instances)
    return metrics


def evaluate_by_source(model_name, preds, instances):
    by_src = defaultdict(list)
    for p in preds:
        by_src[instances[p["instance_id"]].get("source", "unknown")].append(p)
    rows = []
    for src, src_preds in by_src.items():
        acc = sum(1 for p in src_preds if is_correct(p, instances[p["instance_id"]])) / len(src_preds)
        rows.append({"model": model_name, "source": src, "accuracy": acc, "n": len(src_preds)})
    return rows


def evaluate_by_variant(model_name, preds, instances):
    by_var = defaultdict(list)
    for p in preds:
        by_var[instances[p["instance_id"]]["variant"]].append(p)
    rows = []
    for v, var_preds in sorted(by_var.items()):
        acc = sum(1 for p in var_preds if is_correct(p, instances[p["instance_id"]])) / len(var_preds)
        rows.append({"model": model_name, "variant": v, "accuracy": acc, "n": len(var_preds)})
    return rows


def evaluate_by_position(model_name, preds, instances):
    by_pos = defaultdict(list)
    for p in preds:
        inst = instances[p["instance_id"]]
        pos = inst.get("correct_evidence_position")
        if pos:
            by_pos[pos].append(p)
    rows = []
    for pos, pos_preds in sorted(by_pos.items()):
        acc = sum(1 for p in pos_preds if is_correct(p, instances[p["instance_id"]])) / len(pos_preds)
        rows.append({"model": model_name, "position": pos, "accuracy": acc, "n": len(pos_preds)})
    return rows


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    instances = load_instances()
    all_preds = load_all_predictions()
    metrics_dir = os.path.join(BASE_DIR, 'outputs', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    overall_rows = []
    source_rows = []
    variant_rows = []
    position_rows = []

    for model_name, preds in all_preds.items():
        overall_rows.append(evaluate_model(model_name, preds, instances))
        source_rows.extend(evaluate_by_source(model_name, preds, instances))
        variant_rows.extend(evaluate_by_variant(model_name, preds, instances))
        position_rows.extend(evaluate_by_position(model_name, preds, instances))

    write_csv(overall_rows, os.path.join(metrics_dir, 'overall_metrics.csv'))
    write_csv(source_rows, os.path.join(metrics_dir, 'by_source_metrics.csv'))
    write_csv(variant_rows, os.path.join(metrics_dir, 'by_variant_metrics.csv'))
    write_csv(position_rows, os.path.join(metrics_dir, 'position_metrics.csv'))

    print("Evaluation complete. Metrics saved to outputs/metrics/")
    for row in overall_rows:
        print(f"  {row['model']}: Acc={row['accuracy']:.3f} PSR={row['PSR']:.3f} PBR={row['PBR']:.3f} CAR={row['CAR']:.3f}")


if __name__ == "__main__":
    # Default: v2 metrics only (does NOT touch overall_metrics.csv).
    # Pass --legacy to regenerate the original metric CSVs.
    if "--legacy" in sys.argv:
        main()
    else:
        main_v2()

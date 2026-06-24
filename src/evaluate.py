"""Evaluate parsed predictions and compute all metrics."""
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, match_answer, is_abstention, normalize_answer

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
    main()

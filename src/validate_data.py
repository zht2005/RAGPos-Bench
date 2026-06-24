"""Data validation: check consistency of base samples and eval instances."""
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, normalize_answer

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def main():
    base_path = os.path.join(BASE_DIR, 'data', 'base_samples.jsonl')
    inst_path = os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl')
    bases = load_jsonl(base_path)
    instances = load_jsonl(inst_path)

    report = []
    report.append("# Data Validation Report\n")

    report.append(f"## Summary")
    report.append(f"- Base samples: {len(bases)}")
    report.append(f"- Eval instances: {len(instances)}")

    type_counts = Counter(b["task_type"] for b in bases)
    report.append(f"- Types: {dict(type_counts)}\n")

    errors = []

    for b in bases:
        if not b.get("gold_answer"):
            errors.append(f"{b['sample_id']}: missing gold_answer")
        if b["task_type"] != "insufficient_evidence":
            gold_norm = normalize_answer(b["gold_answer"])
            for i, d in enumerate(b.get("distractor_evidences", [])):
                if gold_norm in normalize_answer(d):
                    errors.append(f"{b['sample_id']}: distractor {i} may contain gold answer")
        if b["task_type"] == "insufficient_evidence":
            if b.get("correct_evidence"):
                errors.append(f"{b['sample_id']}: insufficient type should not have correct_evidence")

    by_sample = defaultdict(list)
    for inst in instances:
        by_sample[inst["sample_id"]].append(inst)

    for sample_id, insts in by_sample.items():
        if len(insts) != 6:
            errors.append(f"{sample_id}: expected 6 variants, got {len(insts)}")
        ev_sets = []
        for inst in insts:
            ev_sets.append(set(inst["evidences"].values()))
        base_set = ev_sets[0]
        for i, s in enumerate(ev_sets[1:], 1):
            if s != base_set:
                pass

    v4_insts = [i for i in instances if i["variant"] == "V4_wrong_front_correct_last"]
    for inst in v4_insts:
        if inst.get("wrong_evidence_position") != "E1":
            errors.append(f"{inst['instance_id']}: V4 wrong_evidence not at E1")
        if inst.get("correct_evidence_position") != "E6":
            errors.append(f"{inst['instance_id']}: V4 correct_evidence not at E6")

    balanced = all(v == 50 for v in type_counts.values())
    report.append("## Checks")
    report.append(f"- Type balance: {'PASS' if balanced else 'WARN'} {dict(type_counts)}")
    report.append(f"- V4 position check: {'PASS' if not any('V4' in e for e in errors) else 'FAIL'}")
    report.append(f"- Total errors: {len(errors)}")
    if errors:
        report.append("\n## Errors")
        for e in errors[:20]:
            report.append(f"- {e}")

    out_path = os.path.join(BASE_DIR, 'outputs', 'data_validation_report.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Validation report -> {out_path} ({len(errors)} errors)")


if __name__ == "__main__":
    main()

"""Validate released-v1 data and report its known construction limitations."""

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from utils import VARIANT_ORDER, load_jsonl  # noqa: E402


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    bases = load_jsonl(os.path.join(BASE_DIR, "data/base_samples_with_wrong.jsonl"))
    instances = load_jsonl(os.path.join(BASE_DIR, "data/eval_instances.jsonl"))
    errors = []

    if len(bases) != 2_500:
        errors.append(f"expected 2,500 base samples, found {len(bases)}")
    if len(instances) != 15_000:
        errors.append(f"expected 15,000 instances, found {len(instances)}")

    base_by_id = {}
    for base in bases:
        sample_id = base.get("sample_id")
        if not sample_id or sample_id in base_by_id:
            errors.append(f"missing or duplicate base sample_id: {sample_id!r}")
            continue
        base_by_id[sample_id] = base
        for field in ("question", "gold_answer", "correct_evidence"):
            if not str(base.get(field) or "").strip():
                errors.append(f"{sample_id}: empty {field}")

    expected_sources = {"hotpotqa": 1_000, "musique": 1_000, "squad": 500}
    source_counts = Counter(base.get("source") for base in bases)
    if source_counts != expected_sources:
        errors.append(f"unexpected source counts: {dict(source_counts)}")

    by_sample = defaultdict(dict)
    instance_ids = set()
    for inst in instances:
        instance_id = inst.get("instance_id")
        sample_id = inst.get("sample_id")
        variant = inst.get("variant")
        if instance_id in instance_ids:
            errors.append(f"duplicate instance_id: {instance_id}")
        instance_ids.add(instance_id)
        if variant in by_sample[sample_id]:
            errors.append(f"{sample_id}: duplicate variant {variant}")
        by_sample[sample_id][variant] = inst

        base = base_by_id.get(sample_id)
        if base is None:
            errors.append(f"{instance_id}: unknown sample_id {sample_id}")
            continue
        for field in ("question", "gold_answer", "source"):
            if inst.get(field) != base.get(field):
                errors.append(f"{instance_id}: {field} differs from base sample")

        evidences = inst.get("evidences") or {}
        correct_pos = inst.get("correct_evidence_position")
        if evidences.get(correct_pos) != base.get("correct_evidence"):
            errors.append(f"{instance_id}: correct evidence position mismatch")
        wrong_pos = inst.get("wrong_evidence_position")
        wrong_evidence = str(base.get("wrong_evidence") or "").strip()
        if wrong_pos and wrong_evidence and evidences.get(wrong_pos) != wrong_evidence:
            errors.append(f"{instance_id}: wrong evidence position mismatch")

    expected_variants = set(VARIANT_ORDER)
    for sample_id, variants in by_sample.items():
        if set(variants) != expected_variants:
            errors.append(f"{sample_id}: variants={sorted(variants)}")

    for sample_id, variants in by_sample.items():
        v4 = variants.get("conflict_before_correct")
        v5 = variants.get("correct_before_conflict")
        if v4 and (v4.get("wrong_evidence_position"), v4.get("correct_evidence_position")) != ("E2", "E5"):
            errors.append(f"{sample_id}: V4 positions are not wrong=E2, correct=E5")
        if v5 and (v5.get("wrong_evidence_position"), v5.get("correct_evidence_position")) != ("E5", "E2"):
            errors.append(f"{sample_id}: V5 positions are not wrong=E5, correct=E2")

    nonempty_wrong = {
        sample_id for sample_id, base in base_by_id.items()
        if str(base.get("wrong_evidence") or "").strip()
    }
    same_multiset_v4_v5 = set()
    only_key_swap_v4_v5 = set()
    for sample_id, variants in by_sample.items():
        v4 = variants.get("conflict_before_correct")
        v5 = variants.get("correct_before_conflict")
        if not v4 or not v5:
            continue
        ev4 = v4["evidences"]
        ev5 = v5["evidences"]
        if Counter(ev4.values()) == Counter(ev5.values()):
            same_multiset_v4_v5.add(sample_id)
        if sample_id in nonempty_wrong and all(
                ev4[slot] == ev5[slot] for slot in ("E1", "E3", "E4", "E6")):
            only_key_swap_v4_v5.add(sample_id)

    if errors:
        print(f"Data validation failed with {len(errors)} errors:")
        for error in errors[:20]:
            print(f"- {error}")
        raise SystemExit(1)

    print("Structural validation passed: 2,500 base samples, 15,000 unique "
          "instances, six variants per sample, and consistent declared positions.")
    print("Released-v1 construction audit:")
    print(f"- non-empty planted wrong evidence: {len(nonempty_wrong)}/2,500")
    print(f"- V4/V5 with the same evidence multiset: "
          f"{len(same_multiset_v4_v5)}/2,500")
    print(f"- V4/V5 differing only at E2/E5 with non-empty wrong evidence: "
          f"{len(only_key_swap_v4_v5)}/2,500")
    print("These are documented release limitations, not validation failures.")


if __name__ == "__main__":
    main()

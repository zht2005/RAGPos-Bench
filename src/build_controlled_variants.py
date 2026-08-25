"""Build corrected content-controlled layouts for a future rerun.

V1--V5 are permutations of one fixed six-passage multiset per base sample.
V4/V5 differ only by swapping the correct and planted-wrong passages at E2/E5.
The script refuses samples without a generated wrong passage instead of
silently filling the designated wrong slot with an ordinary distractor.
For the three released samples with fewer than five distractors, existing
distractors are deterministically cycled so that every layout remains valid.

The released v1 predictions do not correspond to this output. New model
inference is required before reporting results on the controlled layouts.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl  # noqa: E402


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 123


def make_instance(
    sample, variant, slots, correct_pos, wrong_pos, reused_distractor_count
):
    return {
        "instance_id": f"{sample['sample_id']}_{variant}",
        "sample_id": sample["sample_id"],
        "variant": variant,
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        "source": sample.get("source", ""),
        "evidences": {f"E{i + 1}": text for i, text in enumerate(slots)},
        "correct_evidence_position": f"E{correct_pos + 1}",
        "wrong_evidence_position": (
            f"E{wrong_pos + 1}" if wrong_pos is not None else None
        ),
        "construction_version": "controlled-v2",
        "reused_distractor_count": reused_distractor_count,
    }


def build_variants_for_sample(sample):
    correct = str(sample.get("correct_evidence") or "").strip()
    wrong = str(sample.get("wrong_evidence") or "").strip()
    distractors = [str(x).strip() for x in sample.get("distractor_evidences", [])
                   if str(x).strip()]
    if not correct:
        raise ValueError(f"{sample.get('sample_id')}: empty correct evidence")
    if not wrong:
        raise ValueError(f"{sample.get('sample_id')}: empty wrong evidence")
    if not distractors:
        raise ValueError(f"{sample.get('sample_id')}: no non-empty distractors")

    rng = random.Random(f"{SEED}:{sample['sample_id']}")
    rng.shuffle(distractors)
    original_distractor_count = len(distractors)
    distractors = [
        distractors[i % original_distractor_count] for i in range(5)
    ]
    reused_distractor_count = max(0, 5 - original_distractor_count)
    d0, d1, d2, d3, d4 = distractors[:5]

    # V1--V3 keep the wrong passage fixed at E5. Moving the correct passage
    # necessarily swaps it with one distractor, while preserving the multiset.
    layouts = [
        ("correct_front", [correct, d0, d1, d2, wrong, d3], 0, 4),
        ("correct_middle", [d1, d0, correct, d2, wrong, d3], 2, 4),
        ("correct_end", [d3, d0, d1, d2, wrong, correct], 5, 4),
        # V4/V5 preserve every non-key slot and swap only E2/E5.
        ("conflict_before_correct", [d0, wrong, d1, d2, correct, d3], 4, 1),
        ("correct_before_conflict", [d0, correct, d1, d2, wrong, d3], 1, 4),
        # V6 intentionally replaces the wrong passage with a fifth distractor.
        ("distractor_dominant", [d0, d1, d2, d3, correct, d4], 4, None),
    ]
    return [
        make_instance(sample, *layout, reused_distractor_count)
        for layout in layouts
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "data", "eval_instances_controlled_v2.jsonl"),
    )
    parser.add_argument(
        "--skip-missing-wrong",
        action="store_true",
        help="build the 2,439-sample valid subset instead of failing",
    )
    args = parser.parse_args()

    source = os.path.join(BASE_DIR, "data", "base_samples_with_wrong.jsonl")
    samples = load_jsonl(source)
    missing = [s["sample_id"] for s in samples
               if not str(s.get("wrong_evidence") or "").strip()]
    if missing and not args.skip_missing_wrong:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"Cannot build controlled-v2: {len(missing)} samples have no wrong "
            f"evidence (first ids: {preview}). Regenerate and validate them first."
        )

    if missing:
        samples = [sample for sample in samples
                   if str(sample.get("wrong_evidence") or "").strip()]
        print(f"Skipping {len(missing)} samples without wrong evidence")

    instances = []
    reused_samples = 0
    for sample in samples:
        sample_instances = build_variants_for_sample(sample)
        if sample_instances[0]["reused_distractor_count"]:
            reused_samples += 1
        instances.extend(sample_instances)
    save_jsonl(instances, args.output)
    print(f"Generated {len(instances)} controlled-v2 instances -> {args.output}")
    print(f"Samples using deterministically repeated distractors: {reused_samples}")


if __name__ == "__main__":
    main()

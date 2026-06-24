"""Build 6 position variants per base sample -> 15000 eval instances."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl

random.seed(123)
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

VARIANTS = [
    "correct_front",
    "correct_middle",
    "correct_end",
    "conflict_before_correct",
    "correct_before_conflict",
    "distractor_dominant",
]


def build_variants_for_sample(sample):
    correct_ev = sample.get("correct_evidence", "")
    wrong_ev = sample.get("wrong_evidence", "")
    distractors = sample.get("distractor_evidences", [])[:]

    while len(distractors) < 5:
        if distractors:
            distractors.append(random.choice(distractors))
        else:
            distractors.append("No additional information available.")

    def fill_slots(correct_pos, wrong_pos=None):
        slots = [None] * 6
        slots[correct_pos] = correct_ev
        if wrong_pos is not None and wrong_ev:
            slots[wrong_pos] = wrong_ev
        available = [i for i in range(6) if slots[i] is None]
        fillers = distractors[:]
        random.shuffle(fillers)
        for i, slot_idx in enumerate(available):
            slots[slot_idx] = fillers[i % len(fillers)]
        return slots

    variants = []

    # V1: correct at E1
    slots = fill_slots(correct_pos=0, wrong_pos=4)
    variants.append(make_instance(sample, "correct_front", slots, 0, 4))

    # V2: correct at E3
    slots = fill_slots(correct_pos=2, wrong_pos=4)
    variants.append(make_instance(sample, "correct_middle", slots, 2, 4))

    # V3: correct at E6
    slots = fill_slots(correct_pos=5, wrong_pos=1)
    variants.append(make_instance(sample, "correct_end", slots, 5, 1))

    # V4: conflict before correct — wrong at E2, correct at E5
    slots = fill_slots(correct_pos=4, wrong_pos=1)
    variants.append(make_instance(sample, "conflict_before_correct", slots, 4, 1))

    # V5: correct before conflict — correct at E2, wrong at E5
    slots = fill_slots(correct_pos=1, wrong_pos=4)
    variants.append(make_instance(sample, "correct_before_conflict", slots, 1, 4))

    # V6: distractor dominant — correct at E5, no wrong, heavy distractors
    slots = fill_slots(correct_pos=4, wrong_pos=None)
    variants.append(make_instance(sample, "distractor_dominant", slots, 4, None))

    return variants


def make_instance(sample, variant_name, slots, correct_pos, wrong_pos):
    return {
        "instance_id": f"{sample['sample_id']}_{variant_name}",
        "sample_id": sample["sample_id"],
        "variant": variant_name,
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        "source": sample.get("source", ""),
        "evidences": {f"E{i+1}": slots[i] for i in range(6)},
        "correct_evidence_position": f"E{correct_pos+1}",
        "wrong_evidence_position": f"E{wrong_pos+1}" if wrong_pos is not None else None,
    }


def main():
    wrong_path = os.path.join(BASE_DIR, 'data', 'base_samples_with_wrong.jsonl')
    base_path = os.path.join(BASE_DIR, 'data', 'base_samples.jsonl')
    if os.path.exists(wrong_path):
        samples = load_jsonl(wrong_path)
        print(f"Using base_samples_with_wrong.jsonl ({len(samples)} samples)")
    else:
        samples = load_jsonl(base_path)
        print(f"Using base_samples.jsonl ({len(samples)} samples, no wrong evidence)")

    all_instances = []
    for s in samples:
        all_instances.extend(build_variants_for_sample(s))

    out_path = os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl')
    save_jsonl(all_instances, out_path)
    print(f"Generated {len(all_instances)} eval instances -> {out_path}")


if __name__ == "__main__":
    main()

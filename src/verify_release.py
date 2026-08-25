"""Fail-fast consistency checks for the public PRICAI 2026 release."""

import csv
import json
import os
from collections import Counter, defaultdict

from build_controlled_variants import build_variants_for_sample


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    "gpt-5.4-medium": (0.7362, 0.9275, 0.1020, 0.1354, 0.8115, 0.1294, 0.9666),
    "deepseek-chat": (0.5390, 0.9677, 0.2828, 0.3082, 0.5212, 0.2673, 0.8170),
    "gpt-5.4-mini": (0.5105, 0.9666, 0.3188, 0.3416, 0.2625, 0.3344, 0.7216),
    "claude-haiku-4-5-20251001": (0.4594, 0.7629, 0.2256, 0.1874, 0.3653, 0.2113, 0.9096),
    "claude-sonnet-4-6": (0.4273, 0.4998, 0.0564, 0.0698, 0.8177, 0.0815, 0.9836),
    "deepseek-reasoner": (0.2307, 0.3991, 0.2588, 0.2466, 0.4021, 0.2426, 0.6880),
    "gemini-2.5-flash": (0.1286, 0.1939, 0.1308, 0.2116, 0.2753, 0.1923, 0.4409),
}
METRIC_COLUMNS = ("accuracy", "valid_response_rate", "PSR", "PBR", "CAR", "EAR", "CEU")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    instances = list(read_jsonl(os.path.join(BASE, "data/eval_instances.jsonl")))
    assert len(instances) == 15_000
    variants = defaultdict(set)
    for item in instances:
        variants[item["sample_id"]].add(item["variant"])
    assert len(variants) == 2_500
    assert all(len(value) == 6 for value in variants.values())

    for directory in ("raw_predictions", "parsed_predictions"):
        root = os.path.join(BASE, "outputs", directory)
        files = sorted(name for name in os.listdir(root) if name.endswith(".jsonl"))
        assert {name[:-6] for name in files} == set(MODELS)
        assert all(sum(1 for _ in read_jsonl(os.path.join(root, name))) == 15_000
                   for name in files)

    claims = Counter(item["status"] for item in read_jsonl(
        os.path.join(BASE, "data/wrong_claims.jsonl")))
    assert claims == {"decided": 2_004, "undecidable": 496}
    assert sum(item.get("has_nonempty_wrong_evidence") is True for item in read_jsonl(
        os.path.join(BASE, "data/wrong_claims.jsonl"))) == 2_439

    overall = {row["model"]: row for row in read_csv(
        os.path.join(BASE, "outputs/metrics/overall_metrics.csv"))}
    assert set(overall) == set(MODELS)
    for model, expected in MODELS.items():
        observed = tuple(float(overall[model][column]) for column in METRIC_COLUMNS)
        assert all(abs(a - b) < 0.00005 for a, b in zip(observed, expected)), (
            model, observed, expected)

    failures = read_csv(os.path.join(BASE, "outputs/metrics/failure_taxonomy.csv"))
    totals = [row for row in failures if row["variant_code"] == "TOTAL"]
    assert sum(int(row["total"]) for row in totals) == 105_000
    assert sum(int(row["n_ok"]) for row in totals) == 70_763

    audit = {row["check"]: int(row["count"]) for row in read_csv(
        os.path.join(BASE, "outputs/metrics/construction_audit.csv"))}
    assert audit["nonempty_wrong_evidence"] == 2_439
    assert audit["empty_wrong_evidence"] == 61
    assert audit["v1_v3_same_evidence_multiset"] == 583
    assert audit["v4_v5_same_evidence_multiset"] == 574
    assert audit["v4_v5_matched_nonempty_wrong"] == 513
    assert audit["v4_v5_only_key_swap"] == 21

    controlled_names = (
        "correct_front",
        "correct_middle",
        "correct_end",
        "conflict_before_correct",
        "correct_before_conflict",
    )
    reused_samples = 0
    for sample in read_jsonl(os.path.join(
            BASE, "data/base_samples_with_wrong.jsonl")):
        if not str(sample.get("wrong_evidence") or "").strip():
            continue
        generated = {
            row["variant"]: row for row in build_variants_for_sample(sample)
        }
        assert set(generated) == set(controlled_names) | {"distractor_dominant"}
        multisets = {
            tuple(sorted(generated[name]["evidences"].values()))
            for name in controlled_names
        }
        assert len(multisets) == 1, sample["sample_id"]
        v4 = generated["conflict_before_correct"]["evidences"]
        v5 = generated["correct_before_conflict"]["evidences"]
        assert all(v4[slot] == v5[slot]
                   for slot in ("E1", "E3", "E4", "E6"))
        assert v4["E2"] == v5["E5"] and v4["E5"] == v5["E2"]
        if generated["correct_front"]["reused_distractor_count"]:
            reused_samples += 1
    assert reused_samples == 3

    matched = read_csv(os.path.join(
        BASE, "outputs/metrics/content_matched_v4_v5.csv"))
    significant = [row for row in matched
                   if row["significant_at_0_01"] == "yes"]
    assert len(significant) == 1
    assert significant[0]["model"] == "claude-haiku-4-5-20251001"

    mitigation = read_csv(os.path.join(BASE, "outputs/mitigation/mitigation_delta.csv"))
    sig_counts = Counter(row["metric"] for row in mitigation
                         if row["significant"] == "yes")
    assert sig_counts["ACC"] == 6
    assert sig_counts["CAR"] == 5
    assert sig_counts["PBR"] == 2
    assert sig_counts["EAR"] == 2

    print("Release verification passed: artifacts are internally consistent, "
          "known v1 construction limitations are quantified, and the "
          "matched-content sensitivity results match.")


if __name__ == "__main__":
    main()

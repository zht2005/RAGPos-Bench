"""Failure taxonomy over raw model outputs.

Classifies every raw prediction record (7 models x 15,000 instances) into
exactly one of {ok, api_error, model_empty, unparseable} via
utils.classify_raw_output, broken down by variant code (V1..V6). It also
reconciles the taxonomy against empty parsed answers.
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_jsonl, classify_raw_output, VARIANT_CODES,
                   RECORD_OK, RECORD_API_ERROR, RECORD_MODEL_EMPTY,
                   RECORD_UNPARSEABLE)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
VARIANT_CODE_ORDER = ("V1", "V2", "V3", "V4", "V5", "V6")
def main():
    inst_variant = {}
    for inst in load_jsonl(os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl')):
        inst_variant[inst["instance_id"]] = VARIANT_CODES[inst["variant"]]

    raw_dir = os.path.join(BASE_DIR, 'outputs', 'raw_predictions')
    counts = {}  # (model, variant_code) -> Counter of record classes
    models = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith('.jsonl'):
            continue
        model = fname.replace('.jsonl', '')
        models.append(model)
        for rec in load_jsonl(os.path.join(raw_dir, fname)):
            vcode = inst_variant[rec["instance_id"]]
            key = (model, vcode)
            if key not in counts:
                counts[key] = Counter()
            counts[key][classify_raw_output(rec.get("raw_output"))] += 1

    rows = []
    model_totals = {}
    for model in models:
        total_c = Counter()
        for vcode in VARIANT_CODE_ORDER:
            c = counts.get((model, vcode), Counter())
            total_c.update(c)
            rows.append({
                "model": model, "variant_code": vcode,
                "n_ok": c[RECORD_OK], "n_api_error": c[RECORD_API_ERROR],
                "n_model_empty": c[RECORD_MODEL_EMPTY],
                "n_unparseable": c[RECORD_UNPARSEABLE],
                "total": sum(c.values()),
            })
        rows.append({
            "model": model, "variant_code": "TOTAL",
            "n_ok": total_c[RECORD_OK], "n_api_error": total_c[RECORD_API_ERROR],
            "n_model_empty": total_c[RECORD_MODEL_EMPTY],
            "n_unparseable": total_c[RECORD_UNPARSEABLE],
            "total": sum(total_c.values()),
        })
        model_totals[model] = total_c

    out_path = os.path.join(BASE_DIR, 'outputs', 'metrics', 'failure_taxonomy.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Failure taxonomy saved to {out_path} ({len(rows)} rows)\n")

    print(f"{'model':<32s} {'ok':>7s} {'api_err':>8s} {'empty':>7s} {'unpars':>7s} {'total':>7s}")
    grand = Counter()
    for model in models:
        c = model_totals[model]
        grand.update(c)
        print(f"{model:<32s} {c[RECORD_OK]:>7d} {c[RECORD_API_ERROR]:>8d} "
              f"{c[RECORD_MODEL_EMPTY]:>7d} {c[RECORD_UNPARSEABLE]:>7d} "
              f"{sum(c.values()):>7d}")
    print(f"{'ALL':<32s} {grand[RECORD_OK]:>7d} {grand[RECORD_API_ERROR]:>8d} "
          f"{grand[RECORD_MODEL_EMPTY]:>7d} {grand[RECORD_UNPARSEABLE]:>7d} "
          f"{sum(grand.values()):>7d}")

    # ----- Reconciliation against parsed predictions -----
    parsed_dir = os.path.join(BASE_DIR, 'outputs', 'parsed_predictions')
    empty_parsed = 0
    n_parsed = 0
    for fname in sorted(os.listdir(parsed_dir)):
        if not fname.endswith('.jsonl'):
            continue
        for p in load_jsonl(os.path.join(parsed_dir, fname)):
            n_parsed += 1
            if not (p.get("answer") or "").strip():
                empty_parsed += 1

    non_ok = sum(grand.values()) - grand[RECORD_OK]
    print("\nReconciliation:")
    print(f"  empty parsed answers observed : {empty_parsed} / {n_parsed} "
          f"({100.0 * empty_parsed / n_parsed:.1f}%)")
    print(f"  taxonomy non-ok records       : {non_ok} "
          f"(api_error {grand[RECORD_API_ERROR]} + model_empty {grand[RECORD_MODEL_EMPTY]}"
          f" + unparseable {grand[RECORD_UNPARSEABLE]})")
    print(f"  empty-parsed minus non-ok     : {empty_parsed - non_ok}")
    print("  (positive => 'ok' records with an empty answer field; negative =>\n"
          "   non-ok records whose parsed answer is non-empty, e.g. unparseable\n"
          "   raws stored verbatim as the answer with parse_error=True)")


if __name__ == "__main__":
    main()

"""Parse raw model outputs into structured predictions."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl, extract_json


def parse_single(raw_record):
    raw = raw_record["raw_output"]
    parsed = extract_json(raw)
    result = {
        "instance_id": raw_record["instance_id"],
        "model": raw_record["model"],
        "parse_error": False,
    }
    if parsed is None:
        result["parse_error"] = True
        result["answer"] = raw[:200] if raw else ""
        result["selected_evidence_ids"] = []
        result["has_conflict"] = False
        result["abstained"] = False
        result["confidence"] = 0
        result["brief_reason"] = ""
    else:
        result["answer"] = str(parsed.get("answer", ""))
        result["selected_evidence_ids"] = parsed.get("selected_evidence_ids", [])
        result["has_conflict"] = bool(parsed.get("has_conflict", False))
        result["abstained"] = bool(parsed.get("abstained", False))
        result["confidence"] = int(parsed.get("confidence", 0))
        result["brief_reason"] = str(parsed.get("brief_reason", ""))
    return result


def main():
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'raw_predictions')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'parsed_predictions')
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith('.jsonl'):
            continue
        raw_data = load_jsonl(os.path.join(raw_dir, fname))
        parsed = [parse_single(r) for r in raw_data]
        errors = sum(1 for p in parsed if p["parse_error"])
        save_jsonl(parsed, os.path.join(out_dir, fname))
        print(f"{fname}: {len(parsed)} parsed, {errors} errors")


if __name__ == "__main__":
    main()

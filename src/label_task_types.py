"""Offline task-type labelling for the 2,500 base samples.

Categories: entity, numerical, temporal, compositional, other.
Uses GPT-5.4-mini via the same right.codes proxy as wrong_evidence_qc.py.

Inputs:
  data/eval_instances.jsonl
Outputs:
  outputs/metrics/task_type_labels.csv      (sample_id, task_type)
  outputs/metrics/task_type_breakdown.csv   (model x task_type x metrics)
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

BASE = os.environ.get("RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import load_jsonl, match_answer  # noqa


def is_correct(pred, inst):
    return match_answer(pred.get("answer", ""), inst["gold_answer"], "general")


def label_one(client, model_id, q, a):
    PROMPT = ("Classify the QA task type. Output one JSON object with key "
              "\"task_type\" valued exactly one of: entity, numerical, "
              "temporal, compositional, other.\n"
              "- entity: asks about a person, place, organization, work, etc.\n"
              "- numerical: asks for a count, amount, percentage, or numeric value.\n"
              "- temporal: asks about a date, year, century, duration, or "
              "temporal ordering.\n"
              "- compositional: requires combining facts from multiple "
              "passages (multi-hop) where neither passage alone suffices.\n"
              "- other: yes/no, reasoning, list, anything else.\n"
              f"\nQUESTION: {q}\nGOLD ANSWER: {a}\n")
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.0,
        response_format={"type": "json_object"},
        timeout=30,
    )
    raw = resp.choices[0].message.content or ""
    d = json.loads(raw)
    t = (d.get("task_type") or "other").strip().lower()
    if t not in ("entity", "numerical", "temporal", "compositional", "other"):
        t = "other"
    return t


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)
    from openai import OpenAI
    base_url = "https://www.right.codes/codex/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    model_id = "gpt-5.4-mini"

    instances = load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))
    by_sample = {}
    for i in instances:
        sid = i["sample_id"]
        if sid not in by_sample:
            by_sample[sid] = i
    samples = list(by_sample.values())
    print(f"Labelling {len(samples)} base samples ...")

    out_path = os.path.join(BASE, "outputs/metrics/task_type_labels.csv")
    rows = []
    counts = defaultdict(int)
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            t = label_one(client, model_id, s["question"], s["gold_answer"])
        except Exception as e:
            t = "error"
        counts[t] += 1
        rows.append({"sample_id": s["sample_id"], "source": s["source"],
                     "question": s["question"][:200],
                     "gold_answer": s["gold_answer"][:80],
                     "task_type": t})
        if (i + 1) % 100 == 0 or i == len(samples) - 1:
            print(f"  {i+1}/{len(samples)} {dict(counts)} elapsed={time.time()-t0:.0f}s")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[ok] labels -> {out_path}")

    # Aggregate per (model, task_type)
    label_by_sid = {r["sample_id"]: r["task_type"] for r in rows}
    insts_by_iid = {i["instance_id"]: i for i in instances}
    pred_dir = os.path.join(BASE, "outputs/parsed_predictions")
    by_mt = defaultdict(lambda: {"acc_n": 0, "acc_correct": 0,
                                  "ear_n": 0, "ear_adopted": 0,
                                  "ceu_n": 0, "ceu_used": 0})
    models = []
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.endswith(".jsonl"):
            continue
        m = fn[:-6]; models.append(m)
        for p in load_jsonl(os.path.join(pred_dir, fn)):
            iid = p["instance_id"]; inst = insts_by_iid.get(iid)
            if not inst: continue
            tt = label_by_sid.get(inst["sample_id"], "other")
            if tt == "error": continue
            key = (m, tt)
            by_mt[key]["acc_n"] += 1
            if is_correct(p, inst): by_mt[key]["acc_correct"] += 1
            v = inst.get("variant")
            if v in ("conflict_before_correct", "correct_before_conflict"):
                by_mt[key]["ear_n"] += 1
                wrong_pos = inst.get("wrong_evidence_position")
                wrong_text = inst["evidences"].get(wrong_pos, "") if wrong_pos else ""
                # EAR proxy: answer wrong AND not correct
                if not is_correct(p, inst): by_mt[key]["ear_adopted"] += 1
            cor = inst.get("correct_evidence_position")
            if cor:
                by_mt[key]["ceu_n"] += 1
                if cor in (p.get("selected_evidence_ids") or []):
                    by_mt[key]["ceu_used"] += 1

    out2 = os.path.join(BASE, "outputs/metrics/task_type_breakdown.csv")
    with open(out2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task_type", "n", "accuracy", "ear", "ceu"])
        for (m, tt), d in sorted(by_mt.items()):
            acc = d["acc_correct"]/d["acc_n"] if d["acc_n"] else 0
            ear = d["ear_adopted"]/d["ear_n"] if d["ear_n"] else 0
            ceu = d["ceu_used"]/d["ceu_n"] if d["ceu_n"] else 0
            w.writerow([m, tt, d["acc_n"], f"{acc:.4f}", f"{ear:.4f}", f"{ceu:.4f}"])
    print(f"[ok] breakdown -> {out2}")


if __name__ == "__main__":
    main()

"""Wrong-evidence integrity check (Phase A: local rules; Phase B: LLM judge).

Phase A is run unconditionally and produces conservative pass rates.
Phase B is gated on OPENAI_API_KEY being set in the environment.

Inputs:
  data/eval_instances.jsonl   (each instance carries question, gold_answer,
                               6 evidence slots, correct_evidence_position,
                               wrong_evidence_position)

Outputs:
  outputs/metrics/wrong_evidence_qc_local.csv     -- per-sample local QC results
  outputs/metrics/wrong_evidence_qc_llm.csv       -- per-sample LLM judge results (if Phase B ran)
  outputs/metrics/wrong_evidence_qc_summary.csv   -- aggregate pass rates
  paper_assets/wrong_evidence_qc_sample100.csv    -- 100-row deterministic sample for human inspection
"""
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

BASE = ".."
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import load_jsonl, normalize_answer  # noqa

DATA_PATH = os.path.join(BASE, "data/eval_instances.jsonl")
METRICS_DIR = os.path.join(BASE, "outputs/metrics")
ASSETS_DIR = os.path.join(BASE, "paper_assets")
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z'’\-]{2,}(?:\s+[A-Z][A-Za-z'’\-]{2,}){0,4})\b")


def deduplicate(instances):
    """Wrong evidence is reused across all 6 variants of one base sample.
    Pick one representative instance per sample_id (the one with the
    correct_front variant) and harvest e_dagger from there.
    """
    by_sample = {}
    for inst in instances:
        sid = inst["sample_id"]
        if sid in by_sample:
            continue
        wrong_pos = inst.get("wrong_evidence_position")
        if not wrong_pos:
            continue
        wrong_text = inst["evidences"].get(wrong_pos, "").strip()
        if not wrong_text:
            continue
        by_sample[sid] = {
            "sample_id": sid,
            "source": inst.get("source", "unknown"),
            "question": inst["question"],
            "gold_answer": inst["gold_answer"],
            "wrong_evidence": wrong_text,
            "correct_evidence": inst["evidences"].get(inst["correct_evidence_position"], ""),
        }
    return list(by_sample.values())


def check_plausible(text):
    n = len(text.split())
    if n < 25 or n > 600:
        return False, f"len={n}"
    if any(tok in text.lower() for tok in ("placeholder", "example text", "lorem ipsum")):
        return False, "template tok"
    return True, f"len={n}"


def check_topical(question, wrong, correct):
    q_ents = set(ENTITY_RE.findall(question))
    c_ents = set(ENTITY_RE.findall(correct))
    target = (q_ents | c_ents) - {"The", "What", "Which", "Who", "When", "Where", "Why", "How"}
    if not target:
        return False, "no entities in q/correct"
    hits = [e for e in target if e in wrong]
    return (len(hits) > 0), f"hits={len(hits)}/{len(target)}"


def check_contradiction(gold, wrong):
    """Wrong evidence should NOT simply repeat the gold answer phrase as an
    affirmative claim. Contradiction signal: (i) wrong does not literally
    contain the gold answer phrase, OR (ii) it contains the gold but in a
    negated context. We approximate (ii) by the presence of negation tokens
    near the gold mention.
    """
    g = (gold or "").strip()
    if not g:
        return False, "empty gold"
    g_norm = normalize_answer(g)
    w_norm = normalize_answer(wrong)
    if g_norm and g_norm in w_norm:
        # gold mentioned: look for negation within ±60 chars window
        idx = w_norm.find(g_norm)
        window = w_norm[max(0, idx - 60): idx + len(g_norm) + 60]
        neg_tokens = [" not ", " no ", " never ", " incorrect", " false", " untrue",
                      " contrary", " however ", " rather ", " instead "]
        if any(t in " " + window + " " for t in neg_tokens):
            return True, "gold negated"
        return False, "gold appears affirmatively"
    return True, "gold absent (contradicts by alternative)"


def phase_a():
    instances = load_jsonl(DATA_PATH)
    samples = deduplicate(instances)
    rows = []
    for s in samples:
        p_ok, p_why = check_plausible(s["wrong_evidence"])
        t_ok, t_why = check_topical(s["question"], s["wrong_evidence"], s["correct_evidence"])
        c_ok, c_why = check_contradiction(s["gold_answer"], s["wrong_evidence"])
        rows.append({
            "sample_id": s["sample_id"],
            "source": s["source"],
            "plausible": int(p_ok), "plausible_reason": p_why,
            "topical":   int(t_ok), "topical_reason":   t_why,
            "contradicts": int(c_ok), "contradiction_reason": c_why,
            "all_pass": int(p_ok and t_ok and c_ok),
        })
    out = os.path.join(METRICS_DIR, "wrong_evidence_qc_local.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    summary = {
        "n_samples": n,
        "plausible_pass": sum(r["plausible"] for r in rows) / n,
        "topical_pass": sum(r["topical"] for r in rows) / n,
        "contradicts_pass": sum(r["contradicts"] for r in rows) / n,
        "all_three_pass": sum(r["all_pass"] for r in rows) / n,
    }
    print("=== Phase A (local rules) ===")
    for k, v in summary.items():
        print(f"  {k:<20s} {v:.4f}" if isinstance(v, float) else f"  {k:<20s} {v}")
    return rows, samples, summary


def write_inspection_sample(rows, samples, n=100):
    """Deterministic stratified 100-row sample for human checklist."""
    by_sid = {s["sample_id"]: s for s in samples}
    by_pass = defaultdict(list)
    for r in rows:
        by_pass[r["all_pass"]].append(r)
    rng = np.random.default_rng(42)
    take_pass = min(60, len(by_pass[1]))
    take_fail = min(40, len(by_pass[0]))
    pass_rows = list(rng.choice(by_pass[1], size=take_pass, replace=False)) if by_pass[1] else []
    fail_rows = list(rng.choice(by_pass[0], size=take_fail, replace=False)) if by_pass[0] else []
    pick = list(pass_rows) + list(fail_rows)
    out = os.path.join(ASSETS_DIR, "wrong_evidence_qc_sample100.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source", "question", "gold_answer",
                    "wrong_evidence", "all_pass_local",
                    "plausible_reason", "topical_reason", "contradiction_reason",
                    "human_label_topical", "human_label_plausible",
                    "human_label_contradiction", "human_notes"])
        for r in pick:
            sid = r["sample_id"]; s = by_sid[sid]
            w.writerow([sid, s["source"], s["question"], s["gold_answer"],
                        s["wrong_evidence"], r["all_pass"],
                        r["plausible_reason"], r["topical_reason"], r["contradiction_reason"],
                        "", "", "", ""])
    print(f"[ok] human-inspection sample -> {out} ({len(pick)} rows)")


def phase_b(rows_a, samples, summary):
    """LLM-judge QC.

    Strategy (per user request 2026-06-21): do not bulk-audit all 2,500.
    Audit only (i) every Phase-A failed sample (any of the 3 rules == 0) and
    (ii) a deterministic random 300-sample audit set drawn from Phase-A passes.
    The LLM returns three independent labels: topical / plausible / contradictory.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[skip] Phase B: OPENAI_API_KEY not set")
        return None
    try:
        from openai import OpenAI
    except Exception as e:
        print(f"[skip] Phase B: openai SDK missing ({e})")
        return None

    by_sid = {s["sample_id"]: s for s in samples}
    fails = [r for r in rows_a if r["all_pass"] == 0]
    passes = [r for r in rows_a if r["all_pass"] == 1]
    rng = np.random.default_rng(42)
    n_random = min(300, len(passes))
    audit_pass = list(rng.choice(passes, size=n_random, replace=False)) if passes else []
    audit = list(fails) + list(audit_pass)
    audit_sids = [r["sample_id"] for r in audit]

    print(f"\n=== Phase B audit set ===")
    print(f"  Phase-A failed:        {len(fails):>4d}")
    print(f"  Random pass sample:    {len(audit_pass):>4d}")
    print(f"  Total LLM calls:       {len(audit):>4d}")

    base_url = "https://www.right.codes/codex/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    model_id = "gpt-5.4-mini"

    PROMPT = """You are a careful annotator auditing a wrong-evidence passage that
was synthetically generated to mislead an LLM in a RAG benchmark.

Given a QUESTION, the GOLD ANSWER, and the candidate WRONG EVIDENCE, judge
three independent properties.

1. topical: does the wrong evidence stay on the topic of the question?
2. plausible: does it read like a plausible encyclopedic snippet (coherent,
   not garbled, factual-sounding)?
3. contradictory: does it make a claim that contradicts the gold answer
   (i.e., would mislead a reader who trusts only this passage)?

Output a single JSON object with three keys, each one of "yes", "no", or
"unclear":
{"topical":"yes|no|unclear","plausible":"yes|no|unclear","contradictory":"yes|no|unclear"}

QUESTION: {q}
GOLD ANSWER: {a}
WRONG EVIDENCE: {e}
"""

    rows_b = []
    counts = {"topical": defaultdict(int), "plausible": defaultdict(int),
              "contradictory": defaultdict(int)}
    t0 = time.time()
    for i, r in enumerate(audit):
        sid = r["sample_id"]
        s = by_sid[sid]
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": PROMPT
                           .replace("{q}", s["question"])
                           .replace("{a}", s["gold_answer"])
                           .replace("{e}", s["wrong_evidence"][:1500])}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = resp.choices[0].message.content or ""
            d = json.loads(raw)
            verdict = {k: (d.get(k) or "").strip().lower() for k in
                       ("topical", "plausible", "contradictory")}
            for k in verdict:
                if verdict[k] not in ("yes", "no", "unclear"):
                    verdict[k] = "unclear"
        except Exception as e:
            verdict = {"topical": "error", "plausible": "error",
                       "contradictory": "error"}
            r["err"] = str(e)[:120]
        for k in verdict:
            counts[k][verdict[k]] += 1
        rows_b.append({
            "sample_id": sid, "source": s["source"],
            "from_phase_a": "fail" if r["all_pass"] == 0 else "pass-random",
            "topical_llm": verdict["topical"],
            "plausible_llm": verdict["plausible"],
            "contradictory_llm": verdict["contradictory"],
        })
        if (i + 1) % 50 == 0 or i == len(audit) - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(audit)}  topical={dict(counts['topical'])}  "
                  f"contradictory={dict(counts['contradictory'])}  "
                  f"elapsed={elapsed:.0f}s")

    out = os.path.join(METRICS_DIR, "wrong_evidence_qc_llm.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        w.writeheader(); w.writerows(rows_b)
    print(f"\n[ok] LLM-audit -> {out}")

    n = len(rows_b)
    for k in ("topical", "plausible", "contradictory"):
        summary[f"llm_{k}_yes"] = counts[k]["yes"] / n
        summary[f"llm_{k}_no"] = counts[k]["no"] / n
        summary[f"llm_{k}_unclear"] = counts[k]["unclear"] / n

    pass_subset = [r for r in rows_b if r["from_phase_a"] == "pass-random"]
    if pass_subset:
        all_yes = sum(1 for r in pass_subset
                      if r["topical_llm"] == "yes" and r["plausible_llm"] == "yes"
                      and r["contradictory_llm"] == "yes")
        summary["llm_audit_pass_rate_on_phaseA_pass"] = all_yes / len(pass_subset)

    fail_subset = [r for r in rows_b if r["from_phase_a"] == "fail"]
    if fail_subset:
        any_yes = sum(1 for r in fail_subset
                      if r["topical_llm"] == "yes" and r["plausible_llm"] == "yes"
                      and r["contradictory_llm"] == "yes")
        summary["llm_audit_recovers_phaseA_fail"] = any_yes / len(fail_subset)

    print("\n=== Phase B aggregates ===")
    for k, v in summary.items():
        if k.startswith("llm_"):
            print(f"  {k:<40s} {v:.4f}")
    return rows_b


def write_summary(summary):
    out = os.path.join(METRICS_DIR, "wrong_evidence_qc_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])
    print(f"\n[ok] summary -> {out}")


def main():
    rows_a, samples, summary = phase_a()
    write_inspection_sample(rows_a, samples)
    phase_b(rows_a, samples, summary)
    write_summary(summary)


if __name__ == "__main__":
    main()

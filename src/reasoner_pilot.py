"""Pilot: 498-instance DeepSeek-Reasoner re-run with max_tokens=2048.

Goal: test whether the high empty-output rate drops under a larger output
budget, and measure accuracy against the max_tokens=512 baseline.

Sampling: 83 instances from each of the six variants, drawn with deterministic
seed 42, for 498 instances in total.

Outputs:
  outputs/reasoner_pilot/raw.jsonl   -- new predictions
  outputs/reasoner_pilot/summary.txt -- comparison vs original
"""
import os, sys, json, time, random
from collections import defaultdict

BASE = os.environ.get("RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import load_jsonl, save_jsonl, match_answer  # noqa

OUT_DIR = os.path.join(BASE, "outputs/reasoner_pilot")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_TEMPLATE = """You are given a question and several retrieved evidence passages. Answer the question using only the provided evidence.

Important instructions:
1. If the evidence is conflicting, say that the evidence is conflicting and do not arbitrarily choose one answer unless one passage is clearly more recent or more reliable.
2. If the evidence is insufficient, say that the answer cannot be determined from the provided evidence.
3. Do not use outside knowledge.
4. Return your response in valid JSON only.

Question:
{question}

Retrieved Evidence:
[E1] {E1}
[E2] {E2}
[E3] {E3}
[E4] {E4}
[E5] {E5}
[E6] {E6}

Return JSON with the following fields:
{{"answer": "...", "selected_evidence_ids": ["E1"], "has_conflict": true/false, "abstained": true/false, "confidence": 1-5, "brief_reason": "..."}}"""


def build_prompt(inst):
    e = inst["evidences"]
    return PROMPT_TEMPLATE.format(
        question=inst["question"],
        E1=e["E1"], E2=e["E2"], E3=e["E3"], E4=e["E4"], E5=e["E5"], E6=e["E6"],
    )


def parse_json(text):
    if not text: return None
    import re
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try: return json.loads(t)
    except: pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: return None
    return None


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    # Load instances
    insts = list(load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl")))
    by_variant = defaultdict(list)
    for i in insts:
        by_variant[i["variant"]].append(i)
    rng = random.Random(42)
    pilot = []
    PER_VARIANT = 500 // len(by_variant)  # ~83
    for v in sorted(by_variant.keys()):
        sub = by_variant[v][:]
        rng.shuffle(sub)
        pilot.extend(sub[:PER_VARIANT])
    rng.shuffle(pilot)
    print(f"Pilot size: {len(pilot)} instances ({PER_VARIANT}/variant)")

    out_path = os.path.join(OUT_DIR, "raw.jsonl")
    done = set()
    results = []
    if os.path.exists(out_path):
        results = list(load_jsonl(out_path))
        done = {r["instance_id"] for r in results}
        print(f"Resuming: {len(done)} already done")

    t0 = time.time()
    for i, inst in enumerate(pilot):
        iid = inst["instance_id"]
        if iid in done:
            continue
        prompt = build_prompt(inst)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2048,
                    timeout=120,
                )
                raw = resp.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raw = json.dumps({"error": str(e)[:200]})
        results.append({
            "instance_id": iid, "sample_id": inst["sample_id"],
            "variant": inst["variant"], "raw_output": raw,
        })
        if (i + 1) % 25 == 0 or i == len(pilot) - 1:
            save_jsonl(results, out_path)
            print(f"  {i+1}/{len(pilot)}  elapsed={time.time()-t0:.0f}s")

    save_jsonl(results, out_path)
    print(f"\n[ok] {out_path}")

    # Compare against baseline (parsed_predictions/deepseek-reasoner.jsonl)
    baseline = {p["instance_id"]: p for p in load_jsonl(
        os.path.join(BASE, "outputs/parsed_predictions/deepseek-reasoner.jsonl"))}
    inst_by_iid = {i["instance_id"]: i for i in insts}

    new_n = len(results)
    new_empty_raw = sum(1 for r in results if not (r["raw_output"] or "").strip())
    new_parsed_empty = 0
    new_correct = 0
    base_n = 0
    base_empty_raw = 0
    base_parsed_empty = 0
    base_correct = 0

    for r in results:
        inst = inst_by_iid[r["instance_id"]]
        d = parse_json(r["raw_output"]) or {}
        ans = (d.get("answer") or "").strip()
        if not ans: new_parsed_empty += 1
        if ans and match_answer(ans, inst["gold_answer"], "general"): new_correct += 1
        b = baseline.get(r["instance_id"])
        if not b: continue
        base_n += 1
        ba = (b.get("answer") or "").strip()
        if not ba: base_parsed_empty += 1
        if ba and match_answer(ba, inst["gold_answer"], "general"): base_correct += 1
        # raw_output not in parsed file; check raw_predictions
    # raw_predictions baseline:
    raw_base = {p["instance_id"]: p for p in load_jsonl(
        os.path.join(BASE, "outputs/raw_predictions/deepseek-reasoner.jsonl"))}
    base_empty_raw = sum(1 for r in results if r["instance_id"] in raw_base and
                          not (raw_base[r["instance_id"]].get("raw_output") or "").strip())

    summary = f"""=== DeepSeek-Reasoner Pilot Comparison ===
Pilot size: {new_n} instances (max_tokens=2048)
Baseline matches: {base_n} (max_tokens=512)

Empty raw_output (API returned ""):
  baseline (512):  {base_empty_raw}/{base_n} ({base_empty_raw/base_n*100:.1f}%)
  pilot   (2048):  {new_empty_raw}/{new_n} ({new_empty_raw/new_n*100:.1f}%)
  delta:           {(new_empty_raw/new_n - base_empty_raw/base_n)*100:+.1f}pp

Parsed answer empty (couldn't extract any answer):
  baseline (512):  {base_parsed_empty}/{base_n} ({base_parsed_empty/base_n*100:.1f}%)
  pilot   (2048):  {new_parsed_empty}/{new_n} ({new_parsed_empty/new_n*100:.1f}%)
  delta:           {(new_parsed_empty/new_n - base_parsed_empty/base_n)*100:+.1f}pp

Accuracy (gold-answer match, only on matched-instance subset):
  baseline (512):  {base_correct}/{base_n} ({base_correct/base_n*100:.2f}%)
  pilot   (2048):  {new_correct}/{new_n} ({new_correct/new_n*100:.2f}%)
  delta:           {(new_correct/new_n - base_correct/base_n)*100:+.2f}pp
"""
    print(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()

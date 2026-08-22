"""Re-run Gemini-2.5-Flash on Phase A conflict-aware prompt using Google's
official google-generativeai SDK (the right.codes proxy channel for Gemini
was down for 10+ hours during the main mitigation run).

Same 300-sample subset, same V4+V5 (600 instances), same conflict-aware
prompt. Output overwrites
outputs/mitigation/phase_a__gemini-2.5-flash.jsonl.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

import google.generativeai as genai

BASE = os.environ.get("RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import load_jsonl, save_jsonl  # noqa
from mitigation_utility import build_prompt, select_subset, CONFLICT_AWARE_PRELUDE  # noqa

OUT = os.path.join(BASE, "outputs/mitigation/phase_a__gemini-2.5-flash.jsonl")
SEED = 42
SAMPLE_N = 300
MODEL_ID = "gemini-2.5-flash"

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY env var required", file=sys.stderr)
    sys.exit(1)

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    MODEL_ID,
    generation_config={"temperature": 0.0, "max_output_tokens": 512},
)


def call(prompt, max_retries=3):
    last = None
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt, request_options={"timeout": 60})
            txt = resp.text or ""
            if txt.strip():
                return txt
            last = "empty content"
        except Exception as e:
            last = str(e)[:250]
            time.sleep(2.0 ** attempt)
    return json.dumps({"error": last or "unknown"})


def main():
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    subset = select_subset(insts.values())
    work = [iv4 for _, iv4, _ in subset] + [iv5 for _, _, iv5 in subset]
    print(f"Re-running Gemini on {len(work)} instances...")

    done = set()
    results = []
    if os.path.exists(OUT):
        results = list(load_jsonl(OUT))
        done = {r["instance_id"] for r in results}
        print(f"Resuming: {len(done)} already in {OUT}")

    todo = [iid for iid in work if iid not in done]
    n_err = 0
    t0 = time.time()
    for i, iid in enumerate(todo):
        inst = insts[iid]
        prompt = build_prompt(inst["question"], inst["evidences"], conflict_aware=True)
        raw = call(prompt)
        if raw.startswith('{"error"'):
            n_err += 1
        results.append({
            "instance_id": iid, "sample_id": inst["sample_id"],
            "variant": inst["variant"], "model": MODEL_ID,
            "condition": "conflict_aware", "raw_output": raw,
        })
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            save_jsonl(results, OUT)
            print(f"  {i+1}/{len(todo)} elapsed={time.time()-t0:.0f}s err={n_err}")
    save_jsonl(results, OUT)
    print(f"\nDone. n={len(results)} err={n_err}")


if __name__ == "__main__":
    main()

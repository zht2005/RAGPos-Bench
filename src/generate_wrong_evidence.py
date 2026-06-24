"""Generate plausible-but-wrong evidence for each base sample using LLM."""
import json
import os
import sys
import time
import yaml
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

PROMPT = """Given a question and its correct answer with supporting evidence, generate a short paragraph (2-3 sentences) that:
1. Discusses the same topic as the question
2. Sounds plausible and authoritative
3. But contains a DIFFERENT, INCORRECT answer to the question

The wrong paragraph should be misleading — it should look like a valid source but lead to a wrong conclusion.

Question: {question}
Correct Answer: {gold_answer}
Correct Evidence: {correct_evidence}

Generate ONLY the wrong/misleading paragraph, nothing else. Do not include any explanation or labels."""


def main():
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wm = config["wrong_evidence_model"]
    api_key = os.environ.get(wm["env_key"])
    if not api_key:
        print(f"[ERROR] No API key found for {wm['env_key']}")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=wm["base_url"])
    samples = load_jsonl(os.path.join(BASE_DIR, 'data', 'base_samples.jsonl'))

    out_path = os.path.join(BASE_DIR, 'data', 'base_samples_with_wrong.jsonl')
    already_done = set()
    if os.path.exists(out_path):
        done = load_jsonl(out_path)
        already_done = {s["sample_id"] for s in done if s.get("wrong_evidence")}
        print(f"  Resuming: {len(already_done)} already done")

    results = [s for s in load_jsonl(out_path)] if os.path.exists(out_path) else []
    results_map = {s["sample_id"]: s for s in results}

    todo = [s for s in samples if s["sample_id"] not in already_done]
    print(f"  Generating wrong evidence for {len(todo)} samples...")

    for sample in tqdm(todo, desc="Generating wrong evidence"):
        prompt = PROMPT.format(
            question=sample["question"],
            gold_answer=sample["gold_answer"],
            correct_evidence=sample["correct_evidence"][:500],
        )
        try:
            resp = client.chat.completions.create(
                model=wm["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            wrong_ev = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [WARN] {sample['sample_id']}: {e}")
            wrong_ev = ""
            time.sleep(2)

        sample["wrong_evidence"] = wrong_ev
        results_map[sample["sample_id"]] = sample

        if len(results_map) % 50 == 0:
            save_jsonl(list(results_map.values()), out_path)

    final = []
    for s in samples:
        if s["sample_id"] in results_map:
            final.append(results_map[s["sample_id"]])
        else:
            s["wrong_evidence"] = ""
            final.append(s)

    save_jsonl(final, out_path)
    print(f"Done. {len(final)} samples with wrong evidence -> {out_path}")


if __name__ == "__main__":
    main()

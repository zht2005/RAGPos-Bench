"""Run LLM inference on eval instances via OpenAI-compatible APIs."""
import json
import os
import random
import sys
import time
import yaml
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

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


def build_prompt(instance):
    evs = instance["evidences"]
    return PROMPT_TEMPLATE.format(
        question=instance["question"],
        E1=evs["E1"], E2=evs["E2"], E3=evs["E3"],
        E4=evs["E4"], E5=evs["E5"], E6=evs["E6"],
    )


def call_model(client, model_id, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return json.dumps({"error": str(e)})


def run_single_model(model_cfg, instances, out_dir):
    model_id = model_cfg["model_id"]
    base_url = model_cfg["base_url"]
    api_key = os.environ.get(model_cfg["env_key"])

    if not api_key:
        print(f"  [SKIP] {model_id}: no API key ({model_cfg['env_key']})")
        return

    out_path = os.path.join(out_dir, f"{model_id}.jsonl")

    already_done = set()
    if os.path.exists(out_path):
        existing = load_jsonl(out_path)
        already_done = {r["instance_id"] for r in existing}
        print(f"  Resuming {model_id}: {len(already_done)} already done")

    todo = [inst for inst in instances if inst["instance_id"] not in already_done]
    if not todo:
        print(f"  {model_id}: all done ({len(already_done)} instances)")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    results = list(load_jsonl(out_path)) if os.path.exists(out_path) else []

    print(f"  Running {model_id}: {len(todo)} remaining...")
    for inst in tqdm(todo, desc=model_id):
        prompt = build_prompt(inst)
        raw_output = call_model(client, model_id, prompt)
        results.append({
            "instance_id": inst["instance_id"],
            "model": model_id,
            "raw_output": raw_output,
        })
        if len(results) % 100 == 0:
            save_jsonl(results, out_path)

    save_jsonl(results, out_path)
    print(f"  {model_id}: saved {len(results)} predictions -> {out_path}")


def main():
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    instances = load_jsonl(os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl'))
    out_dir = os.path.join(BASE_DIR, 'outputs', 'raw_predictions')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running inference on {len(instances)} instances...")
    for model_cfg in config["models"]:
        run_single_model(model_cfg, instances, out_dir)

    print("\nInference complete.")


if __name__ == "__main__":
    main()

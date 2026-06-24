"""Run a single model inference. Usage: python run_single_model.py <model_index>"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl
from run_models import build_prompt, call_model, BASE_DIR

from openai import OpenAI
from tqdm import tqdm


def main():
    model_idx = int(sys.argv[1])
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["models"][model_idx]
    model_id = model_cfg["model_id"]
    base_url = model_cfg["base_url"]
    api_key = os.environ.get(model_cfg["env_key"])

    if not api_key:
        print(f"[SKIP] {model_id}: no key")
        return

    instances = load_jsonl(os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl'))
    out_dir = os.path.join(BASE_DIR, 'outputs', 'raw_predictions')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_id}.jsonl")

    already_done = set()
    results = []
    if os.path.exists(out_path):
        results = load_jsonl(out_path)
        already_done = {r["instance_id"] for r in results}

    todo = [inst for inst in instances if inst["instance_id"] not in already_done]
    if not todo:
        print(f"{model_id}: already complete ({len(results)})")
        return

    print(f"{model_id}: {len(todo)} remaining (resuming from {len(already_done)})")
    client = OpenAI(api_key=api_key, base_url=base_url)

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
    print(f"{model_id}: done ({len(results)} total)")


if __name__ == "__main__":
    main()

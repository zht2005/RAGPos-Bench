"""Run gemini-2.5-flash using Google official API."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl
from run_models import build_prompt, BASE_DIR

from google import genai
from tqdm import tqdm

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_ID = "gemini-2.5-flash"


def call_gemini(client, prompt, max_retries=3):
    import time
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return json.dumps({"error": str(e)})


def main():
    client = genai.Client(api_key=GEMINI_API_KEY)

    instances = load_jsonl(os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl'))
    out_dir = os.path.join(BASE_DIR, 'outputs', 'raw_predictions')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{MODEL_ID}.jsonl")

    already_done = set()
    results = []
    if os.path.exists(out_path):
        results = load_jsonl(out_path)
        already_done = {r["instance_id"] for r in results}

    # Filter out old blocked results
    results = [r for r in results if "blocked" not in r.get("raw_output", "")]
    already_done = {r["instance_id"] for r in results}

    todo = [inst for inst in instances if inst["instance_id"] not in already_done]
    if not todo:
        print(f"{MODEL_ID}: already complete ({len(results)})")
        return

    print(f"{MODEL_ID}: {len(todo)} remaining (resuming from {len(already_done)})")

    for inst in tqdm(todo, desc=MODEL_ID):
        prompt = build_prompt(inst)
        raw_output = call_gemini(client, prompt)
        results.append({
            "instance_id": inst["instance_id"],
            "model": MODEL_ID,
            "raw_output": raw_output,
        })
        if len(results) % 100 == 0:
            save_jsonl(results, out_path)

    save_jsonl(results, out_path)
    print(f"{MODEL_ID}: done ({len(results)} total)")


if __name__ == "__main__":
    main()

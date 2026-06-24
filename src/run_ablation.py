"""Run ablation experiments."""
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl
from run_models import build_prompt, mock_predict, PROMPT_TEMPLATE, run_model
from parse_outputs import parse_single
from evaluate import (load_instances, evaluate_model, write_csv)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

PROMPT_NO_CONFLICT = """You are given a question and several retrieved evidence passages. Answer the question using only the provided evidence.

Important instructions:
1. If the evidence is insufficient, say that the answer cannot be determined from the provided evidence.
2. Do not use outside knowledge.
3. Return your response in valid JSON only.

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

PROMPT_NO_EID = """You are given a question and several retrieved evidence passages. Answer the question using only the provided evidence.

Important instructions:
1. If the evidence is conflicting, say that the evidence is conflicting.
2. If the evidence is insufficient, say that the answer cannot be determined.
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
{{"answer": "...", "has_conflict": true/false, "abstained": true/false, "confidence": 1-5, "brief_reason": "..."}}"""


def run_ablation(ablation_name, prompt_template, models_to_test=None):
    instances = load_jsonl(os.path.join(BASE_DIR, 'data', 'eval_instances.jsonl'))
    out_dir = os.path.join(BASE_DIR, 'outputs', 'ablation', ablation_name)
    os.makedirs(out_dir, exist_ok=True)

    if models_to_test is None:
        models_to_test = ["gpt-4o-mini", "claude-sonnet-4-20250514", "deepseek-chat"]

    all_metrics = []
    for model_name in models_to_test:
        results = []
        for inst in instances:
            evs = inst["evidences"]
            prompt = prompt_template.format(
                question=inst["question"],
                E1=evs["E1"], E2=evs["E2"], E3=evs["E3"],
                E4=evs["E4"], E5=evs["E5"], E6=evs["E6"],
            )
            raw_output = mock_predict(inst, model_name)
            results.append({
                "instance_id": inst["instance_id"],
                "model": model_name,
                "prompt": prompt,
                "raw_output": raw_output,
            })
        parsed = [parse_single(r) for r in results]
        save_jsonl(parsed, os.path.join(out_dir, f"{model_name}.jsonl"))

        inst_map = {i["instance_id"]: i for i in instances}
        metrics = evaluate_model(model_name, parsed, inst_map)
        metrics["ablation"] = ablation_name
        all_metrics.append(metrics)

    write_csv(all_metrics, os.path.join(out_dir, 'metrics.csv'))
    print(f"  Ablation '{ablation_name}' done -> {out_dir}")
    return all_metrics


def main():
    print("Running ablation experiments...")
    r1 = run_ablation("no_conflict_instruction", PROMPT_NO_CONFLICT)
    r2 = run_ablation("no_evidence_id", PROMPT_NO_EID)

    combined = r1 + r2
    out_path = os.path.join(BASE_DIR, 'outputs', 'ablation', 'ablation_summary.csv')
    write_csv(combined, out_path)
    print(f"Ablation summary -> {out_path}")


if __name__ == "__main__":
    main()

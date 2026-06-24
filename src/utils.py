import json
import re
import os
import string


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def normalize_answer(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_number(text):
    if not text:
        return None
    text = text.replace(',', '').replace('$', '')
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None


def match_numerical(pred, gold):
    pred_num = extract_number(pred)
    gold_num = extract_number(gold)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 0.01
    return normalize_answer(pred) == normalize_answer(gold)


def match_answer(pred, gold, task_type):
    if task_type == "numerical_conflict":
        return match_numerical(pred, gold)
    return normalize_answer(gold) in normalize_answer(pred)


def is_abstention(answer, abstained_flag):
    if abstained_flag:
        return True
    if not answer:
        return False
    lower = answer.lower()
    keywords = ["cannot be determined", "insufficient",
                "not enough information", "cannot determine",
                "unable to determine", "no sufficient"]
    return any(k in lower for k in keywords)


def extract_json(text):
    if not text:
        return None
    # Strip markdown code block wrapper
    cleaned = text.strip()
    if cleaned.startswith('```'):
        # Remove opening ```json or ```
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        # Remove closing ```
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try greedy match for outermost {...}
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try non-greedy
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try with markdown stripping again
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract just answer field manually for truncated output
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if answer_match:
        return {"answer": answer_match.group(1)}
    return None

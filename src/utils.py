import json
import re
import os
import string


# --- Variant vocabulary (single source of truth) ------------------------------
# The paper refers to variants as V1..V6; the data files use descriptive names.
VARIANT_CODES = {
    "correct_front": "V1",
    "correct_middle": "V2",
    "correct_end": "V3",
    "conflict_before_correct": "V4",
    "correct_before_conflict": "V5",
    "distractor_dominant": "V6",
}
VARIANT_ORDER = ("correct_front", "correct_middle", "correct_end",
                 "conflict_before_correct", "correct_before_conflict",
                 "distractor_dominant")
# Variants intended to contain the planted wrong evidence and the correct
# evidence. In released v1, 61 generation failures have no planted wrong text;
# metric code excludes those sample ids from conflict-specific denominators.
CONFLICT_VARIANTS = ("conflict_before_correct", "correct_before_conflict")


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


# --- Record-level failure classification --------------------------------------
# API failures were serialised into the raw prediction files as JSON objects with
# an "error" key, which json.loads() happily accepts. Consequently the
# `parse_error` flag in outputs/parsed_predictions/ is False for these records
# even though no model answer exists. Any denominator over model behaviour must
# therefore be built from these classes, not from `parse_error` alone.
RECORD_OK = "ok"
RECORD_API_ERROR = "api_error"
RECORD_MODEL_EMPTY = "model_empty"
RECORD_UNPARSEABLE = "unparseable"
RECORD_CLASSES = (RECORD_OK, RECORD_API_ERROR, RECORD_MODEL_EMPTY, RECORD_UNPARSEABLE)


def classify_raw_output(raw):
    """Classify a raw model output into exactly one RECORD_* class."""
    if raw is None:
        return RECORD_MODEL_EMPTY
    text = raw.strip()
    if not text:
        return RECORD_MODEL_EMPTY
    try:
        direct = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        direct = None
    if isinstance(direct, dict) and "error" in direct:
        return RECORD_API_ERROR
    parsed = extract_json(text)
    if parsed is None:
        return RECORD_UNPARSEABLE
    # Parseable JSON whose answer field is empty carries no model answer
    # (observed for DeepSeek-Reasoner when CoT exhausts the output budget);
    # classify as model_empty so the taxonomy matches is_valid_prediction.
    if isinstance(parsed, dict) and not str(parsed.get("answer") or "").strip():
        return RECORD_MODEL_EMPTY
    return RECORD_OK


def is_valid_prediction(pred):
    """True when a parsed prediction carries an actual model answer.

    Excludes (a) parse failures -- model_empty and unparseable records, which
    parse_outputs.py flags with parse_error=True, and (b) API-error records,
    whose `answer` field ends up as the empty string.
    """
    if pred.get("parse_error"):
        return False
    return bool((pred.get("answer") or "").strip())

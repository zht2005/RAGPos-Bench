"""Generate base samples from HotpotQA, MuSiQue, and SQuAD."""
import json
import os
import random
import sys

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import save_jsonl, normalize_answer

random.seed(42)
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def extract_hotpotqa(n=1000):
    print(f"  Loading HotpotQA (target: {n})...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
    samples = []
    for item in tqdm(ds, desc="HotpotQA", total=n):
        if len(samples) >= n:
            break
        question = item["question"]
        answer = item["answer"]
        if not answer or len(answer) < 2:
            continue
        supporting_titles = set(item["supporting_facts"]["title"])
        correct_paragraphs = []
        distractor_paragraphs = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            text = " ".join(sentences).strip()
            if not text or len(text) < 20:
                continue
            if title in supporting_titles:
                correct_paragraphs.append(text)
            else:
                distractor_paragraphs.append(text)
        if not correct_paragraphs or len(distractor_paragraphs) < 3:
            continue
        correct_evidence = " ".join(correct_paragraphs)
        if normalize_answer(answer) not in normalize_answer(correct_evidence):
            continue
        samples.append({
            "sample_id": f"hotpotqa_{len(samples):04d}",
            "source": "hotpotqa",
            "question": question,
            "gold_answer": answer,
            "correct_evidence": correct_evidence,
            "distractor_evidences": distractor_paragraphs[:5],
            "metadata": {"type": "multi_hop"}
        })
    print(f"  HotpotQA: extracted {len(samples)} samples")
    return samples


def extract_musique(n=1000):
    print(f"  Loading MuSiQue (target: {n})...")
    ds = load_dataset("voidful/MuSiQue", split="train", streaming=True)
    samples = []
    for item in tqdm(ds, desc="MuSiQue", total=n):
        if len(samples) >= n:
            break
        question = item["question"]
        answer = item["answer"]
        if not answer or len(answer) < 2:
            continue
        paragraphs = item.get("paragraphs", [])
        if not paragraphs:
            continue
        correct_paragraphs = []
        distractor_paragraphs = []
        for p in paragraphs:
            text = p.get("paragraph_text", "").strip()
            if not text or len(text) < 20:
                continue
            if p.get("is_supporting", False):
                correct_paragraphs.append(text)
            else:
                distractor_paragraphs.append(text)
        if not correct_paragraphs or len(distractor_paragraphs) < 3:
            continue
        correct_evidence = " ".join(correct_paragraphs)
        if normalize_answer(answer) not in normalize_answer(correct_evidence):
            continue
        samples.append({
            "sample_id": f"musique_{len(samples):04d}",
            "source": "musique",
            "question": question,
            "gold_answer": answer,
            "correct_evidence": correct_evidence,
            "distractor_evidences": distractor_paragraphs[:5],
            "metadata": {"type": "multi_hop"}
        })
    print(f"  MuSiQue: extracted {len(samples)} samples")
    return samples


def extract_squad(n=500):
    print(f"  Loading SQuAD (target: {n})...")
    ds = load_dataset("rajpurkar/squad", split="train", streaming=True)
    samples = []
    seen_questions = set()
    for item in tqdm(ds, desc="SQuAD", total=n):
        if len(samples) >= n:
            break
        question = item["question"]
        if question in seen_questions:
            continue
        seen_questions.add(question)
        answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
        if not answer or len(answer) < 2:
            continue
        context = item["context"].strip()
        if len(context) < 50:
            continue
        samples.append({
            "sample_id": f"squad_{len(samples):04d}",
            "source": "squad",
            "question": question,
            "gold_answer": answer,
            "correct_evidence": context,
            "distractor_evidences": [],
            "metadata": {"type": "single_hop"}
        })
    print(f"  SQuAD: extracted {len(samples)} samples")
    return samples


def fill_distractors_for_squad(squad_samples, all_samples):
    """Borrow distractor paragraphs from other samples for SQuAD."""
    all_distractors = []
    for s in all_samples:
        all_distractors.extend(s.get("distractor_evidences", []))
    random.shuffle(all_distractors)
    idx = 0
    for s in squad_samples:
        if len(s["distractor_evidences"]) < 5:
            needed = 5 - len(s["distractor_evidences"])
            s["distractor_evidences"].extend(all_distractors[idx:idx+needed])
            idx = (idx + needed) % len(all_distractors)


def main():
    import yaml
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Generating base samples from real datasets...")
    hotpot = extract_hotpotqa(config["data"]["hotpotqa_samples"])
    musique = extract_musique(config["data"]["musique_samples"])
    squad = extract_squad(config["data"]["squad_samples"])

    fill_distractors_for_squad(squad, hotpot + musique)

    all_samples = hotpot + musique + squad
    random.shuffle(all_samples)

    out_path = os.path.join(BASE_DIR, 'data', 'base_samples.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_jsonl(all_samples, out_path)
    print(f"\nTotal: {len(all_samples)} base samples -> {out_path}")
    print(f"  HotpotQA: {len(hotpot)}, MuSiQue: {len(musique)}, SQuAD: {len(squad)}")


if __name__ == "__main__":
    main()

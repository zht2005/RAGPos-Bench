"""Heuristic extraction of planted wrong-answer claims from wrong evidence.

For each base sample, propose candidate answer strings that the wrong evidence
(e-dagger) asserts, so that "adopted the wrong evidence" can later be tested by
matching a model answer against these candidates (see evaluate.py EAR_new).

Pure string heuristics, deterministic, no API calls:
  candidates = capitalized multi-word spans + numbers + dates + quoted spans
  keep those NOT present in the correct evidence (e-star) and NOT
  containing/contained-in/equal to the gold answer (normalized);
  rank by (frequency in e-dagger, keyword co-occurrence with the question).

Output: data/wrong_claims.jsonl with
  {sample_id, gold, candidates (ranked, all survivors), status}
status = "decided" when at least one candidate survives, else "undecidable".
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, save_jsonl, normalize_answer

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

# Lowercase connector words allowed inside a capitalized multi-word span.
CONNECTORS = {"of", "the", "and", "for", "in", "on", "at", "de", "la", "le",
              "du", "da", "di", "van", "von", "der", "den", "el", "al", "y"}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "which", "what", "who", "whom", "whose", "when", "where", "why", "how",
    "did", "do", "does", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "that", "this", "these", "those", "it", "its",
    "their", "his", "her", "he", "she", "they", "than", "then", "there",
    "had", "has", "have", "not", "no", "but", "also", "such", "known", "called",
    "name", "named",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.\-]*")
CAP_WORD_RE = re.compile(r"^[A-Z][A-Za-z0-9'’.\-]*$")
NUMBER_RE = re.compile(r"[$£€]?\d[\d,]*(?:\.\d+)?%?")
# Matched quote pairs only (mixing open/close styles in one class would merge
# adjacent quoted spans). Straight single quotes are skipped: apostrophes.
QUOTE_RES = (
    re.compile(r'"([^"]{2,80})"'),
    re.compile(r"“([^“”]{2,80})”"),
    re.compile(r"‘([^‘’]{2,80})’"),
)
DATE_RES = (
    # April 23, 1936 / April 1936
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|"
               r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"),
    re.compile(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)\s+\d{4}\b"),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|"
               r"October|November|December)\s+\d{4}\b"),
    re.compile(r"\b(?:1[0-9]|20)\d{2}\b"),  # bare years 1000-2099
)


def capitalized_spans(text):
    """Consecutive capitalized tokens (lowercase connectors allowed inside),
    keeping only spans with >= 2 capitalized words."""
    tokens = [(m.group(0), m.start()) for m in TOKEN_RE.finditer(text)]
    spans = []
    i = 0
    while i < len(tokens):
        if CAP_WORD_RE.match(tokens[i][0]):
            j = i
            last_cap = i
            while j + 1 < len(tokens):
                nxt = tokens[j + 1][0]
                if CAP_WORD_RE.match(nxt):
                    j += 1
                    last_cap = j
                elif nxt.lower() in CONNECTORS and j + 2 < len(tokens) and \
                        CAP_WORD_RE.match(tokens[j + 2][0]):
                    j += 2
                    last_cap = j
                else:
                    break
            j = last_cap
            n_cap = sum(1 for k in range(i, j + 1) if CAP_WORD_RE.match(tokens[k][0]))
            if n_cap >= 2:
                start = tokens[i][1]
                end = tokens[j][1] + len(tokens[j][0])
                spans.append(text[start:end].strip(" .,;:"))
            i = j + 1
        else:
            i += 1
    return spans


def extract_candidates(wrong_text):
    cands = []
    cands.extend(capitalized_spans(wrong_text))
    for rx in DATE_RES:
        cands.extend(m.group(0) for m in rx.finditer(wrong_text))
    cands.extend(m.group(0) for m in NUMBER_RE.finditer(wrong_text))
    for rx in QUOTE_RES:
        cands.extend(m.group(1).strip(" .,;:") for m in rx.finditer(wrong_text))
    return [c for c in cands if c.strip()]


def question_keywords(question):
    return {t.lower() for t in TOKEN_RE.findall(question)
            if len(t) > 2 and t.lower() not in STOPWORDS}


def main():
    samples = load_jsonl(os.path.join(BASE_DIR, 'data', 'base_samples_with_wrong.jsonl'))
    out = []
    n_decided = n_undecidable = 0

    for s in samples:
        gold = s["gold_answer"]
        gold_n = normalize_answer(gold)
        estar_n = normalize_answer(s.get("correct_evidence", ""))
        wrong = s.get("wrong_evidence", "") or ""
        qkeys = question_keywords(s.get("question", ""))

        raw = extract_candidates(wrong)
        # Deduplicate on normalized form, keep first-seen surface form and
        # count frequency for ranking.
        freq = Counter()
        surface = {}
        first_pos = {}
        for pos, c in enumerate(raw):
            cn = normalize_answer(c)
            if not cn:
                continue
            freq[cn] += 1
            if cn not in surface:
                surface[cn] = c
                first_pos[cn] = pos

        survivors = []
        for cn in freq:
            # Quality gate: reject trivially short, low-information candidates
            # (e.g. a bare "2" or "04") whose containment matching produces
            # false-positive adoptions against unrelated answers.
            if len(cn) < 4:
                continue
            # Drop anything present in the correct evidence e-star.
            if estar_n and cn in estar_n:
                continue
            # Drop anything containing / contained in / equal to gold.
            if gold_n and (gold_n in cn or cn in gold_n):
                continue
            cand_tokens = {t for t in cn.split() if t not in STOPWORDS}
            overlap = len(cand_tokens & qkeys)
            survivors.append((freq[cn], overlap, -first_pos[cn], surface[cn]))

        # Rank: frequency desc, question-keyword co-occurrence desc,
        # earlier appearance first; deterministic.
        survivors.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))
        candidates = [t[3] for t in survivors]

        status = "decided" if candidates else "undecidable"
        if candidates:
            n_decided += 1
        else:
            n_undecidable += 1
        out.append({"sample_id": s["sample_id"], "gold": gold,
                    "candidates": candidates, "status": status})

    out_path = os.path.join(BASE_DIR, 'data', 'wrong_claims.jsonl')
    save_jsonl(out, out_path)
    print(f"Wrong-claim candidates saved to {out_path}")
    print(f"  samples      : {len(out)}")
    print(f"  decided      : {n_decided}")
    print(f"  undecidable  : {n_undecidable}")
    n_cands = [len(o["candidates"]) for o in out]
    if n_cands:
        import statistics
        print(f"  candidates per decided sample: mean="
              f"{sum(n_cands)/len(n_cands):.2f}, median={statistics.median(n_cands)}, "
              f"max={max(n_cands)}")


if __name__ == "__main__":
    main()

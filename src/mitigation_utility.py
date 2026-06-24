"""Mitigation utility experiment.

Two phases:
  A. Conflict-aware prompt over all 7 models on a fixed 300-sample subset
     (V4 + V5 only). Baseline is reused from outputs/parsed_predictions/.
  B. Evidence-shuffling self-consistency (3 shuffles, majority vote) on
     GPT-5.4-mini and DeepSeek-Chat over the SAME 300-sample subset.

Fixed seed = 42 for sample selection AND shuffle order, so re-running is
deterministic. Outputs go under outputs/mitigation/.

Phase B uses original (un-modified) prompt; only the order of E1..E6 is
permuted, with the shuffle mapping persisted so we can map
selected_evidence_ids back to original positions before scoring.

Usage:
  python3 src/mitigation_utility.py --phase a   # all-model conflict-aware
  python3 src/mitigation_utility.py --phase b   # 2-model shuffle
  python3 src/mitigation_utility.py --phase aggregate  # combine + bootstrap
  python3 src/mitigation_utility.py --phase dryrun     # 1 call per endpoint
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

BASE = ".."
sys.path.insert(0, os.path.join(BASE, "src"))
from utils import load_jsonl, save_jsonl, match_answer, normalize_answer  # noqa

OUT_DIR = os.path.join(BASE, "outputs/mitigation")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
SAMPLE_N = 300
SHUFFLES = 3

CHAT_MODELS = [
    {"model_id": "gpt-5.4-medium",            "base_url": "https://www.right.codes/codex/v1",      "key_env": "OPENAI_API_KEY"},
    {"model_id": "gpt-5.4-mini",              "base_url": "https://www.right.codes/codex/v1",      "key_env": "OPENAI_API_KEY"},
    {"model_id": "claude-haiku-4-5-20251001", "base_url": "https://www.right.codes/claude-aws/v1", "key_env": "OPENAI_API_KEY"},
    {"model_id": "claude-sonnet-4-6",         "base_url": "https://www.right.codes/claude-aws/v1", "key_env": "OPENAI_API_KEY"},
    {"model_id": "gemini-2.5-flash",          "base_url": "https://www.right.codes/gemini/v1",     "key_env": "OPENAI_API_KEY"},
    {"model_id": "deepseek-chat",             "base_url": "https://api.deepseek.com/v1",           "key_env": "DEEPSEEK_API_KEY"},
    {"model_id": "deepseek-reasoner",         "base_url": "https://api.deepseek.com/v1",           "key_env": "DEEPSEEK_API_KEY"},
]
SHUFFLE_MODELS = [m for m in CHAT_MODELS if m["model_id"] in ("gpt-5.4-mini", "deepseek-chat")]


CONFLICT_AWARE_PRELUDE = """## Reasoning instructions
1. Carefully compare all evidence passages before answering.
2. Evidence order should not determine credibility.
3. If two passages conflict, identify the conflict explicitly.
4. Do not simply follow the first passage if later evidence contradicts it.
5. If a conflict is found, set has_conflict to true.
6. selected_evidence_ids must include the passage that supports the final answer.

"""

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


def build_prompt(question, evs_by_id, conflict_aware=False):
    body = PROMPT_TEMPLATE.format(
        question=question,
        E1=evs_by_id["E1"], E2=evs_by_id["E2"], E3=evs_by_id["E3"],
        E4=evs_by_id["E4"], E5=evs_by_id["E5"], E6=evs_by_id["E6"],
    )
    if conflict_aware:
        # Insert prelude BEFORE the question paragraph to keep schema instructions intact
        idx = body.find("Question:")
        return body[:idx] + CONFLICT_AWARE_PRELUDE + body[idx:]
    return body


def select_subset(insts):
    """Return list of (sample_id, V4_iid, V5_iid) for SAMPLE_N base samples."""
    by_sample = defaultdict(dict)
    for i in insts:
        by_sample[i["sample_id"]][i["variant"]] = i["instance_id"]
    eligible = [(sid, m["conflict_before_correct"], m["correct_before_conflict"])
                for sid, m in by_sample.items()
                if "conflict_before_correct" in m and "correct_before_conflict" in m]
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    return eligible[:SAMPLE_N]


def make_client(model_cfg):
    from openai import OpenAI
    api_key = os.environ.get(model_cfg["key_env"])
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=model_cfg["base_url"])


def call_model(client, model_id, prompt, max_retries=3, max_tokens=512):
    # DeepSeek-Reasoner needs more tokens because chain-of-thought eats the budget.
    if "reasoner" in model_id.lower():
        max_tokens = 2048
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=90,
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content
            last_err = "empty content"
        except Exception as e:
            last_err = str(e)[:200]
            # 503 channel-unavailable: longer back-off
            sleep = 5.0 if "503" in last_err or "No available" in last_err else (1.5 ** attempt)
            time.sleep(sleep)
    return json.dumps({"error": last_err or "unknown"})


def parse_json(text):
    """Forgiving JSON extractor (mirrors utils.extract_json behaviour)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        import re
        t = re.sub(r"^```(?:json)?\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception:
        import re
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def dryrun():
    print("=== Dry-run: 1 call per endpoint ===")
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    subset = select_subset(insts.values())
    sid, iv4, iv5 = subset[0]
    inst = insts[iv4]
    for cfg in CHAT_MODELS:
        client = make_client(cfg)
        if client is None:
            print(f"  [{cfg['model_id']}] SKIP: env {cfg['key_env']} not set")
            continue
        t0 = time.time()
        prompt = build_prompt(inst["question"], inst["evidences"], conflict_aware=True)
        out = call_model(client, cfg["model_id"], prompt, max_tokens=384)
        d = parse_json(out)
        ok = bool(d and "answer" in d)
        print(f"  [{cfg['model_id']}] {'OK' if ok else 'FAIL'}  {time.time()-t0:.1f}s "
              f"answer={(d or {}).get('answer','')[:60] if ok else out[:80]!r}")


def phase_a(target_models=None):
    """All-model conflict-aware over 300 V4+V5 (600 instances per model)."""
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    subset = select_subset(insts.values())
    work = [iv4 for _, iv4, _ in subset] + [iv5 for _, _, iv5 in subset]
    print(f"=== Phase A: conflict-aware prompt | {len(work)} instances per model ===")

    models = [m for m in CHAT_MODELS if (target_models is None or m["model_id"] in target_models)]
    for cfg in models:
        client = make_client(cfg)
        if client is None:
            print(f"  [{cfg['model_id']}] SKIP: env {cfg['key_env']} not set")
            continue
        out_path = os.path.join(OUT_DIR, f"phase_a__{cfg['model_id']}.jsonl")
        done = set()
        results = []
        if os.path.exists(out_path):
            results = list(load_jsonl(out_path))
            done = {r["instance_id"] for r in results}
        todo = [iid for iid in work if iid not in done]
        print(f"  [{cfg['model_id']}] todo={len(todo)} done={len(done)}")
        t0 = time.time()
        for i, iid in enumerate(todo):
            inst = insts[iid]
            prompt = build_prompt(inst["question"], inst["evidences"], conflict_aware=True)
            raw = call_model(client, cfg["model_id"], prompt)
            results.append({
                "instance_id": iid, "sample_id": inst["sample_id"],
                "variant": inst["variant"], "model": cfg["model_id"],
                "condition": "conflict_aware", "raw_output": raw,
            })
            if (i + 1) % 50 == 0 or i == len(todo) - 1:
                save_jsonl(results, out_path)
                print(f"    [{cfg['model_id']}] {i+1}/{len(todo)} elapsed={time.time()-t0:.0f}s")
        save_jsonl(results, out_path)


def phase_b():
    """Evidence shuffle + self-consistency for 2 models, 3 shuffles each."""
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    subset = select_subset(insts.values())
    work = [iv4 for _, iv4, _ in subset] + [iv5 for _, _, iv5 in subset]
    print(f"=== Phase B: 3-shuffle self-consistency | {len(work)} instances x {SHUFFLES} ===")
    rng = random.Random(SEED)
    shuffle_seeds = {iid: [rng.randint(0, 1 << 30) for _ in range(SHUFFLES)] for iid in work}
    for cfg in SHUFFLE_MODELS:
        client = make_client(cfg)
        if client is None:
            print(f"  [{cfg['model_id']}] SKIP: env {cfg['key_env']} not set")
            continue
        out_path = os.path.join(OUT_DIR, f"phase_b__{cfg['model_id']}.jsonl")
        done = set()
        results = []
        if os.path.exists(out_path):
            results = list(load_jsonl(out_path))
            done = {(r["instance_id"], r["shuffle_idx"]) for r in results}
        t0 = time.time()
        n_total = len(work) * SHUFFLES
        n_done_existing = len(done)
        n = 0
        for iid in work:
            inst = insts[iid]
            for k in range(SHUFFLES):
                if (iid, k) in done:
                    continue
                local_rng = random.Random(shuffle_seeds[iid][k])
                slots = ["E1","E2","E3","E4","E5","E6"]
                shuffled = slots[:]
                local_rng.shuffle(shuffled)
                # mapping: for new slot Ei, original was mapping[Ei]
                mapping = {f"E{j+1}": shuffled[j] for j in range(6)}
                evs_after = {f"E{j+1}": inst["evidences"][shuffled[j]] for j in range(6)}
                prompt = build_prompt(inst["question"], evs_after, conflict_aware=False)
                raw = call_model(client, cfg["model_id"], prompt)
                results.append({
                    "instance_id": iid, "sample_id": inst["sample_id"],
                    "variant": inst["variant"], "model": cfg["model_id"],
                    "condition": "shuffle", "shuffle_idx": k,
                    "shuffle_mapping": mapping, "raw_output": raw,
                })
                n += 1
                if n % 50 == 0 or (n + n_done_existing) == n_total:
                    save_jsonl(results, out_path)
                    print(f"    [{cfg['model_id']}] {n+n_done_existing}/{n_total} "
                          f"elapsed={time.time()-t0:.0f}s")
        save_jsonl(results, out_path)


def aggregate():
    """Compute per-model Acc, PBR, CAR, EAR, CEU on the 300-sample subset
    for original baseline (from outputs/parsed_predictions/) and for each
    mitigation condition. Apply paired bootstrap.
    """
    insts = {i["instance_id"]: i for i in load_jsonl(os.path.join(BASE, "data/eval_instances.jsonl"))}
    subset = select_subset(insts.values())
    subset_iids_v4 = {iv4 for _, iv4, _ in subset}
    subset_iids_v5 = {iv5 for _, _, iv5 in subset}
    subset_iids = subset_iids_v4 | subset_iids_v5

    def metrics_from_records(records):
        acc_n = pbr_n = car_n = ear_n = ceu_n = 0
        acc = pbr = car = ear = ceu = 0
        per_sample_acc = defaultdict(list)
        for r in records:
            iid = r["instance_id"]
            inst = insts.get(iid)
            if not inst:
                continue
            d = parse_json(r["raw_output"]) or {}
            ans = (d.get("answer") or "")
            sel = d.get("selected_evidence_ids") or []
            hc = d.get("has_conflict")
            correct = match_answer(ans, inst["gold_answer"], "general")
            wrong_pos = inst.get("wrong_evidence_position")
            wrong_text = inst["evidences"].get(wrong_pos, "") if wrong_pos else ""
            cor_pos = inst.get("correct_evidence_position")

            acc_n += 1; acc += int(correct)
            per_sample_acc[inst["sample_id"]].append(int(correct))

            if inst["variant"] == "conflict_before_correct":
                pbr_n += 1
                if not correct and ans.strip():
                    pbr += 1

            if inst["variant"] in ("conflict_before_correct", "correct_before_conflict"):
                ear_n += 1
                if not correct and ans.strip():
                    ear += 1
                car_n += 1
                if hc is True and cor_pos in sel:
                    car += 1

            if cor_pos:
                ceu_n += 1
                if cor_pos in sel:
                    ceu += 1
        f = lambda a, n: a / n if n else 0.0
        return {
            "n": acc_n, "Acc": f(acc, acc_n), "PBR": f(pbr, pbr_n),
            "CAR": f(car, car_n), "EAR": f(ear, ear_n), "CEU": f(ceu, ceu_n),
            "per_sample_acc": dict(per_sample_acc),
        }

    rows = []
    for cfg in CHAT_MODELS:
        m = cfg["model_id"]
        # Baseline: pull from parsed_predictions filtered to subset
        base_path = os.path.join(BASE, f"outputs/parsed_predictions/{m}.jsonl")
        if not os.path.exists(base_path):
            print(f"[skip-baseline] {m}: no parsed file")
            continue
        base_recs = [{"instance_id": p["instance_id"],
                      "raw_output": json.dumps(p)}
                     for p in load_jsonl(base_path)
                     if p["instance_id"] in subset_iids]
        # Note: parsed_predictions stores already-parsed fields; rebuild raw json so parse_json works
        base_metrics = metrics_from_records(base_recs)

        # Conflict-aware
        ca_path = os.path.join(OUT_DIR, f"phase_a__{m}.jsonl")
        ca_metrics = metrics_from_records(load_jsonl(ca_path)) if os.path.exists(ca_path) else None

        # Shuffle (only for the 2 representative models)
        sh_path = os.path.join(OUT_DIR, f"phase_b__{m}.jsonl")
        sh_metrics = None
        if os.path.exists(sh_path):
            # majority vote over 3 shuffles
            by_iid = defaultdict(list)
            for r in load_jsonl(sh_path):
                by_iid[r["instance_id"]].append(r)
            agg = []
            for iid, group in by_iid.items():
                ans_votes = []
                hc_votes = []
                sel_remapped = []
                for g in group:
                    d = parse_json(g["raw_output"]) or {}
                    ans = (d.get("answer") or "").strip()
                    if ans:
                        ans_votes.append(ans)
                    hc = d.get("has_conflict")
                    if isinstance(hc, bool):
                        hc_votes.append(hc)
                    sel = d.get("selected_evidence_ids") or []
                    mapping = g.get("shuffle_mapping") or {}
                    sel_remapped.extend(mapping.get(s, s) for s in sel)
                # majority answer (by normalized form)
                norm_to_count = defaultdict(int)
                norm_to_orig = {}
                for a in ans_votes:
                    n = normalize_answer(a)
                    norm_to_count[n] += 1
                    norm_to_orig.setdefault(n, a)
                final_ans = max(norm_to_orig.items(), key=lambda kv: norm_to_count[kv[0]])[1] if norm_to_orig else ""
                final_hc = sum(hc_votes) >= 2 if len(hc_votes) >= 2 else (any(hc_votes) if hc_votes else False)
                # selected: any slot that appeared in any shuffle (union)
                final_sel = list(set(sel_remapped))
                synth = {"answer": final_ans, "selected_evidence_ids": final_sel,
                         "has_conflict": final_hc, "abstained": False}
                agg.append({"instance_id": iid, "raw_output": json.dumps(synth)})
            sh_metrics = metrics_from_records(agg)

        rows.append({"model": m, "baseline": base_metrics,
                     "conflict_aware": ca_metrics, "shuffle": sh_metrics})

    out_csv = os.path.join(OUT_DIR, "mitigation_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "n", "Acc", "PBR", "CAR", "EAR", "CEU"])
        for r in rows:
            base = r["baseline"]
            w.writerow([r["model"], "baseline", base["n"], f"{base['Acc']:.4f}",
                        f"{base['PBR']:.4f}", f"{base['CAR']:.4f}",
                        f"{base['EAR']:.4f}", f"{base['CEU']:.4f}"])
            if r["conflict_aware"]:
                ca = r["conflict_aware"]
                w.writerow([r["model"], "conflict_aware", ca["n"], f"{ca['Acc']:.4f}",
                            f"{ca['PBR']:.4f}", f"{ca['CAR']:.4f}",
                            f"{ca['EAR']:.4f}", f"{ca['CEU']:.4f}"])
            if r["shuffle"]:
                sh = r["shuffle"]
                w.writerow([r["model"], "shuffle_majority", sh["n"], f"{sh['Acc']:.4f}",
                            f"{sh['PBR']:.4f}", f"{sh['CAR']:.4f}",
                            f"{sh['EAR']:.4f}", f"{sh['CEU']:.4f}"])
    print(f"[ok] summary -> {out_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dryrun", "a", "b", "aggregate"], required=True)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    if args.phase == "dryrun":
        dryrun()
    elif args.phase == "a":
        phase_a(target_models=args.models)
    elif args.phase == "b":
        phase_b()
    elif args.phase == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()

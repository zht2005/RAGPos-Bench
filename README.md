# RAGPos-Bench

> Diagnosing evidence-layout sensitivity in retrieval-augmented generation.

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)

## Overview

RAGPos-Bench is a diagnostic benchmark for retrieval-augmented generation
(RAG). It contains 2,500 base samples from HotpotQA, MuSiQue, and SQuAD and six
released layouts per sample, producing 15,000 evaluation instances.

The released evaluation contains 105,000 real API predictions from seven model
configurations queried in June 2026:

- `gpt-5.4-medium`
- `gpt-5.4-mini`
- `claude-haiku-4-5-20251001`
- `claude-sonnet-4-6`
- `deepseek-chat`
- `deepseek-reasoner`
- `gemini-2.5-flash`

## Released-v1 Construction Notice

An audit performed for the camera-ready release found that the original v1
layout generator independently shuffled five distractors and selected four for
each layout. The released variants share the same base question, gold passage,
and (when generation succeeded) planted wrong passage, but most do **not** have
the same realized distractor content or ordering. Therefore the full-set
V1/V3 and V4/V5 comparisons are descriptive layout comparisons, not isolated
causal estimates of gold-passage position or conflict order.

- 2,439/2,500 base samples have non-empty planted wrong evidence.
- 574/2,500 V4/V5 pairs have the same six-passage multiset; 513 also have a
  non-empty planted wrong passage.
- Only 21/2,500 V4/V5 pairs differ solely by swapping E2 and E5 while all four
  other slots remain identical.

The released data and predictions are retained unchanged for reproducibility.
`src/construction_audit.py` reproduces these counts and the 513-pair
matched-content sensitivity analysis. `src/build_controlled_variants.py`
implements a corrected construction for a future model rerun; v1 predictions
must not be reused with those new prompts. Three valid samples contain fewer
than five distractors; the corrected builder deterministically cycles their
existing distractors and records the number of reused slots in
`reused_distractor_count` instead of inventing new evidence.

## Important Evaluation Notes

- Accuracy treats API errors, empty answers, and unparseable outputs as errors.
- PBR, CAR, EAR, and CEU are conditional on valid, non-empty parsed responses.
- Overall, 32.6% of records are invalid under the shared 512-token JSON-output
  protocol. Valid-response rates vary substantially by model.
- DeepSeek-Reasoner's main result is a budget-constrained baseline. On a
  deterministic 498-instance pilot, raising `max_tokens` from 512 to 2,048
  increased accuracy from 0.20 to 0.54 and reduced empty parsed answers from
  60.0% to 3.4%.
- CAR and CEU use model-reported conflict flags and evidence IDs. They are
  schema-level diagnostics, not independently verified reasoning judgments.
- PBR is a descriptive V4 wrong-claim adoption rate. The paired order statistic
  is `adopt(V4) - adopt(V5)`.

Formal definitions and denominator rules are in
[`paper_assets/metrics_specification.md`](paper_assets/metrics_specification.md).

## Paper-Aligned Results

| Model | Acc | Valid | PSR | PBR | CAR | EAR | CEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.4-medium | 0.736 | 0.928 | 0.102 | 0.135 | 0.812 | 0.129 | 0.967 |
| DeepSeek-Chat | 0.539 | 0.968 | 0.283 | 0.308 | 0.521 | 0.267 | 0.817 |
| GPT-5.4-mini | 0.511 | 0.967 | 0.319 | 0.342 | 0.263 | 0.334 | 0.722 |
| Claude-Haiku-4.5 | 0.459 | 0.763 | 0.226 | 0.187 | 0.365 | 0.211 | 0.910 |
| Claude-Sonnet-4.6 | 0.427 | 0.500 | 0.056 | 0.070 | 0.818 | 0.082 | 0.984 |
| DeepSeek-Reasoner | 0.231 | 0.399 | 0.259 | 0.247 | 0.402 | 0.243 | 0.688 |
| Gemini-2.5-Flash | 0.129 | 0.194 | 0.131 | 0.212 | 0.275 | 0.192 | 0.441 |

These values are reproduced by `outputs/metrics/overall_metrics.csv`. The
conditional columns must be interpreted together with the valid-response rate.

Key findings under the fixed protocol:

1. Full-set V1/V3 differences are significant at `p < 0.01` for five models,
   but combine gold/wrong-passage relocation with distractor resampling.
2. On the 513 content-matched, non-empty-wrong V4/V5 pairs, only
   Claude-Haiku-4.5 is significant at `p < 0.01` (V4 minus V5 = `+0.072`).
   Because distractor slots can still differ, this is an overall layout effect,
   not an isolated conflict-order estimate.
3. MuSiQue is the hardest source for all seven models, averaging about 0.30
   below SQuAD.
4. Between 7.8% and 54.6% of correct valid answers on eligible conflict variants are
   produced without a reported conflict flag.
5. In the mitigation utility study, conflict-aware prompting significantly
   raises accuracy for six models and CAR for five. V4 wrong-claim adoption and
   EAR fall significantly for DeepSeek-Chat and GPT-5.4-mini.

## Repository Layout

```text
RAGPos-Bench/
|-- data/
|   |-- base_samples_with_wrong.jsonl  # construction-stage base records
|   |-- eval_instances.jsonl           # 15,000 benchmark instances
|   `-- wrong_claims.jsonl              # deterministic wrong-claim candidates
|-- outputs/
|   |-- raw_predictions/                # 7 x 15,000 original API outputs
|   |-- parsed_predictions/             # corrected structured predictions
|   |-- metrics/                        # paper-aligned aggregate results
|   |-- mitigation/                     # mitigation runs and paired deltas
|   `-- reasoner_pilot/                 # 2,048-token Reasoner pilot
|-- figures/
|   |-- fig1_framework_overview.*
|   `-- fig3_position_slope.*
|-- paper_assets/
|   |-- metrics_specification.md
|   |-- case_studies.json
|   `-- wrong_evidence_qc_sample100.csv
`-- src/                               # construction and evaluation pipeline
```

## Reproducing the Released Metrics

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Rebuild every paper-aligned metric from the released raw predictions:

```bash
python src/parse_outputs.py
python src/extract_wrong_claims.py
python src/evaluate.py
python src/failure_taxonomy.py
python src/significance.py
python src/construction_audit.py
python src/psr_audit.py
python src/mitigation_analysis.py
python src/beautify_fig3.py
python src/verify_release.py
```

The scripts use fixed seeds where resampling or subset selection is involved.
`significance.py` performs two-sided paired bootstrap tests over base samples
with 10,000 resamples and a paper threshold of `p < 0.01`.

Re-running model inference is optional and requires provider API keys. Endpoints
and environment-variable names are defined in `src/run_models.py` and
`src/mitigation_utility.py`; no credentials are stored in this repository.

## Dataset Quality Control

Wrong-evidence generation succeeded for 2,439/2,500 base samples. Deterministic
rules pass 90.4% of those passages on plausibility, topicality, and
contradiction jointly. The saved GPT-5.4-mini re-judgment contains 532 relevant
records after excluding failed generations. This is not a substitute for
independent human double-annotation.

## Citation

```bibtex
@inproceedings{zhang2026ragpos,
  title     = {RAGPos-Bench: Diagnosing Evidence Position Bias in Retrieval-Augmented Generation},
  author    = {Zhang, Hantian and Zeng, Linghang and Wu, Wentai},
  booktitle = {Pacific Rim International Conference on Artificial Intelligence (PRICAI)},
  year      = {2026}
}
```

## License and Source Data

Code is released under the MIT License. Source datasets retain their upstream
licenses: HotpotQA and SQuAD use CC BY-SA 4.0, and MuSiQue uses CC BY 4.0.
Repository-generated annotations and evaluation outputs are provided for
research use subject to those upstream terms.

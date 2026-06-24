# RAGPos-Bench

> Diagnosing evidence position bias in retrieval-augmented generation via controlled evidence-layout interventions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

**RAGPos-Bench** is a controlled diagnostic benchmark for retrieval-augmented
generation (RAG) systems. From 2,500 base samples drawn from HotpotQA,
MuSiQue, and SQuAD, we construct **15,000 evaluation instances** by emitting
six position-controlled evidence-layout variants per sample. The variants
hold question and evidence content fixed while permuting only the order of
the supporting and misleading passages, so any per-model output difference
is attributable to **the layout intervention rather than to sample
difficulty**.

We evaluate **seven recent LLMs** (GPT-5.4-medium, GPT-5.4-mini,
Claude-Haiku-4.5, Claude-Sonnet-4.6, DeepSeek-Chat, DeepSeek-Reasoner,
Gemini-2.5-Flash) under a unified JSON-output protocol, yielding
**105,000 real-API predictions**. The benchmark exposes:

- *Position bias* — does the model use the gold passage less when it sits at the end?
- *Conflict-order effects* — does the answer flip when correct and misleading evidence trade slots?
- *Conflict awareness* — does the model report the contradiction it just resolved?

Six diagnostic metrics decouple answer correctness from evidence handling:
**Acc, PSR, PBR, CAR, EAR, CEU** (see `paper_assets/metrics_specification.md`
for the formal definitions).

## Repository layout

```
RAGPos-Bench/
├── README.md                                  this file
├── LICENSE                                    MIT for code, CC-BY-4.0 for data
├── requirements.txt
├── data/
│   └── eval_instances.jsonl                   15,000 evaluation instances
├── outputs/
│   ├── raw_predictions/  *.jsonl              7 × 15,000 raw model outputs (real API)
│   ├── parsed_predictions/ *.jsonl            JSON-parsed predictions
│   ├── metrics/                               aggregate CSVs (overall, per-source, per-variant, position, significance)
│   └── mitigation/                            mitigation utility study outputs
├── figures/                                   paper figures (PDF + PNG)
├── paper_assets/
│   ├── case_studies.json                      4 qualitative case studies
│   └── wrong_evidence_qc_sample100.csv        100-row human-inspection sample
└── src/                                       analysis pipeline
    ├── generate_data.py                       build base samples from HuggingFace datasets
    ├── build_variants.py                      6-variant layout generator
    ├── run_models.py                          unified OpenAI-compatible inference
    ├── parse_outputs.py                       JSON extraction
    ├── evaluate.py                            6-metric computation
    ├── significance.py                        paired bootstrap
    ├── mitigation_utility.py                  conflict-aware prompt + shuffle protocols
    ├── plot_paper_figures.py                  Figures 2-7
    └── ...
```

## Reproducibility

### Re-deriving the published metrics

All numbers in the paper are re-derivable from the shipped raw predictions:

```bash
pip install -r requirements.txt
python src/parse_outputs.py     # parse raw -> JSON
python src/evaluate.py          # produce outputs/metrics/*.csv
python src/significance.py      # paired bootstrap
python src/plot_paper_figures.py
```

### Re-running inference (optional, requires API keys)

```bash
export OPENAI_API_KEY=...        # for the OpenAI-compatible proxy
export DEEPSEEK_API_KEY=...      # for api.deepseek.com
export GOOGLE_API_KEY=...        # for the official Gemini API (optional)
python src/run_models.py
```

API endpoints are configured in `src/run_models.py` and `src/mitigation_utility.py`.

## Key findings (paper, June 2026)

1. **Position bias is statistically significant in 3 / 7 models** (paired
   bootstrap, p<0.05); the largest drop is 12 points in DeepSeek-Chat when
   the gold passage moves from $E_1$ to $E_6$.
2. **Wrong evidence placed before the correct evidence flips answers.**
   The V4-vs.-V5 contrast (identical content, reversed order) is significant
   in 5/7 models.
3. **MuSiQue is uniformly the hardest source** — a 30-point gap from SQuAD.
4. **DeepSeek-Reasoner is uniformly worse than DeepSeek-Chat** on every
   metric, even after accounting for output-budget constraints.
5. **High accuracy does not imply high conflict awareness:** GPT-5.4-medium
   is the most accurate model overall but is the largest example of the
   "right answer / no conflict flag" auditing gap.
6. **Mitigation utility study:** a one-paragraph conflict-aware prompt
   raises CAR significantly on all 7 models, and additionally reduces PBR
   and EAR on Gemini-2.5-Flash and GPT-5.4-mini, demonstrating that
   RAGPos-Bench can quantify intervention effects beyond final-answer
   accuracy.

## Dataset quality control

The 2,500 generated wrong passages were audited along three axes
(plausible / topical / contradictory) by both deterministic rules
(99.8% / 94.1% / 95.3% pass; 89.3% intersection) and an independent
LLM judge over a 568-sample stratified subset (LLM agrees with rules on
99.0% of rule-passes; rescues 82.5% of rule-failures as actually
compliant). Per-sample audit labels are shipped alongside the data.

## Citation

```bibtex
@inproceedings{anonymous2026ragpos,
  title={RAGPos-Bench: Diagnosing Evidence Position Bias in Retrieval-Augmented Generation},
  author={Anonymous Authors},
  booktitle={Pacific Rim International Conference on Artificial Intelligence (PRICAI)},
  year={2026}
}
```

## License

- **Code (`src/`)**: MIT License (see `LICENSE`).
- **Data (`data/`, `outputs/`)**: derived from HotpotQA (CC BY-SA 4.0),
  MuSiQue (CC BY 4.0), and SQuAD (CC BY-SA 4.0); the layout-perturbation
  derivatives in this repository are released under **CC BY 4.0**.

# Metric Specification

This document records the definitions used in the PRICAI 2026 camera-ready
paper and in the default evaluation scripts.

## Response Validity

A prediction is valid when it contains a parseable, non-empty answer. API
errors, empty model outputs, JSON with `answer: null`, and unparseable outputs
are invalid.

- Accuracy and PSR include all 15,000 instances per model. Invalid predictions
  are scored as incorrect.
- PBR, CAR, EAR, and CEU use valid responses only. CAR additionally excludes
  the 61 base samples for which wrong-evidence generation failed.

The conditional metrics must be read together with each model's valid-response
rate in `outputs/metrics/overall_metrics.csv`.

## Metrics

### Accuracy (Acc, higher is better)

The normalized gold answer must occur in the normalized prediction. Matching
lowercases and trims text, removes punctuation except periods, and collapses
whitespace. It does not canonicalize aliases or dates and does not award
partial credit for multi-span answers.

### Position-Layout Sensitivity Rate (PSR, lower is better)

For each base sample, PSR records whether the binary correctness indicator
changes across released-v1 layouts V1, V2, and V3, where the correct passage
appears at E1, E3, and E6. The v1 generator also independently samples and
orders distractors for each layout, so this is a position-layout sensitivity
measure rather than a pure position effect.

### Error Adoption Rate (EAR, lower is better)

EAR is evaluated on V4 and V5. A valid prediction counts as adoption only when
it is not gold-correct and matches a deterministic candidate claim extracted
from the planted wrong passage. The extractor uses capitalized spans, quoted
spans, numbers, and dates, then removes candidates present in the correct
evidence or overlapping the gold answer.

The 496 samples without an extractable wrong claim are excluded, leaving 2,004
base samples eligible before model-specific validity filtering.

### Wrong-First Adoption Rate (PBR, lower is better)

PBR applies the EAR adoption test to V4, where the wrong passage precedes the
correct passage. PBR is descriptive and does not alone identify an order effect
because it also reflects passage persuasiveness.

The paired order statistic is:

```text
adopt(V4) - adopt(V5)
```

It is computed on samples for which both responses are valid and a wrong claim
is extractable.

### Conflict Arbitration Rate (CAR, higher is better)

On valid V4/V5 predictions, CAR is the fraction for which the model both sets
`has_conflict` to true and includes the correct evidence position in
`selected_evidence_ids`. Samples without a non-empty planted wrong passage are
excluded.

### Correct Evidence Usage (CEU, higher is better)

CEU is the fraction of valid predictions that include the correct evidence
position in `selected_evidence_ids`.

CAR and CEU are schema-level, self-reported diagnostics. They do not
independently verify grounding or the model's reasoning process.

## Construction Audit and Significance Tests

The released v1 generator independently shuffled five distractors and selected
four for each layout. Consequently, most paired layouts do not contain the same
realized evidence multiset. The audit in
`outputs/metrics/construction_audit.csv` finds:

- 2,439/2,500 samples have non-empty planted wrong evidence;
- 574/2,500 V4/V5 pairs have the same evidence multiset;
- 513 of those 574 also have non-empty planted wrong evidence; and
- only 21/2,500 V4/V5 pairs differ solely at E2/E5 while every other slot is
  identical.

Full-set V1/V3, V1/V4, and V4/V5 comparisons use a two-sided paired bootstrap
over 2,500 aligned base samples with 10,000 resamples and seed 42, but they are
descriptive comparisons of the released layouts and not content-controlled
causal tests. `outputs/metrics/content_matched_v4_v5.csv` reports the same test
on the 513 V4/V5 pairs with matching evidence multisets and non-empty wrong
evidence. Even there, distractor slot order can differ, so the test identifies
overall layout sensitivity rather than the isolated effect of swapping only
the correct and wrong passages.

Mitigation comparisons use paired bootstrap tests over the deterministic
300-base-sample V4/V5 subset with 1,000 resamples and mark `p < 0.05`.

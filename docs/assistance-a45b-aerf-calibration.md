# Phase 4 A4.5b — AERF implementation binding and calibration-only execution

## Purpose

A4.5b binds one authoritative implementation of Atom-Evidence Relation Factorization (AERF) and authorizes inference only on the 40 A4.5a calibration units. It does not authorize fresh-validation scoring, confirmatory inspection, candidate-model comparison, or post-result rescue.

The A4.4d failure showed that generic three-way NLI neutral was not a valid proxy for epistemic absence of evidence in this task. A4.5b therefore preserves the A4.4e factorization: evidence alignment/relevance and evidence sufficiency are resolved before support/refutation polarity can produce a semantic relation.

## Literature rationale

The design follows the distinction made explicit in FEVER: claims may be supported, refuted, or have not enough information, with evidence extraction forming a separate part of the verification problem. See Thorne et al. (2018), *FEVER: a Large-scale Dataset for Fact Extraction and VERification*, https://arxiv.org/abs/1803.05355.

The sufficiency gate is also motivated by Atanasova et al. (2022), *Fact Checking with Insufficient Evidence*, https://arxiv.org/abs/2204.02007, which studies the need to detect when retrieved evidence is insufficient rather than forcing a veracity judgment.

The authoritative polarity model is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, trained on MultiNLI, FEVER-NLI, and ANLI. It is used only after a minimal evidence span has been selected and is not allowed to map its native neutral class directly to AERF `UNKNOWN`.

The relevance/alignment model is `cross-encoder/ms-marco-MiniLM-L6-v2`, already pinned in the Helix Phase 3 retrieval stack. It scores each claim against individual evidence sentences and deterministically selects the highest-scoring sentence, with the earliest sentence winning an exact tie.

HHEM was considered as a useful factual-consistency signal but not selected as the authoritative primary relation layer. Its public model card defines hallucination/factual inconsistency primarily as lack of support by the evidence. That is valuable for support checking but does not by itself provide the explicit three-way distinction AERF requires between refutation and insufficient or irrelevant evidence. See https://huggingface.co/vectara/hallucination_evaluation_model.

## Frozen implementation

The AERF execution order is:

1. split each evidence document into punctuation-retaining sentences;
2. score every claim/sentence pair with the frozen MS MARCO cross-encoder;
3. choose the highest-scoring sentence as the minimal candidate evidence span;
4. apply the frozen relevance threshold to the selected span score;
5. run the frozen FEVER+ANLI DeBERTa model on selected evidence span as premise and claim as hypothesis;
6. define sufficiency strength as `max(P(entailment), P(contradiction))`;
7. if relevance fails, emit `UNKNOWN`;
8. if relevance passes but sufficiency fails, emit `UNKNOWN`;
9. if both pass, use entailment versus contradiction probability to emit `ENTAILED` or `CONTRADICTED`;
10. apply deterministic claim composition and existing safety gates.

Native NLI neutral is retained in the raw probability artifact for forensic analysis but is never directly translated to `UNKNOWN`.

## Frozen model identities

### Alignment and relevance

- model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- revision: `c5f2b386de279a97c53a702dd5189d1c407160dc`
- weights: `model.safetensors`
- SHA256: `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`
- license: Apache-2.0

### Sufficiency and polarity

- model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- revision: `0e2603d5d3d3ef9b2910814b34eebe1a2101da65`
- weights: `model.safetensors`
- SHA256: `06d6fd89edd4f97816831626daafbdb0b029cf63bae8edc0bccab1d64e2e7707`
- native labels: 0 entailment, 1 neutral, 2 contradiction
- license: MIT

The workflow independently hashes the downloaded weight files before any scientific inference.

## Calibration partition

A4.5b materializes only the 40 A4.5a calibration units:

- 360 semantic component pairs;
- 360 deterministic claim-composition cases;
- pair SHA256 `2339b1328c1dd2854e712862f5e4183300581997a22a7000db2629313a49092f`;
- claim SHA256 `e04813b1aec29335ffdbfbd8baa2e6f021ca6aa6a64625c255b454139e948ee5`.

The calibration-only materializer contains no fresh-validation unit identifiers and reproduces the exact A4.5a calibration hashes before inference.

## Threshold calibration

Two thresholds are jointly fit on calibration evidence only:

- relevance raw logit: `-12.0` through `12.0` in increments of `0.1`, 241 values;
- sufficiency probability: `0.50` through `0.99` in increments of `0.01`, 50 values.

This creates exactly 12,050 preregistered threshold pairs. No temperature, class-specific threshold, model search, or validation feedback is permitted.

A feasible threshold pair must satisfy every A4.5b calibration-readiness requirement. If multiple are feasible, selection follows the frozen lexicographic rule in `assistance_grounding_a45b_v1.json`. If none are feasible, the deterministic best diagnostic pair is still recorded, but the scientific status is a failure and fresh validation remains unauthorized.

The calibration-readiness gate uses 0.90 component/relationship floors and 0.05 maximum failure-directed false-contradiction rates. These are not the final validity requirements. The harder A4.5a fresh-validation requirements remain unchanged and are not evaluated in A4.5b.

## Sealed evidence boundary

Throughout A4.5b:

- fresh-validation units: 20, unopened for scoring;
- fresh-validation component pairs: 180, unscored;
- fresh-validation claim cases: 180, unscored;
- confirmatory queries: 68, uninspected and unscored;
- A4.4a/A4.4d validation rows: not rescored;
- candidate-model comparison: forbidden.

A successful GitHub Actions workflow means only that execution integrity and independent reconstruction passed. It does not imply that calibration readiness passed.

## Post-result rule

If every readiness requirement passes, the selected relevance and sufficiency thresholds are frozen and A4.5b may be closed as calibration-ready. Fresh validation still requires separate approval for A4.5c.

If any readiness requirement fails, the negative result is frozen. A4.5b may not change thresholds, substitute a model, rerun because of scientific failure, or open fresh validation. A new methodological decision would require a separately registered checkpoint.

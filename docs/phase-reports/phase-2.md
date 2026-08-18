# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: Active — A0/A1 checkpoint passed; A2 next
- Date opened: 2026-08-18
- Public version: 0.1.0

## Frozen inputs

- Phase 1 status: Passed.
- BANKING77 train: 7,904 rows; SHA-256 `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3`.
- BANKING77 validation: 1,976 rows; SHA-256 `5a6e2bef72257bb3aa33aba4ca4a93a13738e0a487be88e7846b986b33713455`.
- BANKING77 confirmatory test: 3,080 rows; SHA-256 `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468`.
- Model ladder: `routing-ladder-v1`, A0-A3 required; A4 disabled unless bounded remediation is authorized.
- Selection partition: validation only.

## Required deliverables

- [x] Public A0-A3 model-ladder contract.
- [x] Routing evaluation protocol.
- [ ] Reproducible A0-A3 implementations and evaluation artifacts.
  - [x] A0/A1 development benchmark and evidence checkpoint.
  - [ ] A2 sentence-embedding + linear classifier.
  - [ ] A3 compact transformer classifier.
- [ ] Calibration comparison.
- [ ] Frozen out-of-scope benchmark and evaluation.
- [ ] Declared routing cost matrix.
- [ ] Final risk-coverage report and operating-point selection.
- [ ] Router model card.
- [ ] `/v1/tickets/route` contract tests.
- [ ] One frozen routing configuration and operating threshold.
- [ ] Registered confirmatory result for H3 and H4.

## A0/A1 checkpoint

The first development benchmark was executed twice in independent GitHub Actions runs against the frozen train/validation partitions only. The generated result JSON, Markdown report, and all A1 validation predictions were byte-identical across both runs.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE |
|---|---:|---:|---:|---:|
| A0 most-frequent | 0.0005 | 0.0130 | 0.0491 | 0.9813 |
| A0 stratified | 0.0163 | 0.0162 | 0.0471 | 0.9833 |
| **A1 TF-IDF + logistic regression** | **0.8422** | **0.8407** | **0.9534** | **0.4895** |

A1 validation accuracy is 0.8563 while mean maximum probability is 0.3667, showing substantial under-confidence. Raw A1 confidence nevertheless orders risk usefully: selective risk is approximately 2.53% at 50% coverage and 6.07% at 70% coverage. No operating threshold is frozen from this checkpoint.

Permanent development evidence is recorded in:

- `benchmarks/routing/results/a0_a1_validation_v1.json`
- `benchmarks/routing/results/a0_a1_validation_v1.md`

## Test-set status

The official BANKING77 test split remains confirmatory and has not been downloaded or opened by the Phase 2 routing benchmark. It remains unauthorized for model selection, calibration selection, feature selection, or operating-threshold selection.

## Current decision

**A1 survives as the classical routing baseline.** Its classification performance is strong enough that A2 and A3 must justify their additional complexity through measurable improvements in routing quality, calibration, selective risk, OOS behavior, or later cost-aware evaluation.

Phase 2 remains open. The next locked implementation action is **A2 — frozen sentence embeddings plus a linear classifier** using exactly the same train/validation contract. Preserve A1 outputs for the later calibration comparison. Do not open the confirmatory test and do not introduce a classifier family outside the frozen ladder.

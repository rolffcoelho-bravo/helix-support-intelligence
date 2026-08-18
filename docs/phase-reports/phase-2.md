# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: Active — A0/A1/A2 checkpoints passed; A3 next
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
  - [x] A2 sentence-embedding + linear-classifier benchmark and evidence checkpoint.
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

## A2 checkpoint

A2 was frozen before evaluation as `sentence-transformers/all-MiniLM-L6-v2` at revision `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`, used only as a 384-dimensional normalized sentence-embedding feature extractor. The linear classifier retained the fixed A1 logistic-regression specification. No alternate encoder or hyperparameter search was allowed.

Two independent GitHub Actions executions produced byte-identical deterministic result JSON, report, and all 1,976 validation predictions. The complete A2 script dependency graph is now committed in `benchmarks/routing/evaluate_a2.py.lock`.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| A1 | 0.8422 | 0.8407 | 0.9534 | 0.4895 | 0.4951 |
| **A2 frozen embeddings + logistic regression** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |
| **A2 − A1** | **+0.0564** | **+0.0556** | **+0.0197** | **−0.1986** | **−0.2450** |

A2 also improves selective risk at every registered coverage point. At 50% coverage, risk falls from approximately 2.53% for A1 to 0.40% for A2; at 70%, from approximately 6.07% to 1.59%. These are development observations only and do not select an operating threshold.

A2 adds a neural embedding stage. On two GitHub-hosted CPU runs, validation embedding required approximately 5.57–5.71 ms per example, but a standardized A1 end-to-end latency comparator has not yet been run. Final complexity and cost selection therefore remains open.

Permanent A2 evidence is recorded in:

- `benchmarks/routing/results/a2_validation_v1.json`
- `benchmarks/routing/results/a2_validation_v1.md`
- `benchmarks/routing/evaluate_a2.py.lock`

## Test-set status

The official BANKING77 test split remains confirmatory and has not been downloaded or opened by the Phase 2 routing benchmarks. It remains unauthorized for model selection, calibration selection, feature selection, error-driven tuning, or operating-threshold selection.

## Current decision

**A2 survives and is the leading Phase 2 development candidate.** A1 remains the required simpler classical reference. A2's gain is large enough that A3 must now justify its additional training and inference complexity against A2 rather than merely outperforming A1.

Phase 2 remains open. The next locked implementation action is **A3 — one frozen compact transformer classifier** under the identical train/validation contract. Its base checkpoint, tokenizer, training budget, seed, optimization specification, and early-stopping rule must be frozen before the first A3 result. Do not open the confirmatory test and do not introduce a classifier family outside the frozen ladder.

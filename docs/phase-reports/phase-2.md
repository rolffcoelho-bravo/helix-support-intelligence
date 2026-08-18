# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: Active — model ladder and calibration complete; OOS benchmark next
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
- [x] Reproducible A0-A3 implementations and evaluation artifacts.
  - [x] A0/A1 development benchmark and evidence checkpoint.
  - [x] A2 sentence-embedding + linear-classifier benchmark and audited evidence checkpoint.
  - [x] A3 compact-transformer benchmark and audited negative-result checkpoint.
- [x] Calibration comparison.
- [ ] Frozen out-of-scope benchmark and evaluation.
- [ ] Declared routing cost matrix.
- [ ] Final risk-coverage report and operating-point selection.
- [ ] Router model card.
- [ ] `/v1/tickets/route` contract tests.
- [ ] One frozen routing configuration and operating threshold.
- [ ] Registered confirmatory result for H3 and H4.

## A0/A1 checkpoint

A1 — fixed TF-IDF plus logistic regression — established the classical reference on the frozen validation partition.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE |
|---|---:|---:|---:|---:|
| A0 most-frequent | 0.0005 | 0.0130 | 0.0491 | 0.9813 |
| A0 stratified | 0.0163 | 0.0162 | 0.0471 | 0.9833 |
| **A1** | **0.8422** | **0.8407** | **0.9534** | **0.4895** |

A1 is a substantial baseline but is strongly under-confident. Permanent evidence is recorded in `benchmarks/routing/results/a0_a1_validation_v1.{json,md}`.

## A2 checkpoint

A2 was frozen before evaluation as `sentence-transformers/all-MiniLM-L6-v2` at revision `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`, used only as a normalized 384-dimensional feature extractor with the same fixed logistic-regression specification as A1.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| A1 | 0.8422 | 0.8407 | 0.9534 | 0.4895 | 0.4951 |
| **A2** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |

A2 improves selective risk at every registered raw-confidence coverage point and survives as the leading uncalibrated candidate. Its audited environment resolves CPU-only PyTorch, and two post-audit CPU runs support exact decision reproducibility plus bounded numerical reproducibility rather than bitwise floating-point identity.

Permanent evidence is recorded in `benchmarks/routing/results/a2_validation_v2.{json,md}` and `benchmarks/routing/evaluate_a2.py.lock`.

## A3 checkpoint

A3 tested end-to-end task-specific fine-tuning of the same MiniLM transformer under a fixed three-epoch, no-rescue contract.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| **A2** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |
| A3 | 0.6898 | 0.7105 | 0.9226 | 0.7221 | 0.9771 |

The hostile audit found no code, split, truncation, dependency, or reproducibility defect that invalidates the negative result. The registered A3 recipe underfits and does not justify its added complexity. This does not establish that transformer fine-tuning is intrinsically inferior; changing the recipe after the result would violate the anti-shopping rule.

**Decision:** A3 is rejected. A2 remains the leader; A1 remains the simpler reference; A4 remains disabled.

Permanent evidence is recorded in `benchmarks/routing/results/a3_validation_v1.{json,md}` and `benchmarks/routing/evaluate_a3.py.lock`.

## Calibration checkpoint

Calibration was evaluated for A1 and A2 only. Rejected A3 was not allowed to re-enter through post-hoc calibration. Three pre-frozen methods were compared: multiclass temperature scaling, one-vs-rest isotonic regression with row normalization, and one-vs-rest Platt scaling with row normalization.

The comparison used deterministic five-fold intent-stratified cross-fitting inside validation, so every scored row was calibrated by a calibrator that did not see that row. The final audited score-fold counts are **390 / 392 / 393 / 402 / 399**.

### A1

| Method | Macro-F1 | Top-3 | ECE | Brier | NLL | Guardrail |
|---|---:|---:|---:|---:|---:|---|
| Raw | 0.8422 | 0.9534 | 0.4895 | 0.4951 | 1.3666 | reference |
| **Temperature** | **0.8422** | **0.9534** | **0.0263** | **0.2150** | **0.5403** | **pass** |
| Isotonic | 0.8319 | 0.9423 | 0.0518 | 0.2461 | 1.6839 | fail |
| Platt | 0.8477 | 0.9555 | 0.0898 | 0.2314 | 0.5920 | pass |

### A2

| Method | Macro-F1 | Top-3 | ECE | Brier | NLL | Guardrail |
|---|---:|---:|---:|---:|---:|---|
| Raw | 0.8986 | 0.9732 | 0.2910 | 0.2501 | 0.6683 | reference |
| **Temperature** | **0.8986** | **0.9732** | **0.0162** | **0.1398** | **0.3350** | **pass** |
| Isotonic | 0.8898 | 0.9615 | 0.0126 | 0.1571 | 1.2764 | fail |
| Platt | 0.8948 | 0.9691 | 0.0305 | 0.1497 | 0.4022 | fail |

**Decision:** temperature scaling is frozen as the calibration method for both A1 and A2. The full-validation refit temperatures are approximately `0.346418` for A1 and `0.457974` for A2. **A2 + temperature scaling is now the leading calibrated development candidate.**

No operating threshold is selected by this checkpoint. H3 is also still open because its registered primary endpoint is expected routing cost, which has not been evaluated.

### Calibration audit

The initial cross-fit implementation was not frozen immediately. The hostile audit found two execution issues:

1. the first deterministic within-intent round-robin always started at fold zero, producing unnecessarily uneven fold sizes of 430 / 411 / 392 / 380 / 363;
2. an initial workflow-based source rewrite updated the configuration but failed to update the actual fold-assignment code.

Both issues were corrected before closure. Intent-specific start offsets are now derived only from the already-frozen salt and intent name, not from model performance. The source/config agreement is protected by tests and CI. Two independent runs from the corrected source and committed read-only CPU environment produced identical fold assignments, predicted intents, classification metrics, selected methods, and selective-risk curves. Floating probabilities differ only at bounded CPU numerical scale.

Permanent calibration evidence is recorded in:

- `benchmarks/routing/results/calibration_validation_v1.json`
- `benchmarks/routing/results/calibration_validation_v1.md`
- `benchmarks/routing/evaluate_calibration.py.lock`

## Post-execution audit rule

Every checkpoint must close with a hostile audit covering code correctness, metric interpretation, split integrity, reproducibility, dependency/hardware consistency, CI behavior, and public wording. Findings must be corrected before the checkpoint is declared complete.

## Test-set status

The official BANKING77 test split remains confirmatory and has not been downloaded or opened by the Phase 2 routing benchmarks. It remains unauthorized for model selection, calibration selection, feature selection, error-driven tuning, OOS benchmark construction, or operating-threshold selection.

## Current decision

The A0-A3 ladder and calibration comparison are complete. **A2 with temperature scaling is the leading calibrated development candidate.** A1 remains the simpler reference, A3 is rejected as registered, and A4 remains disabled.

Phase 2 remains open.

**Next locked action:** freeze the out-of-scope benchmark and its construction rules **before inspecting any OOS score**, then evaluate the frozen A1 reference and calibrated A2 candidate. The confirmatory BANKING77 test remains sealed.

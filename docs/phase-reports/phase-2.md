# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: Active — model ladder, calibration, OOS, cost policy, and development configuration complete; implementation contracts next
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
- [x] Frozen out-of-scope benchmark and evaluation.
- [x] Declared routing cost matrix.
- [x] Final development risk-coverage report and operating-point selection.
- [ ] Router model card.
- [ ] `/v1/tickets/route` contract tests.
- [x] One frozen development routing configuration and operating threshold.
- [ ] Registered confirmatory result for H3 and H4.

## Model ladder

### A1 classical reference

A1 — fixed TF-IDF plus logistic regression — produced macro-F1 `0.8422`, balanced accuracy `0.8407`, top-3 recall `0.9534`, ECE `0.4895`, and Brier `0.4951` on frozen validation. It is a substantial baseline but strongly under-confident.

Permanent evidence: `benchmarks/routing/results/a0_a1_validation_v1.{json,md}`.

### A2 leading candidate

A2 was frozen before evaluation as `sentence-transformers/all-MiniLM-L6-v2` at revision `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`, used as a normalized 384-dimensional feature extractor with the same fixed logistic-regression specification as A1.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| A1 | 0.8422 | 0.8407 | 0.9534 | 0.4895 | 0.4951 |
| **A2** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |

The audited A2 environment resolves CPU-only PyTorch. Independent runs support exact decision reproducibility plus bounded numerical reproducibility rather than bitwise floating-point identity.

Permanent evidence: `benchmarks/routing/results/a2_validation_v2.{json,md}` and `benchmarks/routing/evaluate_a2.py.lock`.

### A3 registered negative result

A3 tested end-to-end task-specific fine-tuning of the same MiniLM transformer under a fixed three-epoch, no-rescue contract. It produced macro-F1 `0.6898`, balanced accuracy `0.7105`, top-3 recall `0.9226`, ECE `0.7221`, and Brier `0.9771`.

The hostile audit found no code, split, truncation, dependency, or reproducibility defect that invalidates this result. The registered A3 recipe underfits and does not justify its added complexity. This does not establish that transformer fine-tuning is intrinsically inferior; changing the recipe after the result would violate the anti-shopping rule.

**Decision:** A3 is rejected. A2 remains the leader; A1 remains the simpler reference; A4 remains disabled.

Permanent evidence: `benchmarks/routing/results/a3_validation_v1.{json,md}` and `benchmarks/routing/evaluate_a3.py.lock`.

## Calibration checkpoint

Calibration was evaluated for A1 and A2 only using deterministic five-fold intent-stratified cross-fitting. The final audited score-fold counts are **390 / 392 / 393 / 402 / 399**. Rejected A3 was not allowed to re-enter through post-hoc calibration.

| Model | Raw ECE | Temperature ECE | Raw Brier | Temperature Brier | Raw NLL | Temperature NLL |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 0.4895 | **0.0263** | 0.4951 | **0.2150** | 1.3666 | **0.5403** |
| A2 | 0.2910 | **0.0162** | 0.2501 | **0.1398** | 0.6683 | **0.3350** |

Temperature scaling wins the frozen calibration selection rule for both A1 and A2 without changing A2 macro-F1 or top-3 recall. The full-validation refit temperatures are approximately `0.346418` for A1 and `0.457974` for A2.

**Decision:** temperature scaling is frozen. A2 + temperature scaling becomes the leading calibrated development candidate.

Permanent evidence: `benchmarks/routing/results/calibration_validation_v1.{json,md}` and `benchmarks/routing/evaluate_calibration.py.lock`.

## Frozen OOS checkpoint

`routing-oos-v1` was frozen before model scoring. It contains 160 hand-authored support-like OOS queries across 20 categories: 80 near-boundary, 64 medium, and 16 far-support. Exact normalized overlap with the frozen BANKING77 source train is zero.

The primary OOS estimate uses the same five cross-fitted calibration folds, so each in-domain row is scored by a calibrator that did not fit on that row.

| Model | Cross-fitted OOS AUROC | ID FPR at >=95% OOS recall |
|---|---:|---:|
| A1 + temperature | 0.8491 | 0.6442 |
| **A2 + temperature** | **0.8956** | **0.4342** |

A2 is better than A1 under the frozen OOS primary rule, but OOS is not solved: at high recall, approximately 43.4% of in-domain A2 cases are still falsely flagged by the metric-specific OOS threshold. This limitation is carried into cost/operating-point selection rather than hidden.

Permanent evidence: `benchmarks/routing/results/oos_validation_v1.{json,md}`.

## Routing cost and selective operating point

The public cost matrix `routing-cost-matrix-v1` uses synthetic human-review-equivalent scenario units. These weights are explicit decision-analysis assumptions and are **not** claims about real-bank economics. The primary OOS prevalence assumption is 10%, with registered 5% and 20% sensitivity cases and three registered cost matrices.

### Primary development comparison

| Candidate | Expected cost | ID coverage | ID selective risk | OOS escalation |
|---|---:|---:|---:|---:|
| A1 raw | 0.5571 | 76.16% | 6.98% | 86.87% |
| A1 + temperature | 0.5428 | 73.73% | 4.39% | 80.62% |
| A2 raw | **0.4029** | 83.30% | 3.52% | 88.75% |
| **A2 + temperature** | 0.4214 | **75.00%** | **1.21%** | **88.00%** |

The Phase 2 protocol requires one calibrated router. Among eligible calibrated candidates, A2 + temperature has lower primary expected cost than A1 + temperature and wins all nine registered calibrated cost/prevalence sensitivity cells.

### H3 development result

**Unsupported for A2 on the registered primary endpoint.** A2 temperature scaling changes minimum expected routing cost from `0.4028846154` raw to `0.4214432566` calibrated, a `+0.0185586412` deterioration. The same direction holds in all nine registered sensitivity cells.

This is an important negative result: temperature scaling materially improves ECE, Brier, and NLL, but it does not earn a routing-cost benefit under the frozen A2 development decision problem after each score representation receives its own validation-selected threshold.

### H4 development result

**Supported on development evidence.** For calibrated A2, the selected cross-fitted policy reduces expected scenario cost from `1.0827935223` under full automation to `0.4214432566`, a reduction of `0.6613502657`. ID selective risk falls from `9.77%` under full automation to `1.21%` at 75% coverage.

At fixed development coverage, calibrated A2 selective risk is approximately `0.20%` at 50%, `1.08%` at 70%, and `4.78%` at 90%.

### Reproducibility and threshold-transfer audit

Two independent cost-policy runs reproduced the selected candidate, all route/escalate decision hashes, expected cost, coverage, selective risk, OOS escalation, H3/H4 development status, and every registered sensitivity winner. The selected cross-fitted threshold differed by only about `1.91e-8` across CPU runs; this checkpoint therefore claims exact decision reproducibility plus bounded numerical threshold reproducibility.

Because unbiased development selection uses cross-fitted calibration while deployment uses one full-validation temperature refit, a separate audit transferred the already-selected 75% coverage to the final calibration scale without re-optimizing model, calibration, cost, or coverage. That audit was reproduced byte-for-byte.

Full-refit transfer result:

- full-validation temperature: approximately `0.457974035`;
- 75%-coverage confidence plateau: `(0.892462899, 0.892944242]`;
- midpoint: `0.892703570`;
- changed acceptance identities relative to cross-fitting: `6 / 1976`;
- accepted-set Jaccard: `0.99596`;
- transfer selective risk: `1.28%`;
- transfer OOS escalation: `88.125%`;
- implied primary scenario cost: approximately `0.4235`.

**Frozen development configuration:** `routing-selected-v1` uses A2, temperature scaling with canonical temperature `0.457974`, and automatic routing when maximum calibrated class probability is `>= 0.892704`; otherwise it returns `ESCALATE_LOW_CONFIDENCE`. The threshold is the six-decimal rounded midpoint of the audited full-refit decision plateau, not a second cost optimization.

Permanent evidence: `benchmarks/routing/results/cost_policy_validation_v1.{json,md}` and `configs/models/routing_selected_v1.json`.

## Post-execution audit rule

Every Phase 2 checkpoint closes only after a hostile audit covering code correctness, metric interpretation, split integrity, reproducibility, dependency/hardware consistency, CI behavior, and public wording. Material findings are corrected before the checkpoint is accepted; negative scientific results are preserved rather than tuned away.

## Test-set status

The official BANKING77 test split remains confirmatory and has not been downloaded or opened by the Phase 2 routing development workflows. It remains unauthorized for model selection, calibration selection, feature selection, error-driven tuning, OOS benchmark construction, cost-weight selection, or operating-threshold selection.

The confirmatory test is not permitted to change `routing-selected-v1`; it can only evaluate the frozen configuration and registered hypotheses.

## Current decision

The model ladder, calibration choice, OOS benchmark, public cost matrix, selective operating point, and one development routing configuration are now frozen. **A2 + temperature scaling remains the selected calibrated development router. H3 is unsupported on development cost evidence; H4 is supported on development selective-routing evidence. Neither is yet a confirmatory verdict.**

Phase 2 remains open because the implementation-facing contracts are incomplete.

**Next locked action:** create the router model card and implement `/v1/tickets/route` contract tests against `routing-selected-v1`. Do not open the confirmatory BANKING77 test until those contracts pass and the final pre-confirmatory audit is complete.

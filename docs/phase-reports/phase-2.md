# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: **Closed — confirmatory result recorded and post-result hostile audit passed**
- Date opened: 2026-08-18
- Date closed: 2026-08-19
- Public version: 0.1.0

## Exit decision

Phase 2 is complete. The routing model ladder, calibration study, OOS development benchmark, declared routing-cost model, selective operating point, frozen router, model card, route contract, pre-confirmatory freeze, one-shot official BANKING77 confirmatory evaluation, and post-result hostile audit are all complete.

The selected Phase 2 router remains `routing-selected-v1`:

- model: **A2**;
- encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- encoder revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`;
- calibration: temperature scaling;
- frozen temperature: `0.457974`;
- automatic routing when maximum calibrated class probability is `>= 0.892704`;
- otherwise: `ESCALATE_LOW_CONFIDENCE`.

No post-test model search, calibration refit, threshold movement, cost change, or hypothesis redefinition occurred.

## Frozen data

- BANKING77 train: 7,904 rows; SHA-256 `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3`.
- BANKING77 validation: 1,976 rows; SHA-256 `5a6e2bef72257bb3aa33aba4ca4a93a13738e0a487be88e7846b986b33713455`.
- BANKING77 confirmatory test: 3,080 rows; canonical SHA-256 `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468`.
- Official source test SHA-256: `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`.
- Test distribution: exactly 77 intents × 40 rows.

The official test remained sealed through all development selection and was opened exactly once only after the pre-confirmatory integrity gate passed.

## Required deliverables

- [x] Public A0-A3 routing ladder.
- [x] Reproducible A0/A1 baseline evaluation.
- [x] Reproducible A2 sentence-embedding + linear classifier evaluation.
- [x] Registered A3 negative result with no rescue tuning.
- [x] Cross-fitted calibration comparison.
- [x] Frozen OOS development benchmark.
- [x] Declared routing cost matrix and sensitivity analysis.
- [x] Frozen selective operating point.
- [x] Full-refit threshold-transfer audit.
- [x] Router model card.
- [x] Framework-neutral `/v1/tickets/route` contract.
- [x] Final pre-confirmatory hostile audit and machine-checkable freeze.
- [x] One-shot registered BANKING77 confirmatory evaluation.
- [x] Mandatory post-result hostile audit.
- [x] Consumed confirmatory execution workflows retired before merge.

## Model ladder

| Model | Development macro-F1 | Balanced accuracy | Top-3 recall | Decision |
|---|---:|---:|---:|---|
| A1 TF-IDF + logistic regression | 0.8422 | 0.8407 | 0.9534 | simpler reference |
| **A2 MiniLM embeddings + logistic regression** | **0.8986** | **0.8963** | **0.9732** | **selected** |
| A3 fixed three-epoch fine-tuning | 0.6898 | 0.7105 | 0.9226 | rejected negative result |

A3's registered recipe underfit. It was preserved as a negative result rather than rescued post hoc. A4 remained disabled throughout Phase 2.

Permanent evidence:

- `benchmarks/routing/results/a0_a1_validation_v1.{json,md}`
- `benchmarks/routing/results/a2_validation_v2.{json,md}`
- `benchmarks/routing/results/a3_validation_v1.{json,md}`

## Calibration

Five-fold deterministic intent-stratified cross-fitting selected temperature scaling for A1 and A2.

For A2 on validation:

- ECE: `0.2910 -> 0.0162`;
- Brier: `0.2501 -> 0.1398`;
- NLL: `0.6683 -> 0.3350`;
- macro-F1 and top-3 recall unchanged.

The full-validation A2 temperature refit is approximately `0.457974035`, frozen publicly as `0.457974`.

Permanent evidence: `benchmarks/routing/results/calibration_validation_v1.{json,md}`.

## OOS development evidence

The frozen `routing-oos-v1` benchmark contains 160 support-like OOS queries across 20 categories. It was frozen before scoring and was valid for development evaluation, but after being used for operating-policy selection it no longer qualified as independent confirmatory evidence.

Development OOS results:

- A1 + temperature AUROC: `0.8491`;
- A2 + temperature AUROC: `0.8956`;
- A2 ID FPR at >=95% OOS recall: `0.4342`.

OOS separation therefore remained materially imperfect.

Permanent evidence: `benchmarks/routing/results/oos_validation_v1.{json,md}`.

## Routing cost and operating point

The registered public routing costs are synthetic decision-analysis units, not real-bank economics:

- correct route: `0`;
- wrong intent, same queue: `2.5`;
- wrong queue: `6`;
- unsafe high-risk wrong automatic route: `20`;
- human escalation: `1`.

Primary development results:

| Candidate | Expected cost | ID coverage | ID selective risk | OOS escalation |
|---|---:|---:|---:|---:|
| A1 raw | 0.5571 | 76.16% | 6.98% | 86.87% |
| A1 + temperature | 0.5428 | 73.73% | 4.39% | 80.62% |
| A2 raw | **0.4029** | 83.30% | 3.52% | 88.75% |
| **A2 + temperature** | 0.4214 | **75.00%** | **1.21%** | **88.00%** |

### H3 development

**Unsupported.** A2 temperature scaling increased the registered minimum development expected routing cost from `0.4028846154` to `0.4214432566`, a deterioration of `+0.0185586412`. The same direction held in all nine registered sensitivity cells.

This result is retained. Better probability calibration did not automatically imply better decision cost.

### H4 development

**Supported on development evidence.** Calibrated A2 reduced scenario cost from `1.0827935223` under full automation to `0.4214432566`, while ID selective risk fell from approximately `9.77%` to `1.21%` at 75% coverage.

Permanent evidence: `benchmarks/routing/results/cost_policy_validation_v1.{json,md}`.

## Threshold transfer

The selected 75% development coverage was transferred to the full-validation calibration scale without a second cost optimization.

- full-validation temperature: approximately `0.457974035`;
- 75%-coverage plateau: `(0.892462899, 0.892944242]`;
- rounded frozen threshold: `0.892704`;
- changed validation acceptance identities: `6 / 1976`;
- accepted-set Jaccard: `0.99596`;
- transfer selective risk: `1.28%`;
- transfer OOS escalation: `88.125%`.

## Final pre-confirmatory gate

Before test access, the hostile audit froze the scientific and execution surface with `routing-preconfirmatory-freeze-v1`. The final authorization-transport version verified **36 exact Git blobs** before opening the test.

The audit also corrected, before test access, a real evaluator bug involving the BANKING77 contract path and narrowed H3's independent confirmatory scope. Because the 160-query OOS set had already participated in development selection, only the unseen BANKING77 **in-domain cost component** could receive an independent H3 confirmatory verdict.

Permanent evidence:

- `benchmarks/routing/results/preconfirmatory_audit_v1.{json,md}`
- `configs/models/routing_preconfirmatory_manifest_v1.json`
- `configs/models/routing_confirmatory_v1.json`
- `docs/phase2-confirmatory-protocol.md`

## One-shot confirmatory execution

The approved one-shot evaluation ran successfully in GitHub Actions:

- workflow run: `32243835846`;
- job: `96039984239`;
- frozen Phase 2 commit actually checked out: `9f69bfc8d8e7f5520bee49cb6e9c8770fa20595a`;
- artifact ID: `9361858275`;
- artifact ZIP SHA-256: `0e57f9f12d1d86f4f52e74238735cc02f0628b6dc27cc29f96ffa4863a16cda3`;
- `results.json` SHA-256: `b82f8da068a2bd4870070fbf2d05159452939d5a39d192b97ef821efca2ba962`;
- `test_predictions.csv` SHA-256: `f8d0fc55f2ca14f02e0df1f15839197996320cce50ca30298910a550255c60cd`.

Later observer comments did not rerun the scientific job; their bridge jobs were skipped because the authorization token did not match.

## Confirmatory performance

| Metric | Official test result |
|---|---:|
| Accuracy | **0.9016** |
| Macro-F1 | **0.9016** |
| Balanced accuracy | **0.9016** |
| Top-3 recall | **0.9744** |
| ECE, 15 bins | **0.0169** |
| Multiclass Brier | **0.1456** |
| Negative log-likelihood | **0.3467** |

The selected router therefore retained approximately the same classification quality and strong aggregate calibration on the untouched official test set.

## Confirmatory H3 — independent in-domain component

| Quantity | Result |
|---|---:|
| Raw-A2 mean in-domain cost | `0.3112013` |
| Calibrated-A2 mean in-domain cost | `0.3152597` |
| Calibrated minus raw | `+0.0040584` |
| Paired-bootstrap 95% CI | `[-0.0259740, 0.0300365]` |
| Verdict | **INCONCLUSIVE** |

The point estimate remains slightly adverse to calibration, but uncertainty spans zero. The independent BANKING77 component does not provide evidence that calibration lowers routing cost.

This does **not** replace the development H3 verdict. The full mixed in-domain/OOS H3 endpoint remains **unsupported on development evidence** and cannot be independently confirmed in Phase 2 because no unseen OOS confirmatory sample exists.

## Confirmatory H4 — selective abstention

| Quantity | Result |
|---|---:|
| Full-automation errors | `303 / 3080` |
| Full-automation risk | **9.84%** |
| Errors at exactly 75% coverage | `45 / 2310` |
| Selective risk at 75% coverage | **1.95%** |
| Selective minus full risk | **-7.89 percentage points** |
| Paired-bootstrap 95% CI | **[-8.78 pp, -6.96 pp]** |
| Verdict | **SUPPORTED** |

H4 survives independent confirmation strongly. The full registered interval lies below zero.

The frozen deployment threshold `0.892704` realizes **74.12%** test coverage, accepts 2,283 rows, and has **1.88%** selective risk. The threshold was not moved after test access to force 75% realized coverage.

At that frozen threshold, three unsafe high-risk wrong automatic routes remain. This residual failure mode is part of the public limitation set.

Permanent evidence:

- `benchmarks/routing/results/confirmatory_test_v1.{json,md}`
- `benchmarks/routing/results/confirmatory_post_audit_v1.{json,md}`

## Post-result hostile audit

The immutable artifact was independently re-read and the critical arithmetic was reconstructed outside the evaluator.

The audit reproduced exactly:

- 303 total top-1 errors;
- 45 errors among the 2,310 highest-confidence rows at 75% coverage;
- 43 errors among 2,283 rows accepted at threshold `0.892704`;
- raw and calibrated routing-event counts;
- raw mean cost `0.3112012987`;
- calibrated mean cost `0.3152597403`;
- H3 difference `+0.0040584416`;
- H3 bootstrap interval `[-0.02597402597, 0.03003652597]`;
- H4 difference `-0.0788961039`;
- H4 bootstrap interval `[-0.08777326840, -0.06958874459]`.

No arithmetic, bootstrap, verdict-rule, split, threshold, calibration, or post-test-selection defect was found.

## Scientific interpretation

Phase 2 closes with three distinct findings that must not be collapsed into one claim:

1. **A2 classification generalizes well** on untouched BANKING77 test data.
2. **Temperature scaling provides strong probability calibration**, but no evidence that calibration itself reduces routing cost; the development mixed endpoint was adverse and the independent in-domain component is inconclusive.
3. **Selective abstention is independently supported** as a risk-control mechanism at the registered 75% confidence-ranked coverage level.

This distinction is methodologically useful: calibration quality, economic decision quality, and selective-routing reliability are related but non-equivalent objectives.

## Limitations

Phase 2 does not establish:

- independent OOS confirmation;
- real-bank costs or customer harm;
- production latency or throughput;
- live traffic drift robustness;
- staffing or business impact;
- elimination of high-risk routing failures.

The public cost weights remain scenario assumptions, and the synthetic OOS set remains development evidence only.

## Engineering closure

The temporary confirmatory authorization bridge and executable one-shot confirmatory workflows were removed after the scientific result was permanently captured. Their exact pre-test blobs remain recorded in the historical freeze manifest and Git history, but they will not become an active rerun surface when Phase 2 is merged.

The evaluator and frozen protocol remain as reproducibility documentation; the consumed one-shot execution path is retired.

## Final Phase 2 verdict

**Phase 2: PASSED AND CLOSED.**

- Selected router: A2 + temperature scaling.
- H3 development mixed endpoint: **unsupported**.
- H3 independent in-domain confirmatory component: **inconclusive**.
- H4 independent confirmatory endpoint: **supported**.
- Post-result hostile audit: **passed**.
- A4: never activated.
- Phase 3: **not started**.

The next blueprint action is the Phase 2 merge decision. Phase 3 retrieval remains forbidden until the Phase 2 branch is merged and the new phase is explicitly opened.

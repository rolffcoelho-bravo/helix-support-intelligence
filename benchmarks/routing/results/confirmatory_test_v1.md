# Phase 2 Registered Confirmatory Result

> One-shot frozen BANKING77 test evaluation. The scientific result is immutable. No post-test tuning, model reselection, calibration refit, threshold movement, cost-weight change, or hypothesis change is permitted.

## Execution provenance

- GitHub Actions run: `32243835846`
- Job: `96039984239`
- Authorization comment: `5340974436`
- Frozen Phase 2 branch head actually checked out by the scientific run: `9f69bfc8d8e7f5520bee49cb6e9c8770fa20595a`
- Confirmatory artifact: `9361858275`
- Artifact ZIP SHA-256: `0e57f9f12d1d86f4f52e74238735cc02f0628b6dc27cc29f96ffa4863a16cda3`
- `results.json` SHA-256: `b82f8da068a2bd4870070fbf2d05159452939d5a39d192b97ef821efca2ba962`
- `test_predictions.csv` SHA-256: `f8d0fc55f2ca14f02e0df1f15839197996320cce50ca30298910a550255c60cd`
- Pre-test integrity verifier: **36 frozen artifacts verified**

The later observer comments did not rerun the scientific evaluation. Their bridge jobs were skipped because their comment bodies did not equal the registered authorization token.

## Confirmatory data integrity

The run opened the official 3,080-row BANKING77 test split only after the frozen integrity verifier and no-test preflight passed.

- source test SHA-256: `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`
- canonical derived test SHA-256: `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468`
- test rows: `3,080`
- intents: `77`
- rows per intent: exactly `40`
- unique sample IDs: `3,080`

## Frozen model and policy

- model: A2
- encoder: `sentence-transformers/all-MiniLM-L6-v2`
- encoder revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`
- classifier: frozen logistic regression fitted on the 7,904-row train partition only
- calibration: temperature scaling
- temperature: `0.457974`
- frozen calibrated application threshold: `0.892704`
- frozen raw-A2 H3 comparator threshold: `0.367217`
- bootstrap: 5,000 paired nonparametric percentile replicates, seed `20260819`, 95% interval

## Confirmatory results

| Metric | Result |
|---|---:|
| Accuracy | **0.9016** |
| Macro-F1 | **0.9016** |
| Balanced accuracy | **0.9016** |
| Top-3 recall | **0.9744** |
| ECE, 15 bins | **0.0169** |
| Multiclass Brier | **0.1456** |
| Negative log-likelihood | **0.3467** |
| Mean maximum probability | **0.8986** |

The balanced accuracy equals ordinary accuracy here because the official BANKING77 test split contains exactly 40 examples for each of the 77 intents.

### Risk-coverage behavior

| Confidence-ranked coverage | Accepted | Selective risk |
|---|---:|---:|
| 50% | 1,540 | **0.45%** |
| 70% | 2,156 | **1.39%** |
| 75% | 2,310 | **1.95%** |
| 90% | 2,772 | **5.30%** |
| 100% | 3,080 | **9.84%** |

The frozen deployment threshold `0.892704` realizes **74.12%** coverage on test, not exactly 75%. It accepts 2,283 rows with 43 errors, giving selective risk **1.88%**. This is an out-of-sample coverage shift and does not authorize threshold movement.

At the frozen application threshold, routing events are:

- correct automatic route: `2,240`
- human escalation: `797`
- wrong intent, same queue: `36`
- wrong queue: `4`
- unsafe high-risk wrong automatic route: `3`

## H3 — independent in-domain component

The registered independent H3 component compares frozen calibrated-A2 in-domain routing cost with the frozen raw-A2 comparator on BANKING77 test cases.

| Quantity | Result |
|---|---:|
| Raw-A2 mean in-domain cost | `0.3112013` |
| Calibrated-A2 mean in-domain cost | `0.3152597` |
| Calibrated minus raw | **`+0.0040584`** |
| Paired bootstrap 95% CI | **`[-0.0259740, 0.0300365]`** |
| Registered verdict | **INCONCLUSIVE** |

The point estimate is slightly adverse to calibration, but the interval crosses zero. Therefore the independent BANKING77 component provides neither confirmatory support nor confirmatory rejection of a cost reduction.

This result must not be confused with the original mixed in-domain/OOS development H3 endpoint. That development endpoint was **unsupported** for A2, and Phase 2 has no unseen OOS sample capable of independently confirming the full mixed endpoint. The confirmatory test does not upgrade the full H3 claim.

## H4 — selective abstention

The registered H4 estimand compares routing error risk at exactly 75% confidence-ranked coverage with full automation on the independent test set.

| Quantity | Result |
|---|---:|
| Full-automation errors | `303 / 3,080` |
| Full-automation risk | **9.84%** |
| Accepted errors at 75% coverage | `45 / 2,310` |
| Selective risk at 75% coverage | **1.95%** |
| Selective minus full risk | **-7.89 percentage points** |
| Paired bootstrap 95% CI | **[-8.78 pp, -6.96 pp]** |
| Registered verdict | **SUPPORTED** |

The entire registered confidence interval lies below zero. H4 is therefore confirmatorily supported on BANKING77: selective abstention materially reduces routing error risk at the pre-registered 75% coverage level.

## Post-result arithmetic audit

The permanent artifact was independently re-read after execution. The audit reproduced:

- `303 / 3080 = 0.0983766234` full-automation risk;
- `45 / 2310 = 0.0194805195` selective risk at 75% coverage;
- H4 difference `-0.0788961039` exactly;
- frozen-threshold `43 / 2283 = 0.0188348664` selective risk;
- raw event-weighted mean cost `0.3112012987` exactly;
- calibrated event-weighted mean cost `0.3152597403` exactly;
- H3 difference `+0.0040584416` exactly;
- both registered 5,000-replicate bootstrap intervals exactly from the row-level artifact and frozen operations map.

No arithmetic, event-classification, CI, verdict-rule, split-integrity, or threshold-movement defect was found.

## Interpretation

The confirmatory evidence strengthens a deliberately non-simplistic Phase 2 conclusion.

A2 generalizes well on the untouched BANKING77 test: macro-F1 is approximately 0.902, top-3 recall approximately 0.974, and aggregate calibration remains strong with ECE approximately 0.017.

However, improved calibration is **not equivalent to improved routing economics**. The independent H3 in-domain component is inconclusive, while the development mixed-cost endpoint was already adverse to temperature scaling. The evidence does not support claiming that calibration itself lowers expected routing cost.

Selective abstention is different. H4 survives independent confirmation strongly: ranking by frozen calibrated confidence and abstaining on the lower-confidence quarter reduces routing error risk from about 9.84% to about 1.95%.

That distinction is central to the value of the result: probability calibration, economic decision quality, and selective-routing reliability are related but not interchangeable objectives.

## Limitations

The confirmatory result is specific to the frozen BANKING77 routing domain. It does not establish production latency, live traffic drift robustness, customer harm reduction, staffing economics, or real-bank financial impact.

The routing cost weights remain synthetic decision-analysis assumptions. The 160-query OOS set remains development evidence only and is not independent confirmation. Phase 2 still lacks an unseen OOS confirmatory sample.

Three unsafe high-risk wrong automatic routes remain at the frozen application threshold. Those residual failures are material limitations and must remain visible in public reporting.

## Decision

- H3 development mixed endpoint: **unsupported**.
- H3 independent BANKING77 in-domain confirmatory component: **inconclusive**.
- H4 independent confirmatory endpoint: **supported**.
- A2 remains the frozen selected router; no post-test rescue or reselection is permitted.
- Phase 3 remains forbidden until the post-result hostile audit and Phase 2 closure are completed.

# Phase 2 Out-of-Scope Routing Checkpoint

> **Development evidence only.** The OOS set is a frozen hand-authored support-like benchmark, not a production-distribution estimate. The confirmatory BANKING77 test split was not downloaded, opened, scored, or used.

## Frozen benchmark

The benchmark was frozen before any OOS model score existed:

- 160 unique OOS queries;
- 20 categories, eight queries each;
- 80 near-boundary banking-support requests;
- 64 medium-distance banking/business requests;
- 16 farther support/service requests;
- zero exact normalized overlap with the BANKING77 source-training pool, which contains the derived train and validation rows.

Primary OOS performance uses the already-audited five-fold calibration cross-fit. For each fold, temperature is fit on four validation folds, then that temperature scores the held-out in-domain fold and all 160 frozen OOS queries. The full-validation temperature is only a diagnostic.

## Primary result

| Model | Cross-fitted OOS AUROC | In-domain FPR at ≥95% OOS recall | Fold AUROC range |
|---|---:|---:|---:|
| A1 + temperature | 0.8491 | 64.42% | 0.8441–0.8559 |
| **A2 + temperature** | **0.8956** | **43.42%** | **0.8825–0.9011** |

A2 is the preferred model under the frozen OOS primary rule, but the result is **not operationally sufficient at the 95% OOS-recall diagnostic point**. Rejecting at least 95% of this frozen OOS set would also reject approximately 43% of held-out in-domain validation traffic.

The 95%-recall threshold used here is metric-specific and is not the final routing operating threshold.

## Boundary difficulty

The aggregate result hides a strong difficulty gradient for A2:

| OOS tier | AUROC | In-domain FPR at tier-specific ≥95% OOS recall |
|---|---:|---:|
| Near boundary | 0.8652 | 86.99% |
| Medium | 0.9250 | 39.12% |
| Far support | 0.9304 | 37.15% |

Tier-specific FPR figures use different diagnostic thresholds and must not be compared as one shared operating policy. They show where the confidence score fails: **semantically adjacent out-of-scope requests are much harder than obviously different requests.**

The clearest example is `direct_debit_management`. BANKING77 contains the in-domain intent `direct_debit_payment_not_recognised`, while the frozen OOS requests concern legitimate mandate creation, cancellation, pausing, and management. A2 obtains only about **0.3881 category AUROC** on those eight queries. This is an intended boundary test, not a mislabeled duplicate intent.

`joint_accounts_guardians` is another difficult near-boundary group with AUROC about **0.8214**. By contrast, categories such as account statements/references, personal loans, accessibility/localization, and insurance are separated much more cleanly.

Each category contains only eight OOS queries. Category-level values are diagnostics, not precise population estimates.

## Calibration creates an OOS tradeoff

Temperature scaling was selected because it materially improves held-out in-domain probability calibration. It does **not** improve max-probability OOS separation in this benchmark.

For A2:

| Score variant | OOS AUROC | In-domain FPR at ≥95% OOS recall |
|---|---:|---:|
| Raw max-probability diagnostic | **0.9336** | **33.65%** |
| Cross-fitted temperature-scaled primary | 0.8956 | 43.42% |
| Full-validation temperature diagnostic | 0.8956 | 43.37% |

This is a real design tension. Better probabilistic calibration for in-domain correctness does not imply better OOS discrimination. Raw confidence was pre-registered as diagnostic-only and therefore cannot replace the calibrated primary policy after seeing this result.

The cost/operating-point stage must account for this tension explicitly rather than equating calibrated confidence with scope certainty.

## Reproducibility

Two independent GitHub Actions runs from the committed CPU-only environment reproduced exactly:

- benchmark metadata and fold assignment;
- A1 and A2 primary AUROC/FPR values;
- A1 and A2 tier diagnostics;
- the frozen-rule preference for A2.

Floating threshold/score values differ only at normal CPU numerical scale; the largest observed numeric delta across float fields was approximately **5.33×10⁻⁷**. Bitwise floating-point identity is not claimed.

## Hostile audit

The checkpoint passed the standing execution audit:

- the 160-query benchmark was committed before the first OOS score;
- the scorer downloads only the frozen BANKING77 source-training CSV and never the official test CSV;
- exact normalized overlap with BANKING77 source-training text is hard-failed;
- the in-domain side of the primary OOS estimate uses calibration-held-out folds, avoiding full-validation calibrator reuse on its fitting rows;
- the audited balanced calibration folds remain 390 / 392 / 393 / 402 / 399;
- CPU-only lock is committed and contains no CUDA/NVIDIA/Triton packages;
- workflow permissions were restored to read-only;
- repository-wide CI passes after formatting corrections;
- the public conclusion is narrowed to a synthetic development benchmark and does not claim production OOS performance.

## Decision

**A2 + temperature scaling remains the leading Phase 2 development candidate, but OOS is not solved.** The high-recall false-positive burden, especially for near-boundary requests, must constrain threshold and cost selection.

No final operating threshold is selected. H3 and H4 remain open. The README release benchmark remains `pending`.

## Next locked action

Freeze the **routing cost matrix before using any cost result**, then evaluate expected routing cost and the final risk-coverage operating point for A1 and calibrated A2. Do not open the confirmatory BANKING77 test split.

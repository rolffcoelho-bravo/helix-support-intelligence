# Phase 2 Confirmatory Post-Result Hostile Audit

> Audit verdict: **PASSED**. The one-shot scientific result is internally consistent and remains unchanged.

## Source evidence

The audit is anchored to GitHub Actions run `32243835846`, job `96039984239`, artifact `9361858275`, and the immutable artifact hashes recorded in `confirmatory_test_v1.json`.

The scientific workflow checked out Phase 2 commit `9f69bfc8d8e7f5520bee49cb6e9c8770fa20595a`, verified all 36 frozen pre-confirmatory artifacts, passed the no-test preflight, and only then opened the official BANKING77 test split.

Later observer comments did **not** rerun the scientific evaluation. The two later bridge runs were both skipped because their comment bodies did not equal the registered `OPEN_FROZEN_TEST_ONCE` token.

## Audit table

| Surface | Result |
|---|---|
| One-shot execution integrity | **passed** |
| Frozen artifact verification before test | **36 / 36 passed** |
| Test source hash | **matched** |
| Test derived hash | **matched** |
| Test rows / unique IDs | **3,080 / 3,080** |
| Intent balance | **77 intents × 40 rows** |
| Model changed after test | **no** |
| Calibration refit on test | **no** |
| Threshold changed after test | **no** |
| Cost matrix changed after test | **no** |
| H3/H4 definitions changed after test | **no** |
| H3 arithmetic | **reproduced exactly** |
| H3 bootstrap CI | **reproduced exactly** |
| H3 registered verdict | **inconclusive** |
| H4 arithmetic | **reproduced exactly** |
| H4 bootstrap CI | **reproduced exactly** |
| H4 registered verdict | **supported** |
| Independent OOS overclaim | **prevented** |
| Residual unsafe high-risk routes hidden | **no** |

## Independent arithmetic reproduction

The artifact contains 3,080 predictions and 3,080 unique sample IDs. Recomputing top-1 errors from `test_predictions.csv` gives exactly `303`, hence:

`303 / 3080 = 0.09837662337662338`

which matches the reported full-automation risk and accuracy `1 - risk = 0.9016233766233767`.

At the registered 75% confidence-ranked coverage, exactly 2,310 rows are accepted. Re-ranking the artifact by calibrated confidence with stable sample-ID tie breaking produces exactly 45 accepted errors:

`45 / 2310 = 0.01948051948051948`.

Therefore H4's primary difference reproduces exactly:

`0.01948051948051948 - 0.09837662337662338 = -0.07889610389610391`.

At the frozen deployment threshold `0.892704`, exactly 2,283 rows are accepted and 43 are wrong:

`43 / 2283 = 0.018834866403854577`.

Realized test coverage is `2283 / 3080 = 0.7412337662337662`. The threshold was **not** moved to force 75% test coverage.

## H3 cost audit

Using the frozen routing operations map and the published `balanced_risk_v1` costs, the row-level event classification was independently rebuilt from the prediction artifact.

Raw comparator event counts reproduce exactly:

- correct route: `2,465`
- human escalation: `505`
- unsafe high-risk wrong automatic route: `6`
- wrong intent, same queue: `83`
- wrong queue: `21`

The corresponding mean cost is exactly `0.3112012987012987`.

Frozen calibrated-policy event counts reproduce exactly:

- correct route: `2,240`
- human escalation: `797`
- unsafe high-risk wrong automatic route: `3`
- wrong intent, same queue: `36`
- wrong queue: `4`

The corresponding mean cost is exactly `0.31525974025974024`.

The independent H3 in-domain point estimate therefore reproduces exactly as `+0.004058441558441558` calibrated minus raw.

The point estimate is adverse to calibration, but the inferential rule—not the sign of the point estimate—determines the registered verdict.

## Bootstrap audit

The audit independently reimplemented the registered paired percentile bootstrap from the artifact:

- 5,000 replicates;
- seed `20260819`;
- row-level paired resampling for H3;
- H4 full and 75%-coverage selective risk recomputed inside each bootstrap sample;
- stable sample-ID tie order.

The independent reproduction obtained **exactly**:

- H3 95% CI: `[-0.025974025974025976, 0.030036525974025913]`;
- H4 95% CI: `[-0.08777326839826838, -0.06958874458874459]`.

The H3 interval crosses zero, so the registered independent in-domain verdict is **inconclusive**. The H4 interval lies completely below zero, so H4 is **supported**.

No CI implementation or verdict-rule defect was found.

## Interpretation audit

The result supports a narrower and more defensible conclusion than a generic claim that calibration improves all routing outcomes.

Temperature scaling remains strongly useful for probability calibration. On the untouched test set, ECE is approximately `0.0169`, Brier `0.1456`, and NLL `0.3467`. But the independent H3 in-domain component does not establish lower routing cost: its point estimate is slightly worse and uncertainty spans zero. Combined with the already-unsupported development mixed-cost endpoint, there is no basis for claiming a calibration-driven routing-cost improvement.

By contrast, selective abstention survives independent confirmation. At exactly 75% confidence-ranked coverage, routing error risk falls from approximately `9.84%` under full automation to `1.95%`, an absolute reduction of approximately `7.89` percentage points under the registered estimand.

The frozen application threshold realizes 74.12% test coverage rather than exactly 75%. That difference is a legitimate out-of-sample transfer result and is not corrected post hoc.

## Residual failures

The selected frozen application policy still produces three unsafe high-risk wrong automatic routes on the official test set. The post-result audit identified them in the immutable prediction artifact. Their existence does not invalidate H4, but it prevents any claim that selective routing eliminates high-risk failures.

The raw comparator produces six such unsafe high-risk automatic routes. The calibrated threshold therefore reduces, but does not remove, this failure mode.

## Public-claim discipline

The permitted Phase 2 conclusions are:

- **A2 test performance:** macro-F1 approximately `0.9016`, balanced accuracy approximately `0.9016`, top-3 recall approximately `0.9744`.
- **Calibration:** aggregate test calibration remains strong.
- **H3 development mixed endpoint:** unsupported.
- **H3 independent BANKING77 in-domain component:** inconclusive.
- **H4 independent BANKING77 confirmatory endpoint:** supported at the registered 75% confidence-ranked coverage.

The following claims remain prohibited:

- that the full mixed in-domain/OOS H3 endpoint was independently confirmed;
- that real-bank economics were validated;
- that the synthetic OOS benchmark is independent confirmatory evidence;
- that production latency, drift robustness, staffing impact, customer harm, or live deployment performance were established;
- that high-risk automatic-routing errors were eliminated.

## Final audit verdict

**PASSED.**

No code, arithmetic, split, calibration, threshold, bootstrap, verdict-rule, reproducibility, or public-interpretation defect requires a scientific rerun or post-test repair. The registered confirmatory result must be preserved as observed.

# Phase 2 Routing Cost and Operating-Point Checkpoint

> Development-only evidence. The cost weights are synthetic decision-analysis assumptions, not real-bank economics. The confirmatory BANKING77 test split remains unopened.

## Primary cost comparison

| Candidate | Threshold | Expected cost | ID coverage | ID selective risk | OOS escalation |
|---|---:|---:|---:|---:|---:|
| A1 raw | 0.160054 | 0.5571 | 76.16% | 6.98% | 86.87% |
| A1 + temperature | 0.793325 | 0.5428 | 73.73% | 4.39% | 80.62% |
| A2 raw | 0.367217 | 0.4029 | 83.30% | 3.52% | 88.75% |
| **A2 + temperature** | **0.893770 cross-fitted scale** | **0.4214** | **75.00%** | **1.21%** | **88.00%** |

The Phase 2 protocol requires one calibrated router. Among the eligible calibrated configurations, **A2 + temperature scaling** has lower primary expected cost than A1 + temperature scaling and wins all nine registered calibrated cost/prevalence sensitivity cells.

## H3 development result: unsupported for A2

H3 registered expected routing cost as its primary endpoint. A2 raw confidence reaches a lower validation-selected minimum expected cost than calibrated A2:

- A2 raw: `0.4028846154`;
- A2 temperature: `0.4214432566`;
- calibrated minus raw: `+0.0185586412`.

This direction remains the same in every registered cost/prevalence sensitivity cell. The correct interpretation is narrow: temperature scaling materially improves ECE, Brier, and NLL, but under the frozen decision costs it does **not** reduce A2's minimum development routing cost after each score representation receives its own threshold. Calibration is therefore not credited with a cost benefit that the experiment did not show.

## H4 development result: supported

For the selected calibrated A2 development policy:

- full automation expected cost: `1.0827935223`;
- selective-policy expected cost: `0.4214432566`;
- expected cost reduction: `0.6613502657`;
- full-automation ID risk: `9.77%`;
- selected-policy ID risk: `1.21%`;
- ID automation coverage: `75.00%`;
- OOS escalation: approximately `88.00%`.

At fixed development coverage, calibrated A2 selective risk is approximately `0.20%` at 50%, `1.08%` at 70%, and `4.78%` at 90%.

## Reproducibility audit

Two independent cost-policy runs reproduced the selected candidate, all route/escalate decision hashes, expected cost, coverage, selective risk, OOS escalation, H3/H4 development status, and every registered sensitivity winner. The selected cross-fitted threshold differed by only about `1.91e-8` because upstream A2 probabilities have bounded CPU numerical variation. The scientific claim is therefore **exact decision reproducibility plus bounded numerical threshold reproducibility**, not bitwise probability identity.

## Transfer to the final calibration scale

Development selection correctly uses cross-fitted calibration so a validation row is not scored by a calibrator fit on that row. Deployment, however, uses the single temperature refit on all validation data. A separate audit therefore transferred the already-selected 75% coverage to the full-refit probability scale without re-optimizing cost, model, calibration method, or coverage.

The transfer audit was reproduced byte-for-byte and found:

- full-validation refit temperature: approximately `0.457974035`;
- full-refit 75%-coverage decision plateau: `(0.892462899, 0.892944242]`;
- midpoint: `0.892703570`;
- changed acceptance identities versus cross-fitting: `6 / 1976`;
- accepted-set Jaccard: `0.99596`;
- full-refit transfer selective risk: `1.28%`;
- full-refit transfer OOS escalation: `88.125%`.

The frozen development configuration therefore uses **temperature `0.457974` and threshold `0.892704`**, the six-decimal rounded full-refit values. This is a numerically robust encoding of the already-selected policy, not a second optimization.

## Decision

`routing-selected-v1` is frozen as the development configuration:

- model: A2 frozen MiniLM embeddings + logistic regression;
- calibration: temperature scaling;
- automatic routing: maximum calibrated class probability `>= 0.892704`;
- otherwise: `ESCALATE_LOW_CONFIDENCE`.

The official BANKING77 test remains sealed. These results do not populate the README release benchmark table and do not constitute the confirmatory H3/H4 verdict.

## Next locked action

Create the router model card and implement `/v1/tickets/route` contract tests against `routing-selected-v1`. The confirmatory test must remain unopened until those implementation contracts are closed.

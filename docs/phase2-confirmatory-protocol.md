# Phase 2 Confirmatory Protocol

## Purpose

This document records the confirmatory evaluation design used for the frozen Phase 2 routing configuration before the official BANKING77 test split was evaluated. It does not reopen model selection, calibration selection, OOS construction, cost-weight selection, threshold selection, or the A0-A3 comparison.

The machine-readable source of truth is `configs/models/routing_confirmatory_v1.json`.

## Frozen model and policy

The confirmatory router is `routing-selected-v1`:

- model: A2;
- encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- encoder revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`;
- classifier: frozen multiclass logistic regression from `routing-a2-v1`;
- classifier fitting data: the frozen 7,904-row train partition only;
- calibration: temperature scaling;
- temperature: `0.457974`;
- calibrated application threshold: `0.892704`;
- raw-A2 H3 comparator threshold: `0.367217`.

The confirmatory result cannot change any of these values.

## Confirmatory data boundary

The independent routing-intent partition is the official 3,080-row BANKING77 test split. Its source SHA-256 is `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`; its canonical Helix JSONL SHA-256 is `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468`.

The 160-query `routing-oos-v1` benchmark is not independent confirmatory evidence because it participated in development evaluation and operating-policy selection. It can provide scenario context only.

This distinction prevents the repository from overstating the independence of a mixed in-domain/OOS expected-cost statistic.

## H3 confirmatory estimand

H3 asks whether calibration reduces damaging automatic-routing cost.

The independent estimand available in Phase 2 is:

`mean BANKING77-test in-domain cost under frozen A2-temperature policy - mean BANKING77-test in-domain cost under frozen A2-raw policy`.

Both policies use the same A2 predictions and the same published `balanced_risk_v1` event costs. The raw policy uses threshold `0.367217`; the calibrated policy uses temperature `0.457974` and threshold `0.892704`. Test data cannot optimize either threshold.

The event costs are synthetic decision-analysis assumptions:

- correct route: `0`;
- wrong intent, same queue: `2.5`;
- wrong queue: `6`;
- unsafe high-risk wrong automatic route: `20`;
- human escalation: `1`.

This is an independent test of the in-domain component of H3 rather than a full independent confirmation of the original mixed in-domain/OOS development endpoint.

### H3 uncertainty and verdict

A paired nonparametric percentile bootstrap with 5,000 replicates, seed `20260819`, and a 95% confidence interval resamples BANKING77 test rows while keeping raw and calibrated row costs paired.

- **Supported component:** upper confidence bound for calibrated-minus-raw mean in-domain cost is below `0`.
- **Unsupported component:** lower confidence bound is at or above `0`.
- **Inconclusive component:** the interval overlaps `0`.

## H4 confirmatory estimand

H4 asks whether selective abstention reduces routing risk.

The primary estimand is:

`selective risk at exactly 75% confidence-ranked BANKING77-test coverage - full-automation BANKING77-test risk`.

Exactly `round(0.75 * 3080) = 2310` test rows are accepted by descending calibrated confidence. Exact-confidence ties are resolved by stable sample ID. This fixed-coverage statistic does not move the application threshold.

The application threshold `0.892704` is evaluated separately and reports realized test coverage, selective risk, event counts, and in-domain expected cost.

### H4 uncertainty and verdict

The same 5,000-replicate paired bootstrap resamples BANKING77 test rows and recomputes both full risk and the 75%-coverage selective policy within each resample.

- **Supported:** upper confidence bound for selective-minus-full risk is below `0`.
- **Unsupported:** lower confidence bound is at or above `0`.
- **Inconclusive:** the interval overlaps `0`.

## Required descriptive metrics

The confirmatory artifact also reports:

- accuracy;
- macro-F1;
- balanced accuracy;
- top-3 recall;
- 15-bin expected calibration error;
- multiclass Brier score;
- negative log-likelihood;
- selective risk at 50%, 70%, 75%, and 90% confidence-ranked coverage;
- fixed-threshold realized coverage and selective risk;
- fixed-threshold routing-event counts.

These metrics describe the frozen router and are not used for post-test model selection.

## Reproducible execution

The confirmatory evaluator is `benchmarks/routing/evaluate_confirmatory.py`. It verifies the frozen model, data, configuration, and integrity manifest before evaluation.

The corresponding GitHub Actions execution uses read-only repository permissions, the CPU-only A2 dependency lock, preflight validation, and evidence artifact upload. Scientific results are stored as immutable artifacts and then copied into the repository as versioned public evidence.

Infrastructure failures before a scientific result is emitted may be retried. Scientific outputs cannot be used to move thresholds, change calibration, alter costs, or redefine the hypotheses.

## Interpretation discipline

A confirmatory result can support, fail to support, or be inconclusive for the independent H3 in-domain component and for H4.

A negative H3 component result remains valid scientific evidence. A positive H4 result remains conditional on the BANKING77 domain, synthetic event-cost assumptions for cost summaries, and the absence of independent production traffic.

The confirmatory evaluation does not validate live production latency, real-bank economics, real customer harm, or independent OOS generalization.

# Phase 2 Confirmatory Protocol

## Purpose

This document registers the final Phase 2 confirmatory evaluation before the official BANKING77 test split is opened. It does not reopen model selection, calibration selection, OOS construction, cost-weight selection, threshold selection, or the A0-A3 ladder.

The machine-readable source of truth is `configs/models/routing_confirmatory_v1.json`.

## Frozen model and policy

The confirmatory router is `routing-selected-v1`:

- model: A2;
- encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- encoder revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`;
- classifier: frozen multiclass logistic regression from `routing-a2-v1`;
- classifier fitting data: the frozen 7,904-row Phase 2 train partition only;
- calibration: temperature scaling;
- frozen temperature: `0.457974`;
- selected calibrated threshold: `0.892704`;
- frozen raw-A2 H3 comparator threshold: `0.367217`.

The official test result cannot change any of these values.

## Confirmatory data boundary

The only unseen routing-intent partition is the official 3,080-row BANKING77 test split. Its source SHA-256 is `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`; its frozen canonical Helix JSONL SHA-256 is `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468`.

The 160-query `routing-oos-v1` benchmark is **not** independent confirmatory evidence. It was frozen before OOS scoring, which made its development use legitimate, but its results were subsequently inspected and used in operating-policy selection. Reusing those same 160 queries can provide auxiliary scenario context only. They cannot be described as a new confirmatory OOS sample.

This distinction prevents the final paper or repository from overstating the independence of a mixed in-domain/OOS expected-cost statistic.

## H3 confirmatory estimand

H3 asks whether calibration reduces damaging automatic-routing cost.

The fully independent estimand available in Phase 2 is:

`mean BANKING77-test in-domain cost under frozen A2-temperature policy - mean BANKING77-test in-domain cost under frozen A2-raw policy`.

Both policies use the same frozen A2 predictions and the same published `balanced_risk_v1` event costs. The raw policy uses threshold `0.367217`; the calibrated policy uses temperature `0.457974` and threshold `0.892704`. Test data cannot optimize either threshold.

The event costs remain synthetic decision-analysis assumptions:

- correct route: `0`;
- wrong intent, same queue: `2.5`;
- wrong queue: `6`;
- unsafe high-risk wrong automatic route: `20`;
- human escalation: `1`.

The primary H3 statistic intentionally excludes a new OOS-population claim because no unseen OOS sample exists in Phase 2.

**Scope limitation:** this is an independent confirmatory test of the **in-domain component of H3**, not a full independent confirmation of the original mixed in-domain/OOS development cost endpoint. The final Phase 2 interpretation must report the development mixed-cost result and the independent BANKING77 test component separately. A favorable BANKING77 test result cannot by itself relabel the full mixed H3 endpoint as confirmed.

### H3 uncertainty and verdict

A paired nonparametric percentile bootstrap with 5,000 replicates, seed `20260819`, and 95% confidence interval resamples BANKING77 test rows while keeping the raw and calibrated row costs paired.

- **Supported component:** upper confidence bound for calibrated-minus-raw mean in-domain cost is below `0`.
- **Unsupported component:** lower confidence bound is at or above `0`.
- **Inconclusive component:** the interval overlaps `0`.

The development OOS-mixture cost may be discussed only as previously observed auxiliary context and cannot determine the independent component verdict.

## H4 confirmatory estimand

H4 asks whether selective abstention reduces routing risk.

The fully independent primary estimand is:

`selective risk at exactly 75% confidence-ranked BANKING77-test coverage - full-automation BANKING77-test risk`.

Exactly `round(0.75 * 3080) = 2310` test rows are accepted by descending frozen calibrated confidence. Exact-confidence ties are resolved by stable sample ID. This fixed-coverage statistic does not move the application threshold.

The frozen application threshold `0.892704` is evaluated separately and reports its realized test coverage, selective risk, event counts, and in-domain expected cost.

### H4 uncertainty and verdict

The same 5,000-replicate paired bootstrap resamples BANKING77 test rows and recomputes both full risk and the 75%-coverage selective policy within each resample.

- **Supported:** upper confidence bound for selective-minus-full risk is below `0`.
- **Unsupported:** lower confidence bound is at or above `0`.
- **Inconclusive:** the interval overlaps `0`.

## Required descriptive metrics

The confirmatory artifact also reports, without using them for post-test selection:

- accuracy;
- macro-F1;
- balanced accuracy;
- top-3 recall;
- 15-bin expected calibration error;
- multiclass Brier score;
- negative log-likelihood;
- selective risk at 50%, 70%, 75%, and 90% confidence-ranked coverage;
- frozen-threshold realized coverage and selective risk;
- frozen-threshold routing-event counts.

These metrics describe the frozen router. They do not create new tuning opportunities.

## One-shot execution control

The confirmatory evaluator is `benchmarks/routing/evaluate_confirmatory.py`. Its default mode is preflight-only and does not download the test source.

The GitHub Actions workflow `.github/workflows/phase2-routing-confirmatory.yml`:

- has `workflow_dispatch` only;
- has no `pull_request` or `push` trigger;
- has read-only repository permissions;
- requires the exact authorization string `OPEN_FROZEN_TEST_ONCE`;
- verifies the pre-confirmatory Git-blob integrity manifest before preflight;
- runs a no-test preflight before the test-access step;
- reuses the audited CPU-only A2 dependency lock;
- uploads evidence but does not write results back to the repository automatically.

An infrastructure failure before any test metric is emitted may be retried. Once a scientific result has been produced, reruns cannot be used for tuning, selection, threshold movement, or hypothesis redefinition.

## Interpretation discipline

A confirmatory result can support, fail to support, or be inconclusive for the independent H3 in-domain component and for H4. It cannot trigger a new Phase 2 model search.

A negative H3 component result remains a valid scientific result. A positive H4 result remains conditional on the frozen BANKING77 domain, synthetic event-cost assumptions for cost summaries, and the absence of independent production traffic.

The confirmatory evaluation does not validate live production latency, real-bank economics, real customer harm, or independent OOS generalization.

## Phase boundary

Phase 3 retrieval remains forbidden until the registered confirmatory result is permanently recorded, the Phase 2 exit report is closed, public claims are audited, and PR #6 is ready for the Phase 2 merge decision.

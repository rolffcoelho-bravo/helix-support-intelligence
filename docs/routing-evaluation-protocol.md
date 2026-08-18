# Phase 2 Routing Evaluation Protocol

## Objective

Phase 2 produces one calibrated routing configuration and one selective operating policy for the frozen BANKING77-derived Helix contract. The goal is not raw accuracy alone. The selected router must make damaging automatic routes less likely while preserving useful automation coverage.

## Frozen inputs

The Phase 2 development boundary inherits the Phase 1 data contract without modification.

| Partition | Rows | Canonical JSONL SHA-256 | Permitted use |
|---|---:|---|---|
| train | 7,904 | `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3` | Parameter fitting |
| validation | 1,976 | `5a6e2bef72257bb3aa33aba4ca4a93a13738e0a487be88e7846b986b33713455` | Model, calibration, and threshold selection |
| test | 3,080 | `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468` | One registered confirmatory evaluation only |
| quarantine | 123 | excluded | Leakage-control audit only |

The official test split cannot be inspected for error-driven tuning, threshold selection, calibration selection, feature selection, or model-family selection.

## Bounded model ladder

The public ladder is frozen in `configs/models/routing_ladder.json`:

- **A0:** majority and stratified routing rules;
- **A1:** TF-IDF plus logistic regression;
- **A2:** fixed sentence embeddings plus a linear classifier;
- **A3:** one compact transformer classifier;
- **A4:** disabled by default and available only as the single bounded PEFT remediation allowed by the project protocol.

No additional classifier family enters Phase 2 merely because a preferred candidate underperforms.

## Calibration and selective routing

Raw class scores and calibrated probabilities are separate experimental objects. Temperature scaling, isotonic regression, and Platt-style calibration may be compared only where mathematically appropriate for the candidate output.

An automatic route is permitted only when the frozen operating policy accepts the prediction. Otherwise the terminal decision is `ESCALATE_LOW_CONFIDENCE` or another already-declared non-automatic state required by the routing contract.

The operating point is selected on validation data. The test set cannot move it.

## Required measurements

Every eligible routing candidate must report, where defined:

- macro-F1;
- balanced accuracy;
- top-3 recall;
- expected calibration error;
- Brier score;
- out-of-scope AUROC;
- out-of-scope false-positive rate at the declared recall;
- expected routing cost;
- risk-coverage curve;
- confusion pairs and subgroup diagnostics.

Point estimates are not sufficient for the final confirmatory comparison. Uncertainty or paired resampling must be reported for metrics that support it.

## Routing cost semantics

Expected routing cost distinguishes at least these events:

1. correct route;
2. wrong intent inside the correct operational queue;
3. wrong operational queue;
4. unsafe automatic routing of a high-risk case;
5. human escalation.

The numerical weights and final operating threshold are experiment parameters, not facts about real bank economics. They remain unpublished while selection is active and must be disclosed as explicit assumptions when the Phase 2 result is published.

## Out-of-scope evaluation

Out-of-scope detection is evaluated independently from in-domain intent accuracy. The Phase 2 OOS set must be frozen before its results are used for selection and must contain support-like language that is outside the 77 BANKING77 intents, including ambiguous and action-seeking requests.

An OOS benchmark cannot be created or edited after inspecting a model's mistakes unless the change is versioned as a new protocol.

## Registered scientific questions

Phase 2 is responsible for two blueprint hypotheses:

- **H3:** calibration reduces damaging automatic routes; primary endpoint: expected routing cost;
- **H4:** selective abstention reduces risk; primary endpoint: selective risk at fixed coverage.

These questions are evaluated only after the model ladder, metrics, data versions, calibration candidates, and operating-policy semantics are frozen.

## Development checkpoints

Validation-only checkpoints may be published before Phase 2 closes, provided they are clearly separated from release evidence and cannot silently alter the frozen selection protocol.

The first checkpoint, `phase2-a0-a1-validation-v1`, records:

- the A0 most-frequent and seeded stratified lower bounds;
- the fixed A1 TF-IDF plus logistic-regression baseline;
- raw calibration diagnostics;
- risk-coverage evidence;
- principal confusion pairs;
- independent rerun hashes.

The checkpoint lives under `benchmarks/routing/results/`. Its numbers do not populate the README release benchmark table because the confirmatory test remains unopened.

## Selection rule

The winning router is the smallest valid configuration that minimizes damaging routing behavior at useful coverage. A more complex model does not win merely by having the highest uncalibrated accuracy.

If the primary comparison fails, remediation is bounded to the already-declared ladder. The simpler valid baseline is allowed to win.

## Phase 2 exit evidence

Phase 2 closes only when the repository contains:

- reproducible A0-A3 evaluation artifacts;
- a frozen calibration choice;
- a frozen OOS benchmark and report;
- a declared routing cost matrix;
- a risk-coverage report;
- a router model card;
- `/v1/tickets/route` contract tests;
- one frozen model configuration and operating threshold;
- a phase exit report stating whether H3 and H4 were supported, unsupported, or inconclusive.

Until those conditions are met, the README benchmark table remains `pending`.

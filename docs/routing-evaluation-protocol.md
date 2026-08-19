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

The audited second checkpoint, `phase2-a2-validation-v2`, freezes one sentence-embedding representation before evaluation and records:

- `sentence-transformers/all-MiniLM-L6-v2` at revision `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15` as a non-fine-tuned, normalized 384-dimensional feature extractor;
- the same fixed logistic-regression specification used for the A1 comparison;
- CPU-only PyTorch resolution through the official PyTorch CPU index;
- a committed uv script lock for the complete A2 dependency graph;
- validation classification, calibration, risk-coverage, and full-count confusion evidence;
- direct deltas against frozen A1;
- descriptive CPU timing kept separate from stable latency claims;
- exact decision reproducibility plus bounded numerical reproducibility across two independent CPU runs.

The original A2 v1 evidence was removed after a post-execution audit identified a top-20 confusion-count inference bug, a CUDA-enabled lock inconsistent with declared CPU execution, and an overstated byte-identical reproducibility claim. Those corrections did not change A2's aggregate validation metrics or model ranking.

A2 materially improves on A1 in the frozen validation checkpoint and therefore survives as the leading development candidate. This does not select the final router, calibration method, or operating threshold. A1 remains the required simpler reference and A3 remains required by the frozen ladder.

All checkpoint evidence lives under `benchmarks/routing/results/`. Development numbers do not populate the README release benchmark table because the confirmatory test remains unopened.

## Execution audit gate

Every implementation checkpoint must finish with an explicit hostile audit before it is considered complete. The audit must examine at least:

- code-path correctness and hidden assumptions;
- metric definitions and whether summaries can misrepresent underlying counts;
- train/validation/test boundary integrity;
- reproducibility claims at the level actually supported by the evidence;
- dependency and hardware consistency with the declared execution mode;
- CI and workflow behavior, including unnecessary or recursive triggers;
- public wording for claims stronger than the experiment supports.

Any material defect discovered by this audit must be corrected in the same checkpoint. Superseded public evidence must be removed or clearly replaced so that the repository has one authoritative interpretation.

## Execution close report

Every completed execution must end with an explicit research-quality close report rather than a pass/fail statement alone. The close report must include:

- a compact table of the execution's authoritative results or contract outcomes;
- an in-depth interpretation of what the results do and do not establish;
- methodological and engineering limitations that could reduce scientific or industrial value;
- problem-solving options or improvements, with an explanation of why each would increase methodological strength, perceived research value, reproducibility, or publicability;
- a recommendation that distinguishes necessary repairs from optional future enhancements so that negative results are not tuned away;
- the next locked blueprint action and a statement of what remains forbidden until that gate opens;
- a final double-check of code, calculations, result semantics, leakage boundaries, reproducibility, CI behavior, and public wording for errors or misleading claims.

Recommendations produced by this close report do not authorize blueprint drift. A proposed improvement that changes the active phase, model ladder, frozen hypothesis, data boundary, confirmatory protocol, or major methodology requires the normal approval gate before execution.

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

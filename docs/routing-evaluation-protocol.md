# Routing Evaluation Protocol

## Objective

The routing study produces one calibrated routing configuration and one selective operating policy for the frozen BANKING77-derived Helix contract. The goal is not raw accuracy alone. The selected router should reduce damaging automatic routes while preserving useful automation coverage.

## Frozen inputs

| Partition | Rows | Canonical JSONL SHA-256 | Permitted use |
|---|---:|---|---|
| train | 7,904 | `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3` | Parameter fitting |
| validation | 1,976 | `5a6e2bef72257bb3aa33aba4ca4a93a13738e0a487be88e7846b986b33713455` | Model, calibration, and threshold selection |
| test | 3,080 | `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468` | Confirmatory evaluation |
| quarantine | 123 | excluded | Leakage-control verification |

The official test split is excluded from error-driven tuning, threshold selection, calibration selection, feature selection, and model-family selection.

## Model comparison

The public comparison set is defined in `configs/models/routing_ladder.json`:

- **A0:** majority and stratified routing rules;
- **A1:** TF-IDF plus logistic regression;
- **A2:** fixed sentence embeddings plus a linear classifier;
- **A3:** compact transformer classifier.

A more complex model is retained only when the measured benefit justifies the additional complexity.

## Calibration and selective routing

Raw class scores and calibrated probabilities are evaluated separately. Temperature scaling, isotonic regression, and Platt-style calibration are compared where mathematically appropriate for the candidate output.

An automatic route is permitted only when the fixed operating policy accepts the prediction. Otherwise the terminal decision is `ESCALATE_LOW_CONFIDENCE` or another declared non-automatic state required by the routing contract.

The operating point is selected on validation data. The test set cannot move it.

## Required measurements

Every eligible routing candidate reports, where defined:

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

Uncertainty or paired resampling is reported for confirmatory comparisons where the metric supports it.

## Routing cost semantics

Expected routing cost distinguishes at least these events:

1. correct route;
2. wrong intent inside the correct operational queue;
3. wrong operational queue;
4. unsafe automatic routing of a high-risk case;
5. human escalation.

The numerical weights are explicit decision-analysis assumptions and are not presented as real-bank economics.

## Out-of-scope evaluation

Out-of-scope detection is evaluated independently from in-domain intent accuracy. The OOS benchmark contains support-like language outside the 77 BANKING77 intents, including ambiguous and action-seeking requests.

A benchmark used for development selection is not relabeled as independent confirmatory evidence.

## Scientific questions

Two questions are evaluated:

- **H3:** whether calibration reduces damaging automatic-routing cost, using expected routing cost as the primary endpoint;
- **H4:** whether selective abstention reduces routing risk at fixed coverage.

Model, metric, data, calibration, and operating-policy definitions are fixed before confirmatory evaluation.

## Development evidence

The repository preserves development checkpoints for A0 through A3, calibration, OOS behaviour, routing cost, and threshold transfer under `benchmarks/routing/results/`.

The A2 checkpoint uses `sentence-transformers/all-MiniLM-L6-v2` at revision `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15` as a normalized 384-dimensional feature extractor, followed by the same logistic-regression specification used for the A1 comparison.

A3's fixed three-epoch fine-tuning recipe underfit and remains a published negative result rather than being rescued through post-result tuning.

## Reproducibility and integrity checks

Routing evidence is considered complete only when the corresponding artifacts support checks of:

- code-path correctness and assumptions;
- metric definitions and underlying event counts;
- train/validation/test boundary integrity;
- reproducibility claims at the level supported by the evidence;
- dependency and hardware consistency with the declared execution mode;
- CI and workflow behaviour;
- public wording that does not exceed the experiment's evidence.

Superseded evidence is removed or clearly replaced so the repository maintains one authoritative interpretation of each result.

## Selection rule

The selected router is the smallest valid configuration that minimizes damaging routing behaviour at useful coverage. A more complex model does not win merely by having higher uncalibrated accuracy.

The final model configuration, calibration choice, operating threshold, model card, and confirmatory result are versioned in the repository for independent inspection.

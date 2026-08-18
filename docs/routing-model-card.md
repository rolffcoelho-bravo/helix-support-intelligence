# Routing Model Card — `routing-selected-v1`

## Status

`routing-selected-v1` is the frozen **development** routing configuration for Phase 2. It is not a release benchmark, not evidence of real-bank deployment, and not a substitute for the still-sealed confirmatory BANKING77 test.

## Intended use

The router supports bounded English-language fictional digital-banking support tickets. Its job is to predict one of the frozen BANKING77 routing intents, map that intent to an operational queue, expose calibrated confidence, and either automatically route or abstain to human review.

It does not authenticate users, inspect real accounts, move money, block cards, execute refunds, or perform any other irreversible financial action.

## Frozen model pipeline

- Model-ladder identifier: **A2**.
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Encoder revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`.
- Representation: normalized 384-dimensional sentence embedding.
- Classifier: fixed multiclass logistic regression defined by `configs/models/routing_a2.json`.
- Calibration: multiclass temperature scaling.
- Full-validation refit temperature used by the frozen application policy: **0.457974**.
- Frozen automatic-routing threshold: **0.892704**.
- Configuration source of truth: `configs/models/routing_selected_v1.json`.

The classifier family, encoder revision, calibration method, temperature, and routing threshold are frozen before confirmatory evaluation.

## Development evidence

On the frozen validation partition, A2 produced macro-F1 `0.8986`, balanced accuracy `0.8963`, top-3 recall `0.9732`, raw ECE `0.2910`, and raw Brier `0.2501`.

Temperature scaling reduced A2 ECE to approximately `0.0162`, Brier to approximately `0.1398`, and NLL to approximately `0.3350` without changing macro-F1 or top-3 recall.

The frozen 160-query OOS development benchmark produced cross-fitted OOS AUROC of approximately `0.8956` for calibrated A2. At the metric-specific threshold required to obtain at least 95% OOS recall, the in-domain false-positive rate remained approximately `0.4342`. OOS detection is therefore materially imperfect and must not be described as solved.

## Cost-policy result

The public cost matrix uses synthetic decision-analysis units rather than real-bank economics.

For A2, temperature scaling did **not** reduce the registered minimum expected routing cost. Raw A2 reached approximately `0.4029`, while calibrated A2 reached approximately `0.4214`. The Phase 2 H3 development result is therefore **unsupported** on its registered primary endpoint.

Selective routing did materially improve the registered development decision problem. The calibrated A2 policy reduced expected scenario cost from approximately `1.0828` under full automation to `0.4214`, while reducing in-domain selective risk from approximately `9.77%` to `1.21%` at 75% development automation coverage. The Phase 2 H4 development result is therefore **supported**, pending confirmatory evaluation.

## Threshold transfer

Development selection used cross-fitted calibration so a validation row was never scored by a calibrator fit on that same row. Application use requires one full-validation temperature refit. A separate transfer audit therefore mapped the already-selected 75% coverage to the full-refit probability scale without re-optimizing model, calibration method, cost, or coverage.

The resulting full-refit confidence plateau was approximately `(0.892462899, 0.892944242]`. The frozen threshold `0.892704` is the six-decimal rounded midpoint of that plateau. The transfer changed only 6 of 1,976 validation acceptance identities and produced an accepted-set Jaccard of approximately `0.99596`.

## Public routing contract

The framework-neutral endpoint contract is:

- method: `POST`;
- path: `/v1/tickets/route`;
- request schema: `data/contracts/routing_request.schema.json`;
- response schema: `data/contracts/routing.schema.json`.

The domain implementation lives in `src/helix_support_intelligence/domain/routing.py`. It intentionally depends on an injected `IntentScorer` interface rather than importing FastAPI, sentence-transformers, scikit-learn, or another provider/framework SDK into the domain contract.

This separation is deliberate: the Phase 2 contract freezes decision semantics and application policy without coupling the public domain layer to a deployment framework or model-serving implementation that belongs to a later infrastructure stage.

## Decision semantics

For a valid scorer output, the endpoint applies the frozen temperature scaling and ranks all declared intents.

If maximum calibrated intent probability is at least `0.892704`, the terminal decision is `AUTO_ROUTE`. The accepted intent and its frozen operational queue are returned, along with up to three lower-ranked alternatives.

If maximum calibrated intent probability is below `0.892704`, the terminal decision is `ESCALATE_LOW_CONFIDENCE`. The response deliberately sets `intent` and `queue` to `null` so downstream systems cannot accidentally treat a rejected prediction as an automatic route. The three strongest calibrated candidates remain available as diagnostic alternatives for human review.

`out_of_scope_score` is currently the Phase 2 diagnostic score `1 - max(calibrated intent probability)`. It is **not** an independently trained OOS probability and must not be interpreted as one.

If the scorer fails, returns an incomplete intent set, emits invalid probabilities, or otherwise violates the frozen model-output contract, the endpoint terminates as `ESCALATE_SYSTEM_FAILURE` rather than silently routing.

## Operational queue mapping

Intent-to-queue mapping is frozen in `configs/models/routing_operations.json`. High-risk intent tags in that file are evaluation/cost semantics used to identify damaging misroutes; they do not authorize financial actions and do not by themselves create a separate routing decision.

## Known limitations

- Development evidence is based on BANKING77 plus the project's frozen synthetic/support-like OOS benchmark, not live production traffic.
- The OOS benchmark is small and intentionally contains near-boundary cases; category-level diagnostics are not population estimates.
- The selected policy intentionally trades automation coverage for lower routing risk.
- A2 probability computations show bounded CPU floating-point variation. Audited decisions are reproducible even when raw probabilities are not bitwise identical.
- The registered A3 fine-tuning recipe underfit badly; this is a result about that frozen recipe, not a general claim that transformer fine-tuning is inferior.
- The frozen public cost units are scenario assumptions, not measured commercial costs.
- No confirmatory BANKING77 test result has been opened or used to alter this configuration.

## Confirmatory lock

The official BANKING77 test may evaluate `routing-selected-v1` only after the route-contract implementation and final pre-confirmatory audit close. The confirmatory result is not permitted to change the model, encoder revision, calibration method, temperature, threshold, cost matrix, or OOS construction rules.

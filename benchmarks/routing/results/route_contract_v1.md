# Phase 2 Route Contract Checkpoint

> Implementation-contract evidence only. No confirmatory BANKING77 test data were opened and no Phase 2 performance metric, model, calibration choice, OOS benchmark, cost matrix, or operating threshold was reselected.

## Results

| Check | Result |
|---|---|
| Frozen router | `routing-selected-v1` |
| Endpoint | `POST /v1/tickets/route` |
| Request contract | `data/contracts/phase2/routing_request.schema.json` |
| Response contract | `data/contracts/routing.schema.json` |
| Automatic-route rule | calibrated confidence `>= 0.892704` |
| Low-confidence terminal state | `ESCALATE_LOW_CONFIDENCE` |
| Scorer/output failure terminal state | `ESCALATE_SYSTEM_FAILURE` |
| Ruff lint | passed |
| Ruff format check | passed |
| Mypy strict typecheck | passed |
| Pytest | **70 passed** |
| Phase 1 offline data contracts | passed |
| Publication audit | passed |
| Confirmatory test opened | **false** |

## What this checkpoint establishes

The selected Phase 2 routing policy now has a framework-neutral, typed application contract. The domain layer reads the already-frozen `routing-selected-v1` configuration, applies temperature scaling, maps intents to the frozen operational queues, and enforces selective abstention without importing FastAPI, scikit-learn, sentence-transformers, or another deployment/provider SDK into the domain contract.

High-confidence accepted predictions terminate as `AUTO_ROUTE`. Low-confidence predictions deliberately clear `intent` and `queue` before returning `ESCALATE_LOW_CONFIDENCE`; the strongest candidates remain available only as diagnostic alternatives. This prevents a downstream consumer from accidentally treating an abstained prediction as an authorized route.

Scorer exceptions, incomplete intent sets, invalid probabilities, non-mapping outputs, and non-string intent keys fail closed as `ESCALATE_SYSTEM_FAILURE`. The endpoint does not silently continue after malformed model output.

## Hostile audit findings and corrections

The audit found several issues before closure:

1. The first request-schema placement accidentally extended the root Phase 1 contract suite. Because Phase 1 is already closed, the correct repair was **not** to weaken its completeness test. The new request schema was moved into `data/contracts/phase2/`, leaving the frozen Phase 1 root suite unchanged.
2. Temperature was initially validated like a probability and incorrectly restricted to `[0,1]`. Temperature scaling requires a positive finite scalar, so the validator was corrected without changing the frozen value `0.457974`.
3. A failure-path test passed queue strings to a probability-typed fake scorer. The test fixture was corrected even though the scorer raised before using the values.
4. Direct `RoutingPolicyConfig` construction could bypass file-backed validation. The dataclass now validates its own model/version identifiers, temperature, threshold, and queue mapping.
5. A protocol-violating scorer could return malformed runtime objects outside its type annotation. The runtime boundary now validates the output object before calibration and fails closed instead of allowing type errors to escape.
6. The exact threshold comparison is regression-tested. Confidence equal to `0.892704` is accepted; a value immediately below it is escalated.

None of these corrections changes A1/A2/A3 validation metrics, calibration evidence, OOS evidence, routing-cost results, H3/H4 development status, or the frozen threshold.

## Methodological value

This checkpoint increases the research value by connecting the statistical decision rule to explicit executable semantics. A routing paper or public repository is weaker when the reported selective-risk curve and the actual application decision can diverge silently. Here the threshold, calibration policy, queue mapping, output schema, failure posture, and terminal-decision vocabulary are versioned and tested together.

It also strengthens reproducibility and reviewability. The model implementation can later be replaced or served through a different framework without changing the domain decision contract, while malformed adapters fail closed. That separation makes the public project look more like a controlled decision system than a benchmark notebook.

## Remaining limitations

This checkpoint does **not** prove network-service behavior, production latency, concurrency, model-artifact loading, live monitoring, or real-bank effectiveness. The scorer is an injected boundary for contract testing; provider/runtime integration belongs to later authorized infrastructure work. `out_of_scope_score` remains `1 - max(calibrated class probability)`, a diagnostic score rather than an independently calibrated probability of OOS membership.

These limitations should remain explicit rather than being disguised as completed production capability.

## Decision

The router model card and `/v1/tickets/route` implementation-contract tests pass. The Phase 1 contract suite remains frozen and the confirmatory test remains sealed.

## Next locked action

Run the **final pre-confirmatory Phase 2 hostile audit** across the complete frozen model, calibration, OOS, cost, threshold, model-card, contract, workflow, and public-claim surface. Only if that audit passes may the one-shot BANKING77 confirmatory evaluation be proposed/unlocked. Phase 3 retrieval remains forbidden.

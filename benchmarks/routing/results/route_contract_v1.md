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
| Publication boundary check | passed |
| Confirmatory test opened | **false** |

## What this checkpoint establishes

The selected Phase 2 routing policy has a framework-neutral, typed application contract. The domain layer reads the frozen `routing-selected-v1` configuration, applies temperature scaling, maps intents to the frozen operational queues, and enforces selective abstention without importing FastAPI, scikit-learn, sentence-transformers, or another deployment/provider SDK into the domain contract.

High-confidence accepted predictions terminate as `AUTO_ROUTE`. Low-confidence predictions deliberately clear `intent` and `queue` before returning `ESCALATE_LOW_CONFIDENCE`; the strongest candidates remain available only as diagnostic alternatives. This prevents a downstream consumer from accidentally treating an abstained prediction as an authorized route.

Scorer exceptions, incomplete intent sets, invalid probabilities, non-mapping outputs, and non-string intent keys fail closed as `ESCALATE_SYSTEM_FAILURE`. The endpoint does not silently continue after malformed model output.

## Integrity findings and corrections

The implementation review found several issues before the contract was accepted:

1. The first request-schema placement accidentally extended the root Phase 1 contract suite. The new request schema was moved into `data/contracts/phase2/`, leaving the frozen Phase 1 root suite unchanged.
2. Temperature was initially validated like a probability and incorrectly restricted to `[0,1]`. Temperature scaling requires a positive finite scalar, so the validator was corrected without changing the frozen value `0.457974`.
3. A failure-path test passed queue strings to a probability-typed fake scorer. The test fixture was corrected even though the scorer raised before using the values.
4. Direct `RoutingPolicyConfig` construction could bypass file-backed validation. The dataclass now validates its own model/version identifiers, temperature, threshold, and queue mapping.
5. A protocol-violating scorer could return malformed runtime objects outside its type annotation. The runtime boundary now validates the output object before calibration and fails closed instead of allowing type errors to escape.
6. The exact threshold comparison is regression-tested. Confidence equal to `0.892704` is accepted; a value immediately below it is escalated.

None of these corrections changes A1/A2/A3 validation metrics, calibration evidence, OOS evidence, routing-cost results, H3/H4 development status, or the frozen threshold.

## Methodological value

This checkpoint connects the statistical decision rule to explicit executable semantics. A selective-routing system is weaker when the reported risk-coverage rule and the actual application decision can diverge silently. Here the threshold, calibration policy, queue mapping, output schema, failure posture, and terminal-decision vocabulary are versioned and tested together.

It also strengthens reproducibility and reviewability. The model implementation can later be replaced or served through a different framework without changing the domain decision contract, while malformed adapters fail closed. That separation makes the public project a more controlled decision system rather than a benchmark-only demonstration.

## Remaining limitations

This checkpoint does **not** establish network-service behavior, production latency, concurrency, model-artifact loading, live monitoring, or real-bank effectiveness. The scorer is an injected boundary for contract testing. `out_of_scope_score` remains `1 - max(calibrated class probability)`, a diagnostic score rather than an independently calibrated probability of OOS membership.

These limitations remain explicit rather than being presented as completed production capability.

## Decision

The router model card and `/v1/tickets/route` implementation-contract tests pass. The Phase 1 contract suite remains frozen and the confirmatory test remained separate at this checkpoint.

## Subsequent verification

The complete frozen routing surface subsequently passed pre-confirmatory integrity verification before the registered BANKING77 confirmatory evaluation. Those results are reported separately in the corresponding confirmatory evidence files.

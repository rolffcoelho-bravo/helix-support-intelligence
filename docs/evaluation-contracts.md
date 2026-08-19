# Evaluation Contracts

Helix defines evaluation semantics before comparing model families.

## Separation of evidence

Helix evaluates five surfaces independently:

1. **routing** — classification, calibration, abstention, and out-of-scope behavior;
2. **retrieval** — ranking relevance and evidence recall;
3. **generation** — claim support, citation validity, refusal, and completeness;
4. **safety** — leakage, unsafe actions, malformed output, injection resistance, and resource bounds;
5. **system** — latency, cost, reliability, and observable failure behavior.

A fluent generated answer cannot repair a retrieval miss, and aggregate accuracy cannot hide unsafe automatic routes.

## Partition contract

For BANKING77:

- `train` may fit model parameters;
- `validation` may select models and operating thresholds;
- `test` is confirmatory and remains untouched until a fixed candidate is evaluated;
- `quarantine` is excluded from fitting and validation because of cross-split similarity risk.

For HelixBank Policy Corpus v1, the committed golden queries and judgments define the initial retrieval and evidence semantics. Development subsets may be created without altering the frozen records.

For Phase 4 evidence-grounded assistance, A4.0 freezes an intent-level partition before any generation implementation. All four variants of one intent stay in the same partition. The registered split contains 60 development intents / 240 queries and 17 confirmatory intents / 68 queries, stratified so the seven conflict-fixture intents contribute five development and two confirmatory intents. The exact deterministic partition rule, evidence-pack semantics, metrics, hypotheses, adversarial transformations, and execution guard are defined in `configs/models/assistance_protocol_v1.json` and `docs/assistance-evaluation-protocol.md`.

Generation candidate comparison begins with retrieval removed as a source of variance: the evaluator constructs bounded oracle evidence packs from the frozen corpus, while query labels, expected decisions, gold citations, relevance grades, conflict labels, and untrusted-content labels remain hidden from the candidate. Retrieval-plus-assistance system evidence is a later surface and cannot replace generation-isolation evidence.

## Metric ownership

Each workstream declares a primary endpoint before confirmatory evaluation. Secondary metrics provide diagnostic context and cannot replace a failed primary comparison after results are observed.

Results include uncertainty where the metric supports resampling or paired comparison. Failed seeds may be removed only for documented system failures, not because their score is inconvenient.

Phase 4 registers strict grounded success rate as its primary assistance endpoint and uses a paired nonparametric cluster bootstrap by intent. Candidate-reported unsupported claims are telemetry only; the registered support verdict comes from the separately frozen evaluation process.

## Public experiment registry

`experiments/registry.yaml` is the public registry surface for versioned experiments that are appropriate for repository scrutiny. Private research notes, unpublished hypotheses, and publication-preparation material are maintained outside the public repository.

## Test-set opening

Opening a final test set for a fixed candidate is a recorded event. Any subsequent change chosen because of that test result belongs to a new candidate/evaluation version and must be disclosed.

For Phase 4, confirmatory assistance scoring is additionally blocked until the separate A4.1 binding checkpoint freezes exact generator, prompt, decoding/runtime parameters, runtime verifier, independent evaluation verifier, evaluator threshold, latency budget, and cost budget. Confirmatory output cannot be used to relax those choices.

## Reproducibility identifiers

A benchmark result is incomplete unless it identifies, as applicable:

- source and corpus versions;
- split hashes;
- code revision;
- model and prompt versions;
- configuration;
- seeds;
- software environment;
- primary metric;
- evaluation timestamp.

This makes an evaluation claim traceable to immutable inputs rather than to a notebook state.

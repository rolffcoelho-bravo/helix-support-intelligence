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

## Metric ownership

Each workstream declares a primary endpoint before confirmatory evaluation. Secondary metrics provide diagnostic context and cannot replace a failed primary comparison after results are observed.

Results include uncertainty where the metric supports resampling or paired comparison. Failed seeds may be removed only for documented system failures, not because their score is inconvenient.

## Public experiment registry

`experiments/registry.yaml` is the public registry surface for versioned experiments that are appropriate for repository scrutiny. Private research notes, unpublished hypotheses, and publication-preparation material are maintained outside the public repository.

## Test-set opening

Opening a final test set for a fixed candidate is a recorded event. Any subsequent change chosen because of that test result belongs to a new candidate/evaluation version and must be disclosed.

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

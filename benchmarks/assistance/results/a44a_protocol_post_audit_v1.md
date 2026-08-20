# A4.4a pre-execution scientific audit

**Status: PASSED_PRE_EXECUTION_NO_RESULTS**

A4.4a freezes a compositional grounding measurement construct only. No replacement semantic verifier is selected or bound, no semantic verifier inference is performed, no assistance candidate is scored, and the confirmatory partition remains unopened.

## Frozen suite

The final candidate-independent validation suite contains 432 cases across the 60 already-opened development intents: 288 calibration cases and 144 untouched validation cases. The canonical SHA256 is `0ad07e9d08678dbc5fa8b625870d2c3140eef83b0dddb013a4ae479c56bdd90c`.

Gold semantic-pair counts are 317 `ENTAILED`, 60 `CONTRADICTED`, and 360 `UNKNOWN`. All three relation classes occur in both the calibration and validation splits.

## Construct audit

Citation identity, evidence freshness, registered unresolved-conflict metadata, and final aggregation are deterministic components. The atomic semantic-relation interface remains `UNBOUND`.

Registered unresolved conflict is a response-level safety veto and is not a proposition-level semantic-negative label. Atomic contradiction is measured separately. Multi-atom claims require independent support for every atom; concatenated-pack entailment is not admissible.

The future validity protocol measures both the three-way atomic relation task and the final claim-verdict task. Atomic macro F1 and recall for each relation must meet their frozen floors, while citation-invalid, stale-evidence, and unresolved-conflict final verdicts require perfect deterministic accuracy and permit zero false `SUPPORTED` decisions.

## Defects caught before execution

The hostile audit corrected three issues before any semantic verifier binding or inference: ambiguous conflict semantics, omission of explicit three-way atomic relation validity requirements, and five incorrect `UNKNOWN` FAQ gold relations in conflict fixtures where the queue sentence was actually entailed by both cited documents. The last repair changed the suite SHA from `f1404bcd53d214ebe07cd44a0cd1f7d7b1f661f76a85c206f5cde13a69cb83bf` to the final frozen SHA above.

## Guards

Candidate calls: 0. OpenAI calls: 0. Semantic-verifier calls: 0. Replacement-model searches: 0. Confirmatory records inspected: 0. Confirmatory queries scored: 0. A4.4b authorization: false.

The protocol is ready to freeze, but binding a semantic verifier requires a separately versioned and separately approved checkpoint.

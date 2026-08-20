# Phase 4 A4.4a: Compositional grounding measurement

A4.4a freezes the measurement construct that must exist before any replacement grounding verifier is selected or evaluated. It does not score assistance candidates, does not call OpenAI, does not run an NLI model, and does not inspect or score the confirmatory query partition.

## Motivation

A4.2 showed that a single frozen pairwise-NLI score could reject literal source-supported claims, and A4.3a independently rejected the frozen MiniLM2 evaluator after calibration because the untouched validation split failed the registered sensitivity, balanced-accuracy, and unresolved-conflict requirements. A4.4a therefore does not substitute another model into the same monolithic measurement architecture.

The registered construct separates grounding into auditable components whose semantics differ materially:

1. citation identity,
2. evidence freshness,
3. unresolved conflict,
4. atomic semantic support,
5. deterministic aggregation.

Only the atomic semantic relation component may eventually use a learned verifier. That component is deliberately unbound in A4.4a.

## Atomic representation

A factual claim is represented as atomic propositions `a_1, ..., a_m`. For validation fixtures, the atoms are supplied as deterministic gold structure and no learned decomposer is used.

For each atom and cited document, a future semantic verifier may return one relation from:

- `ENTAILED`
- `CONTRADICTED`
- `UNKNOWN`

The semantic-verifier family, revision, and thresholds are not selected in A4.4a.

## Deterministic gates

Citation identity fails when a cited document is absent from the presented evidence pack or an atom has no cited evidence available for checking. Its verdict is `CITATION_INVALID`.

Freshness fails when a claim requiring current policy evidence depends on archived evidence. Semantic similarity cannot make archived evidence current. Its verdict is `STALE_EVIDENCE`.

Conflict fails when registered current conflict evidence is present for the relevant intent, or when future atomic evidence both entails and contradicts the same atom. Its verdict is `CONFLICTING_EVIDENCE`.

These gates are not learned and are not tunable.

## Semantic composition

After deterministic gates pass, every atom must have at least one cited document that entails it. If any atom lacks entailing evidence, the claim is `UNSUPPORTED`.

A multi-document claim may draw support for different atoms from different documents, but each atom is checked independently. Concatenating all evidence and treating one aggregate entailment score as sufficient is not admissible under A4.4a.

`SUPPORTED` is therefore the terminal verdict and is available only when all deterministic gates pass and every atom is independently supported without unresolved contradiction.

The frozen precedence is:

`CITATION_INVALID` → `STALE_EVIDENCE` → `CONFLICTING_EVIDENCE` → `UNSUPPORTED` → `SUPPORTED`.

## Validation suite

The deterministic suite uses only the 60 already-opened A4.0 development intents. No confirmatory query record or candidate output is used.

The suite contains 432 cases:

- 60 literal supported cases,
- 60 paraphrase supported cases,
- 60 contradiction unsupported cases,
- 60 unsupported-approval cases,
- 60 citation-invalid cases,
- 60 multi-document supported cases,
- 60 partially unsupported multi-document cases,
- 7 stale-current-evidence cases,
- 5 unresolved-conflict cases.

The intent-level split is 40 calibration intents and 20 untouched validation intents, stratified by ordinary, archived-FAQ, and conflict-fixture status. This yields 288 calibration cases and 144 validation cases.

## Future binding discipline

A4.4a performs no replacement-model search and no semantic-verifier inference. A future separately approved gate must bind one semantic-verifier family before the A4.4a validation split is opened for evaluation.

The A4.2 candidate results may not influence that binding. The failed A4.3a validation results may not be used to shop among replacement model families. Any future calibration may use only the A4.4a calibration split, and the validation split must remain untouched until the binding is frozen.

## Future hard validity requirements

A future bound compositional evaluator must satisfy every registered requirement, including at least 0.95 macro verdict accuracy, at least 0.98 precision for `SUPPORTED`, at least 0.95 recall for supported cases, and category-specific semantic requirements. Citation-invalid, stale-evidence, and unresolved-conflict cases require perfect deterministic accuracy and zero false `SUPPORTED` verdicts.

Failure of any registered requirement rejects the bound evaluator for candidate selection. Post-validation threshold rescue is not permitted.

## Gate boundary

A4.4a authorizes protocol registration only. It does not authorize A4.4b, candidate comparison, confirmatory scoring, or production adoption.

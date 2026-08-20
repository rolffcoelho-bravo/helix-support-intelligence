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

For each valid cited-presented atom-document pair, a future semantic verifier may return one relation from:

- `ENTAILED`
- `CONTRADICTED`
- `UNKNOWN`

The frozen gold relation is `ENTAILED` when the document appears in the atom's `entailed_by` set, `CONTRADICTED` when it appears in `contradicted_by`, and `UNKNOWN` otherwise. Citation-invalid pairs bypass semantic scoring because the purported cited document is not present evidence.

The semantic-verifier family, revision, and thresholds are not selected in A4.4a.

## Deterministic gates

Citation identity fails when a cited document is absent from the presented evidence pack or an atom has no cited evidence available for checking. Its verdict is `CITATION_INVALID`.

Freshness fails when a claim requiring current policy evidence depends on archived evidence. Semantic similarity cannot make archived evidence current. Its verdict is `STALE_EVIDENCE`.

Registered unresolved conflict is a response-level safety veto. The A4.4a conflict fixture deliberately applies this veto to an otherwise supportable atom. The fixture is therefore not a proposition-level negative label and must not be counted as `CONTRADICTED` merely because conflict metadata is present. Its final verdict is `CONFLICTING_EVIDENCE`.

A second, independent conflict mechanism exists at the atomic level: if valid current evidence produces both `ENTAILED` and `CONTRADICTED` relations for the same atom, the final verdict is also `CONFLICTING_EVIDENCE`.

These deterministic gates are not learned and are not tunable. Atomic semantic relations may still be measured diagnostically on valid cited-presented pairs from stale or registered-conflict fixtures, but they cannot override the higher-priority final-verdict veto.

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

The canonical frozen suite SHA256 is `f1404bcd53d214ebe07cd44a0cd1f7d7b1f661f76a85c206f5cde13a69cb83bf`.

## Measurement definitions

The future evaluation has two distinct layers.

At the atomic layer, the semantic verifier is measured on the three-way relation task. The registered metrics include macro F1 across `ENTAILED`, `CONTRADICTED`, and `UNKNOWN`, plus separate recall for each relation. All four floors are 0.95.

At the final claim-verdict layer, macro case-category accuracy is the unweighted mean of exact-verdict accuracy over the nine registered case categories. This avoids allowing the 60-row categories to numerically swamp the seven stale or five conflict safety cases. The registered floor is 0.95.

`SUPPORTED` precision is true `SUPPORTED` predictions divided by all `SUPPORTED` predictions and must be at least 0.98. `SUPPORTED` recall is true `SUPPORTED` predictions divided by all gold `SUPPORTED` cases and must be at least 0.95.

Citation-invalid, stale-current-evidence, and unresolved-conflict final verdicts each require accuracy 1.0, and those three safety categories permit zero false `SUPPORTED` verdicts.

## Future binding discipline

A4.4a performs no replacement-model search and no semantic-verifier inference. A future separately approved gate must bind one semantic-verifier family before the A4.4a validation split is opened for evaluation.

The A4.2 candidate results may not influence that binding. The failed A4.3a validation results may not be used to shop among replacement model families. Any future parameter calibration may use only the A4.4a calibration split, and the validation split must remain untouched until the binding is frozen.

## Future hard validity requirements

A future bound compositional evaluator must satisfy every registered final-verdict and atomic-relation requirement. Failure of any registered requirement rejects the bound evaluator for candidate selection. Post-validation threshold rescue is not permitted.

## Gate boundary

A4.4a authorizes protocol registration only. It does not authorize A4.4b, candidate comparison, confirmatory scoring, or production adoption.

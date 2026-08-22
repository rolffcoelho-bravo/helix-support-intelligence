# Phase 4 A4.5b-M1 relevance/alignment methodology decision

## Decision

The post-A4.5b methodological repair selects **Scope-Conditioned Evidence Compatibility (SCEC)** as the replacement design for the failed scalar relevance/alignment primitive inside AERF.

SCEC is an internal engineering name. This checkpoint makes no novelty claim and binds no model, weights, threshold, prompt, or learned implementation.

## Why the A4.5b relevance primitive is retired

A4.5b used an MS MARCO cross-encoder raw score both to select a sentence span and to decide whether evidence was relevant. The closed calibration result showed that this scalar retrieval score does not implement the AERF relevance construct. Every `relevant_but_insufficient` case fell below the selected relevance threshold, while more than half of the `cross_document_irrelevance` cases exceeded it. Because no threshold in the full registered grid met every relevance-readiness requirement, the defect cannot be repaired by threshold tuning.

The model itself is not declared generally poor. It is retired from the authoritative AERF relevance role because its score measures a different construct from the one AERF requires.

## SCEC construct

SCEC asks whether a candidate evidence span concerns the same evidential target and scope as an atom before asking whether it proves or disproves that atom.

A compatibility frame contains:

1. subject or entity scope;
2. predicate or attribute scope;
3. target-slot identity or type;
4. required contextual qualifiers, including time, location or organizational scope, conditions, modality, and quantification when present.

Support versus refutation polarity, claim truth value, and completeness of decisive evidence are explicitly excluded from the relevance decision.

A candidate is `RELEVANT` when the mandatory compatibility dimensions match, even when the candidate omits information required for a verdict. It is `IRRELEVANT` when a mandatory entity, predicate, target, or scope dimension mismatches.

## Sufficiency remains separate

After compatibility is established, SCEC-compatible spans form an evidence set. Sufficiency is then evaluated as coverage of the decisive claim content required to license support or refutation. Multiple compatible spans may contribute complementary coverage.

This preserves the distinction that A4.5b failed to measure:

- compatible but incomplete evidence => `RELEVANT + INSUFFICIENT` => `UNKNOWN`;
- scope-incompatible evidence => `IRRELEVANT` => `UNKNOWN`;
- compatible and sufficient support => `ENTAILED`;
- compatible and sufficient refutation => `CONTRADICTED`;
- coexisting sufficient support and refutation => `CONFLICTING_EVIDENCE`.

## Decomposition boundary

A4.5b-M1 does not introduce a new free-form claim decomposition stage. Existing AERF atoms remain the unit of analysis. This avoids changing two constructs at once and follows evidence that decomposition itself can introduce downstream noise.

## Measurement implications

A future SCEC protocol must directly measure scope compatibility rather than infer it from final relation accuracy. It must include explicit hard cases for relevant-but-insufficient evidence, cross-document near-neighbor distractors, same-domain distractors, temporal mismatch, qualifier mismatch, and complementary multi-span evidence.

The future protocol must not weaken A4.5a component-validity standards merely to accommodate the replacement method.

## Data governance

The closed A4.5b calibration set was used to discover this failure geometry. It therefore cannot serve as independent validity evidence for SCEC. It may later be used only for descriptive regression checks after a future SCEC implementation is frozen.

A fresh SCEC calibration construction is required before any future validation. The existing A4.5a fresh-validation partition remains sealed. A4.5c is not repurposed and remains ineligible under the failed A4.5b binding.

## Scope of this checkpoint

Authorized and performed:

- closed-result diagnosis;
- literature audit;
- methodological architecture selection;
- measurement and data-governance registration.

Not authorized or performed:

- semantic inference;
- candidate model comparison;
- model binding;
- threshold search;
- calibration fitting;
- fresh-validation scoring;
- confirmatory inspection or scoring;
- A4.4a or A4.4d rescoring.

The next action is only the registration of a **fresh SCEC measurement and calibration protocol**, and it requires separate approval.

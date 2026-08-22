# Phase 4 A4.5b-M2: Fresh SCEC Measurement and Calibration Protocol

## Status

A4.5b-M2 registers a fresh, calibration-only measurement substrate for Scope-Conditioned
Evidence Compatibility (SCEC). It performs no model inference, binds no implementation,
fits no threshold, and does not construct or score validation data.

Source main SHA:

`1b3a0cd1a0e552bef6ae33b44969d67b11dce7de`

Protocol:

`phase4-assistance-a4.5b-m2-scec-calibration-v1`

## Why a new calibration construction is required

A4.5b closed with zero feasible candidates among 12,050 registered threshold pairs. The
terminal failure was not final relation accuracy. It was the internal relevance
measurement. Relevant-but-insufficient evidence was collapsed into irrelevance while
cross-document distractors were often admitted as relevant.

A4.5b-M1 therefore replaced scalar retrieval relevance with SCEC and explicitly separated
compatibility from sufficiency.

The A4.5b calibration rows are now post-result diagnostic material. They cannot be used as
independent evidence that SCEC works and cannot select a replacement implementation or its
parameters. A fresh SCEC-specific calibration substrate is required before implementation
binding and calibration.

## Literature basis for the measurement design

This protocol is not a novelty claim. Its design is grounded in several established
findings:

1. Akhtar, Schlichtkrull, and Vlachos, *Ev2R: Evaluating Evidence Retrieval in Automated
   Fact-Checking*, TACL 2026, argues that evidence evaluation should assess both alignment
   with reference evidence and whether evidence reliably supports the downstream verdict.
   https://aclanthology.org/2026.tacl-1.25/

2. Atanasova et al., *Fact Checking with Insufficient Evidence*, TACL 2022, shows that
   missing evidence must be measured explicitly and that omitted modifiers can be
   especially difficult. This motivates treating compatible but incomplete evidence as a
   sufficiency failure rather than an irrelevance decision.
   https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00486/112498/

3. Wuehrl et al., *What Makes Medical Claims (Un)Verifiable?*, EACL 2024, studies
   subject-relation-object structure and reports that evidence search benefits from
   explicit entity and relation constraints. This supports direct compatibility
   measurement over entity and predicate dimensions.
   https://aclanthology.org/2024.eacl-long.124/

4. Barik, Hsu, and Lee, *Evidence-Based Temporal Fact Verification*, 2024, treats temporal
   information as a first-class part of evidence retrieval and verification. This supports
   an explicit temporal-scope dimension rather than leaving time to generic similarity.
   https://arxiv.org/abs/2407.15291

5. Li et al., *Minimal Evidence Group Identification for Claim Verification*, TrustNLP
   2025, formalizes evidence groups whose pieces collectively provide complete support.
   This motivates set-level sufficiency and complementary evidence coverage.
   https://aclanthology.org/2025.trustnlp-main.8/

6. Zheng et al., *Evidence Retrieval is almost All You Need for Fact Verification*,
   Findings of ACL 2024, reports that heuristic semantic-similarity retrieval can produce
   task-irrelevant evidence. This reinforces the decision not to use one scalar retrieval
   score as the authoritative SCEC compatibility gate.
   https://aclanthology.org/2024.findings-acl.551/

## Fresh calibration corpus

The corpus is fictional and candidate-independent. It is an offline measurement fixture,
not evidence of real-world support-system impact.

Corpus ID:

`helix-scec-calibration-corpus-v1`

Seed:

`20260822`

The construction contains:

- 48 calibration units
- 768 atom-to-evidence pair rows
- 384 evidence-set rows
- 384 claim-composition rows
- no validation partition

Frozen hashes:

- units: `ece3b03fe215cb4847ec1e8ed71f05885bddc6807fe7a87691af55b05dc75d84`
- pair rows: `2ee18830fb2510aae85a936368adea72de145f82b2282831e0d2ee841546e12f`
- evidence-set rows: `698b348b4bb5d5b00e597a7fcea144ce10ecd62e40036602ae5fa25577606d61`
- claim rows: `3dc8b01f797ca1b97ae5330929d8b462bf3b85b3d0788433c23026bf8bef262e`

## Pair-level SCEC construct

Each pair is annotated over eight registered compatibility dimensions:

- entity
- predicate
- target-slot identity
- temporal scope
- location scope
- organizational scope
- conditional scope
- modality or quantification scope

Dimension states are `MATCH`, `MISMATCH`, or `UNSPECIFIED`.

`UNSPECIFIED` is deliberately different from `MISMATCH`. If evidence targets the correct
atom but omits a required temporal or conditional qualifier, it remains compatible and
must fail later at sufficiency. This is the direct repair for the A4.5b failure mode.

Pair subtypes independently probe:

- direct full-scope support
- direct full-scope refutation
- entity mismatch
- predicate mismatch
- target-slot mismatch
- temporal mismatch
- location mismatch
- organizational mismatch
- conditional mismatch
- modality or quantification mismatch
- compatible but missing target value
- compatible but missing temporal scope
- compatible but missing conditional scope
- same-domain irrelevance
- cross-document irrelevance
- context-contaminated support

Every subtype has 48 rows.

## Evidence-set sufficiency construct

SCEC does not assume that one top-ranked span is the complete evidence state. Compatible
spans may jointly cover decisive slots.

The evidence-set layer measures:

- single-span complete support
- single-span complete refutation
- compatible incomplete evidence with a missing value
- compatible incomplete evidence with a scope gap
- complementary two-span support
- complete support mixed with an irrelevant distractor
- support-refute conflict
- multi-span evidence that still leaves a scope gap

The decisive coverage ledger contains entity, predicate, target-slot identity, target
value, temporal scope, location scope, organizational scope, conditional scope, and
modality or quantification scope.

An unresolved decisive slot forces `INSUFFICIENT`. Compatible support and refutation
coexisting in the same evidence state force `CONFLICTING_EVIDENCE`.

## Calibration readiness

The registered readiness gates are intentionally component-level. Final verdict accuracy
cannot compensate for a failed internal compatibility construct.

The protocol requires, among other gates:

- compatibility macro F1 at least 0.90
- compatible and incompatible recall at least 0.90
- each explicit scope-mismatch rejection rate at least 0.90
- relevant-but-insufficient compatibility recall at least 0.95
- cross-document and same-domain irrelevance rejection at least 0.95
- sufficiency macro F1 at least 0.90
- unresolved scope-gap insufficiency recall at least 0.95
- complementary evidence sufficiency recall at least 0.90
- final relation macro F1 and each relation recall at least 0.90
- SUPPORTED precision at least 0.98
- exact safety accuracy for conflict, citation-invalid, stale-evidence, and registered
  conflict cases
- zero false SUPPORTED outcomes on registered safety cases

The complete machine-readable requirement set is frozen in
`configs/models/assistance_grounding_a45bm2_v1.json`.

## Future validity floors

A4.5b-M2 also freezes minimum floors for a later independent validity checkpoint without
constructing that validation data. Core compatibility, sufficiency, polarity, and final
relation metrics remain at 0.95 or higher, with SUPPORTED precision at 0.98 and exact
safety gates.

These floors do not authorize validation.

## Data governance

A4.5b-M2 makes the following boundaries explicit:

- A4.5b failure rows may not select a future SCEC model or parameter.
- A4.5b-M2 calibration rows may be used only after one authoritative SCEC implementation
  is frozen under a separately approved checkpoint.
- A4.5b-M2 calibration is not independent generalization evidence.
- A4.5a fresh validation remains sealed and unscored.
- No new validation corpus is constructed here.
- The 68-query confirmatory partition remains unopened and unscored.
- A4.5c is not repurposed for SCEC.

## Next checkpoint

The registered next action is:

**A4.5b-M3: SCEC implementation binding and calibration-only execution protocol**

A4.5b-M3 is not authorized by this registration. It requires separate approval and must
freeze exactly one authoritative implementation before any calibration score or threshold
is exposed.

# Phase 4 A4.5b-M3: SCEC implementation binding and calibration-only execution

## Status

A4.5b-M3 binds exactly one authoritative Scope-Conditioned Evidence Compatibility (SCEC) implementation before any A4.5b-M2 calibration outcome is exposed. It authorizes one calibration-only execution against the frozen 48-unit M2 substrate. It does not authorize fresh validation, confirmatory scoring, post-result model substitution, or A4.5c repurposing.

SCEC remains an internal engineering name. No methodological novelty claim is made.

## Scientific lineage

A4.5b failed calibration readiness because scalar passage-ranking relevance could not separate evidence that was genuinely irrelevant from evidence that addressed the right atom but remained incomplete. A4.5b-M1 therefore replaced scalar retrieval relevance with SCEC. A4.5b-M2 then registered a fresh candidate-independent calibration substrate that measures explicit compatibility dimensions, set-level coverage and sufficiency, polarity, final relation, and claim-level safety behavior.

A4.5b-M3 now binds the implementation before scoring that fresh calibration substrate.

## Authoritative semantic primitive

The sole learned semantic primitive is:

- model: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
- revision: `91562024e753ad76646a2d0dfcbb26801aa945fe`
- weights: `model.safetensors`
- weights SHA256: `6e8f2af78c828dcbd5243aac40fb87430376f0b8a9c288f4993df3ea3558d557`
- license: MIT
- native labels: `0 = entailment`, `1 = not_entailment`
- runtime: CPU FP32
- batch size: 32
- maximum sequence length: 256

The selection was made from model documentation and architecture suitability before any M2 calibration execution. The protocol prohibits model-family comparison after binding.

The model is used as a binary entailment scoring primitive. Registered candidate hypotheses are compared by softmax over their entailment logits. The native `not_entailment` output is never mapped directly to UNKNOWN.

## Compatibility factorization

For each candidate sentence, the verifier scores three registered hypotheses for each of eight dimensions: entity, predicate, target slot, temporal scope, location scope, organizational scope, conditional scope, and modality/quantification scope.

The three semantic labels are MATCH, MISMATCH, and UNSPECIFIED. A global mismatch threshold is the only fitted compatibility parameter. A dimension is MISMATCH only when the MISMATCH hypothesis has the highest registered probability and that probability meets the global mismatch threshold. Otherwise UNSPECIFIED wins when its probability exceeds MATCH; otherwise the dimension is MATCH.

Overall evidence is COMPATIBLE exactly when none of the eight dimensions is MISMATCH. UNSPECIFIED therefore remains compatible and can become an evidence-coverage gap at the sufficiency stage.

## Minimal compatible span

Evidence text is sentence segmented by the frozen regular expression `(?<=[.!?])\s+`. Each sentence receives the threshold-independent rank `1 - max_dimension(P(MISMATCH))`. The sentence with the highest rank is selected, with the earliest sentence winning an exact tie. After selection, the registered mismatch threshold determines whether that span is COMPATIBLE.

## Set-level evidence coverage

Nine decisive slots are registered: entity, predicate, target-slot identity, target value, temporal scope, location scope, organizational scope, conditional scope, and modality/quantification scope.

For each compatible span and slot, the verifier compares exactly two registered hypotheses, COVERED and MISSING. A single global coverage threshold is fitted. A slot is covered when the registered COVERED probability meets that threshold.

For an evidence set, incompatible spans are discarded, compatible spans may contribute complementary evidence, slot coverage is unioned across compatible spans, and the set is SUFFICIENT only when all nine decisive slots are covered. Irrelevant evidence therefore cannot fill a missing slot.

## Polarity and conflict

For a sufficient compatible span, two fixed hypotheses are compared: SUPPORTS and REFUTES. Polarity uses argmax and has no fitted threshold.

If at least one individually sufficient compatible span SUPPORTS and at least one individually sufficient compatible span REFUTES, the set is immediately CONFLICTING and maps to `CONFLICTING_EVIDENCE`. Otherwise a sufficient compatible set is classified on its compatible spans in registered order. All non-empty subsets needed by the frozen M2 evidence sets are scored before parameter selection, so threshold search does not trigger adaptive inference.

## Claim composition

Claim-level metrics use predicted evidence-set relations, not the gold `set_relations` field. The source-set mapping is frozen as C01→S01, C02→S02, C03→S03, C04→S05, C05→S07, with citation-invalid, stale-evidence, and registered-conflict gates applied to S01 for C06-C08.

## Registered calibration grid

Only two global thresholds may be fitted:

- mismatch threshold: 0.34 through 0.90 inclusive, step 0.02, 29 values;
- coverage threshold: 0.50 through 0.90 inclusive, step 0.02, 21 values.

Total preregistered candidates: **609**.

No additional parameter, prompt, temperature, model, class-specific threshold, or validation feedback may enter selection.

## Selection rule

Candidate selection is lexicographic and frozen before execution: prefer candidates passing every M2 readiness requirement; maximize the number of requirements passed; maximize final-relation macro F1; maximize the minimum core recall across compatibility, sufficiency, polarity, and final-relation recalls; maximize claim-category macro accuracy; maximize compatibility macro F1; maximize sufficiency macro F1; maximize polarity macro F1; then prefer higher coverage and mismatch thresholds.

If no candidate satisfies every readiness requirement, the best candidate under the same deterministic ordering is recorded diagnostically and scientific readiness is FAIL.

## Scientific outcomes

PASS: `PASSED_SCEC_CALIBRATION_READINESS_PARAMETERS_FROZEN`.

FAIL: `FAILED_SCEC_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED`.

A PASS freezes the selected M3 calibration parameters but does not authorize fresh validation. A FAIL freezes the negative calibration result. No post-result rescue, prompt rewrite, model replacement, unregistered threshold search, or validation execution is allowed.

## Sealed data

A4.5b-M3 may score only the A4.5b-M2 calibration substrate: 48 units, 768 atom-evidence pairs, 384 evidence-set rows, and 384 claim rows.

A4.5a fresh validation, the 68-query confirmatory partition, A4.4 validation rows, and the closed A4.5b calibration are prohibited from selection or scoring in M3.

## Reproducibility and audit

The workflow enforces exact predecessor lineage, runs the fail-closed preflight, freezes the PEP 723 runtime lock and registered scientific-input hashes, verifies the exact model-weight SHA256, writes raw semantic scores before threshold selection, evaluates all 609 candidates, re-verifies frozen inputs, reconstructs the registered arithmetic from the immutable raw scores, freezes artifact checksums, and uploads immutable evidence.

Scientific FAIL remains a valid scientific result. Only workflow, artifact-integrity, or reconstruction failures are infrastructure failures.

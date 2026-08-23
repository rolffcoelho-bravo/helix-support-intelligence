# Phase 4 A4.5b-M6: TPAG Implementation Binding and Calibration-Only Execution

## Status

A4.5b-M6 binds exactly one authoritative Typed Proposition Alignment Graph (TPAG)
implementation and authorizes one calibration-only execution on the frozen A4.5b-M5
corpus. It does not authorize A4.5a fresh validation, the 68-query confirmatory partition,
A4.5c, model-family comparison, prompt search, post-result grid expansion, or rescue after
a failed calibration-readiness result.

Source main SHA:

`9070fc4cf0447077a20c7e576e49e9ba5f0536ba`

Protocol:

`phase4-assistance-a4.5b-m6-tpag-binding-calibration-v1`

TPAG remains an internal engineering architecture name and no novelty claim is made.

## Why this binding is materially different from M3

M3 asked one zero-shot NLI primitive to decide compatibility, coverage, and polarity through
many competing hypotheses. The closed M3 result showed that this universal semantic
classification role was structurally unreliable, especially for predicate, entity,
target-slot, conditional, and modality mismatch discrimination and for refutation/conflict.

M6 therefore does not replace M3 with another universal NLI prompt. The authoritative TPAG
pipeline assigns deterministic authority to measurements that are explicit and typed, and
uses one learned model only when literal structured comparison cannot decide whether two
predicate expressions denote the same operation.

The design is consistent with the M4/M5 architecture decision and with two external
research observations already frozen in the research record: query conditioning matters
for NLI-style verification, and evidence sufficiency/minimality is a set-level problem.
Relevant references are Canby et al., *Benchmarking Query-Conditioned Natural Language
Inference*, Findings ACL 2025, DOI `10.18653/v1/2025.findings-acl.765`, and Li et al.,
*Minimal Evidence Group Identification for Claim Verification*, TrustNLP 2025, DOI
`10.18653/v1/2025.trustnlp-main.8`.

## Authoritative pipeline

There is exactly one implementation:

`tpag-deterministic-typed-parser-plus-qc-nli-residual-v1`

The stages are:

1. deterministic proposition segmentation and decontextualization;
2. deterministic typed frame extraction for explicit registered slots;
3. deterministic exact/closed-class comparison for eight typed dimensions;
4. one query-conditioned learned residual for unresolved predicate equivalence;
5. deterministic scope compatibility and decisive-slot coverage;
6. deterministic minimal sufficient evidence-group enumeration;
7. deterministic support/refutation/conflict composition;
8. deterministic claim and safety-gate composition.

No threshold exists in stages 1, 2, 5, 6, 7, or 8.

## Deterministic proposition and frame extraction

Sentence boundaries use the frozen regex:

`(?<=[.!?])\s+`

The decontextualizer may use only explicit document/source-local information:

- explicit alias declarations;
- the nearest preceding explicit active-record declaration for the registered pronoun
  carry-forward rule;
- removal of non-semantic parenthetical appositives;
- Unicode/whitespace/case normalization.

The target selector chooses propositions whose canonical entity equals the claim entity and
that contain sufficient typed information to be a claim-bearing proposition. It does not
free-form redecompose the existing AERF claim atom.

Typed extraction covers the nine M5 decisive slots:

1. entity or subject;
2. predicate or event;
3. target-slot identity;
4. target value;
5. temporal scope;
6. location scope;
7. organizational scope;
8. conditional scope;
9. modality or quantification.

The fixed support-policy schema maps canonical operations to their registered deadline
slots. This schema contains no fitted parameter.

## Typed alignment

Exact canonical equality decides entity, target-slot identity, target value, temporal,
location, organizational, and conditional relations. The closed M5 modality equivalence
class treats the four registered `must`/`required-to` and `every`/`all` formulations as
semantically equivalent. `may process some` remains explicitly different.

An explicit mismatch on a mandatory scope/identity slot makes a proposition
`INCOMPATIBLE`. `UNSPECIFIED` preserves compatibility but leaves that decisive slot
uncovered. A target-value mismatch does not by itself make evidence irrelevant: when the
same target and scope are established, it remains compatible and is eligible for
`REFUTES`.

## The single learned residual

The only learned primitive is:

- model: `cross-encoder/nli-deberta-v3-base`
- revision: `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`
- weights: `model.safetensors`
- weights SHA-256:
  `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`
- tokenizer asset: `spm.model`
- tokenizer SHA-256:
  `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd`
- license: Apache-2.0
- native labels: contradiction, entailment, neutral
- CPU FP32
- batch size 32
- maximum sequence length 256.

The model is trained for three-way NLI on SNLI and MultiNLI and its model card reports
92.38% SNLI-test accuracy and 90.04% MNLI-mismatched accuracy. Those external benchmark
results are used only to justify the pre-calibration binding; M5 outcomes were not used to
compare model families.

For an unresolved predicate edge the frozen query is:

`Claim operation: {claim_value}. Evidence operation: {evidence_value}. Evidence context: {evidence_text}`

The frozen hypothesis is:

`The evidence operation expresses the same operation as the claim operation.`

Decision rule for a registered alignment threshold `t`:

- entailment probability >= `t` -> `MATCH`;
- contradiction probability >= `t` -> `MISMATCH`;
- otherwise -> `UNSPECIFIED`.

The learned residual has no authority to override an explicit deterministic scope mismatch
or to directly emit a final support verdict.

## Evidence-group reasoning

Only compatible propositions may supply decisive-slot coverage. All non-empty subsets of
compatible evidence spans are enumerated, and complete coherent subsets are retained only
when no proper subset is already complete. This yields the predicted minimal sufficient
evidence groups.

A single-scope qualifier that explicitly conflicts with the claim scope cannot be used to
fill that slot and marks the cross-span construction incoherent. A fully scoped proposition
under a different condition is treated as evidence for a different scope and does not
create a conflict with a valid support proposition for the registered claim scope.

Same-scope sufficient support and sufficient refutation force
`CONFLICTING_EVIDENCE`.

## Polarity

M6 exposes no polarity threshold. Polarity is computed only after scope compatibility and
complete decisive-slot coverage:

- complete coverage with no target-value mismatch -> `SUPPORTS`;
- complete coverage with explicit target-value mismatch -> `REFUTES`;
- incomplete compatible evidence -> `UNRESOLVED`.

This prevents lexical dissimilarity alone from becoming contradiction.

## Registered calibration search

M5 allowed at most three scalar thresholds and at most 343 joint candidates. The bound M6
pipeline exposes only one confidence threshold:

`alignment_confidence_min`

The two unused thresholds are disabled because extraction and polarity are deterministic.
The seven registered values are:

`0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90`

Therefore M6 evaluates exactly **7 candidates**, not 343.

All residual NLI outputs are computed once and written to an immutable raw-score artifact
before any M5 gold labels are read for metric evaluation or threshold selection. Threshold
selection then performs no additional model inference.

Class-specific thresholds, slot-specific thresholds, prompt search, model-family search,
and post-result grid expansion remain prohibited.

## Selection and readiness

All 56 M5 readiness requirements remain unchanged. Candidate selection follows the frozen
M5 order:

1. all 56 requirements satisfied;
2. number of requirements passed;
3. final-relation macro F1;
4. minimum safety-critical recall;
5. claim-category macro accuracy;
6. minimal sufficient-group exact match;
7. scope-compatibility macro F1;
8. slot-relation macro F1;
9. target-proposition F1;
10. higher alignment threshold.

If at least one candidate satisfies all requirements, the scientific status is:

`PASSED_TPAG_CALIBRATION_READINESS_PARAMETERS_FROZEN`

If no candidate satisfies all requirements, the scientific status is:

`FAILED_TPAG_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED`

A failure does not authorize threshold rescue, prompt changes, model substitution, or
validation-assisted tuning.

## Frozen runtime

The scientific execution uses:

- Python >=3.12,<3.13;
- `huggingface-hub==0.36.2`;
- `safetensors==0.8.0`;
- `sentencepiece==0.2.1`;
- `torch==2.13.0`;
- `transformers==4.57.6`;
- `protobuf==6.33.6`;
- CPU threads = 1;
- tokenizer parallelism disabled.

The protobuf dependency is registered before execution rather than introduced as a
post-failure repair.

## Sealed boundaries

M6 authorizes only the 64-unit M5 calibration construction:

- 512 proposition rows;
- 1,280 alignment rows;
- 768 evidence-group rows;
- 640 claim/safety rows.

The following remain at zero:

- model-family comparisons;
- prompt searches;
- M2/M3 calibration rescoring;
- A4.5a fresh-validation access or scoring;
- confirmatory record inspection or scoring;
- future-validation construction;
- post-result rescue.

A4.5c remains ineligible and is not repurposed.

No post-M6 checkpoint is pre-authorized. The admissible next step depends on the registered
M6 calibration result and requires separate approval.

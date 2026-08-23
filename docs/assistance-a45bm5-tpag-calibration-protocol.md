# Phase 4 A4.5b-M5: Fresh TPAG Measurement and Calibration Protocol

## Status

A4.5b-M5 registers a fresh, calibration-only measurement substrate for the Typed
Proposition Alignment Graph (TPAG) architecture selected in A4.5b-M4. It performs no
semantic inference, binds no parser or model, fits no threshold, constructs no validation
partition, and does not inspect the confirmatory partition.

Source main SHA:

`2299935cdf940bc0f3774e7d5c35d4a5cd297d87`

Protocol:

`phase4-assistance-a4.5b-m5-tpag-calibration-v1`

TPAG remains an engineering architecture name rather than a novelty claim.

## Why another fresh calibration construction is required

The M2 SCEC calibration corpus was used to bind and evaluate the M3 implementation, and
its resulting failure subsequently informed the M4 architecture decision. It can no
longer provide candidate-independent calibration-readiness evidence for TPAG. Reusing it
to choose a TPAG implementation or its parameters would turn post-result diagnosis into
model selection.

A4.5b-M5 therefore creates a new fictional, candidate-independent calibration fixture
whose vocabulary, units, measurement layers, and subtype structure are distinct from the
M2/M3 corpus. The older corpus remains useful only for historical failure diagnosis and,
after a future TPAG implementation and parameters are frozen, non-independent regression
checks.

## Measurement design

The protocol separates four error surfaces that M3 collapsed into one semantic scoring
mechanism:

1. proposition extraction and decontextualization;
2. typed claim/evidence slot alignment;
3. evidence-group coverage, minimality, sufficiency, polarity, and conflict;
4. claim composition and deterministic safety gates.

This separation follows the architecture selected in M4 and is consistent with recent
fact-verification research. Claim decomposition can help or hurt depending on extraction
quality and verifier interaction, so M5 preserves the existing AERF claim atoms rather
than introducing verifier-adaptive free-form decomposition. Document-level
claim/evidence decontextualization has also been shown to affect evidence retrieval and
verification quality. Relevant references include:

- Hu, Long, and Wang, *Decomposition Dilemmas: Does Claim Decomposition Boost or Burden
  Fact-Checking Performance?*, NAACL 2025, DOI `10.18653/v1/2025.naacl-long.320`.
- Lu et al., *Optimizing Decomposition for Optimal Claim Verification*, ACL 2025, DOI
  `10.18653/v1/2025.acl-long.254`.
- Deng, Schlichtkrull, and Vlachos, *Document-level Claim Extraction and
  Decontextualisation for Fact-Checking*, ACL 2024, DOI
  `10.18653/v1/2024.acl-long.645`.
- Wanner, Van Durme, and Dredze, *DnDScore*, EMNLP 2025, DOI
  `10.18653/v1/2025.emnlp-main.1205`.

The protocol also treats learned semantic matching as a typed, query-conditioned residual
operation rather than a universal whole-pair verdict. This is motivated by work on
query-conditioned NLI and conditional reasoning, including Canby et al., Findings ACL
2025, DOI `10.18653/v1/2025.findings-acl.765`, and Kim, Rahimi, and Allan, Findings EMNLP
2023, DOI `10.18653/v1/2023.findings-emnlp.456`.

Evidence-group coverage is registered independently from pairwise compatibility. Li et
al., *Minimal Evidence Group Identification for Claim Verification*, TrustNLP 2025, DOI
`10.18653/v1/2025.trustnlp-main.8`, formalizes the need to identify groups whose members
jointly provide complete support. M5 uses only the high-level measurement lesson: group
sufficiency and minimality must be measured explicitly rather than inferred from the
highest-scoring sentence.

## Fresh calibration corpus

Corpus ID:

`helix-tpag-calibration-corpus-v1`

Seed:

`20260823`

The construction contains:

- 64 fictional calibration units;
- 512 proposition-extraction rows;
- 1,280 typed-alignment rows;
- 768 evidence-group rows;
- 640 claim-composition and safety rows;
- zero validation rows.

Frozen hashes:

- units: `7ffe0ccd27c4f837374adb9b9041829984d16251bb68f7157b6dd16dffa3fa9f`
- proposition rows:
  `eabc82d897b3720556ceeb6d4034c9e7c3a6dec6ea7e8ab846eb08930cbbd80b`
- alignment rows:
  `c2b0bb66dc13a8c09560730337152ee5d0978b5a995bc7af3beb7782988c2845`
- evidence-group rows:
  `0927bb13d8f8d64ec692f349a6dec403e75faec87ca4812fe07c975e0d462c89`
- claim rows:
  `20063b3fc84f306b43adf3fd1a4a0a255c2aeb7e04863c3c98158da7dfb9324c`

The fixture is synthetic only in the narrow sense of being a deterministic fictional test
construction. It is not real-world evidence and cannot support claims about operational
impact.

## Layer 1: proposition extraction and decontextualization

Eight balanced subtypes probe clean targets, targets beside unrelated material, entity
aliases, pronoun resolution, coordinated propositions, parenthetical context, and documents
with no target proposition. Required outputs are surface proposition boundaries, target
proposition selection, decontextualized target propositions, and typed target frames.

The existing AERF atom remains the claim unit. A future extractor may recover context
needed to interpret evidence, but may not freely rewrite the claim into verifier-preferred
atoms.

Calibration floors include proposition precision and recall of at least `0.95`, target
proposition F1 of at least `0.95`, decontextualized target-frame slot accuracy of at least
`0.93`, and a no-target false-positive rate no greater than `0.02`.

## Layer 2: typed slot alignment

Every claim and evidence proposition is represented with nine registered slots:

### Scope and identity slots

1. entity or subject;
2. predicate or event;
3. target-slot identity;
4. temporal scope;
5. location scope;
6. organizational scope;
7. conditional scope;
8. modality or quantification.

### Content slot

9. target value.

Each slot relation is one of `MATCH`, `MISMATCH`, or `UNSPECIFIED`.

A4.5b-M5 makes an important distinction explicit. An explicit mismatch in a mandatory
scope/identity slot makes the evidence proposition `INCOMPATIBLE` for that claim target.
An `UNSPECIFIED` required slot does not make the evidence irrelevant; the proposition
remains compatible but cannot cover that slot.

A conflicting **target value** is different. If the entity, predicate, target-slot
identity, and applicable scope qualifiers align, a target-value mismatch remains
scope-compatible and becomes eligible for `REFUTES`. This prevents TPAG from recreating
the earlier failure in which semantic difference and evidential irrelevance were
collapsed into the same decision.

The alignment corpus is balanced at 640 `COMPATIBLE` and 640 `INCOMPATIBLE` rows. It
independently probes all eight scope/identity mismatches, target-value refutation, six
registered `UNSPECIFIED` coverage gaps, explicit aliases, predicate paraphrases,
same-domain near misses, and cross-unit distractors.

Calibration floors require slot-relation macro F1 of at least `0.92`, scope-compatibility
macro F1 of at least `0.93`, compatible and incompatible recall of at least `0.93`, each
explicit scope mismatch rejection rate of at least `0.95`, and preservation of each
registered missing-slot case as compatible at at least `0.98`. Same-domain near-miss and
cross-unit distractor rejection must each reach `0.98`.

## Layer 3: evidence-group coverage and minimality

Compatible propositions may jointly cover decisive claim slots. Incompatible propositions
may never supply coverage. A sufficient evidence group must cover all decisive slots with
cross-proposition scope coherence.

The 12 group subtypes measure single-span support/refutation, incomplete evidence,
complementary two- and three-span support, distractor robustness, unresolved multi-span
coverage, same-scope support/refute conflict, different-condition non-conflict, redundant
support minimality, and cross-span scope incoherence.

The protocol requires both the covered/missing slot sets and the minimal sufficient
evidence group. This makes overinclusive evidence selection visible even when the final
verdict happens to be correct.

A support/refute conflict is registered only when sufficient support and refutation apply
to the same aligned target and scope. A refuting proposition under a different condition
must not create `CONFLICTING_EVIDENCE`.

Calibration floors include sufficiency macro F1 at least `0.92`, insufficient recall at
least `0.95`, exact missing-slot accuracy at least `0.95`, minimal-sufficient-group exact
match at least `0.90`, complementary two- and three-span support recall at least `0.92`,
cross-span scope-incoherence insufficiency recall at least `0.98`, and exact same-scope
conflict detection.

## Layer 4: polarity, relation, claim composition, and safety

Polarity is measured only on evidence that has already passed scope compatibility and
sufficiency. Explicit target-value inequality may support deterministic refutation only
after the proposition target and scope are established.

The final relation vocabulary remains `ENTAILED`, `CONTRADICTED`, `UNKNOWN`, and
`CONFLICTING_EVIDENCE`. Claim-level deterministic gates retain citation-invalid,
stale-evidence, and registered-conflict behavior.

Calibration floors require polarity macro F1 at least `0.92`, refutation recall at least
`0.95`, final-relation macro F1 at least `0.92`, contradicted recall at least `0.95`,
unknown recall at least `0.95`, exact conflicting-evidence recall, claim-category macro
accuracy at least `0.95`, `SUPPORTED` precision at least `0.99`, and zero false-SUPPORTED
safety outcomes.

## Deterministic versus learned boundary

The future implementation may deterministically use only low-risk registered operations,
including normalization, canonical identifier equality, trusted explicit alias tables,
exact integer and four-digit-year parsing, the closed `must`/`required-to` and
`every`/`all` equivalence classes, explicit target-value inequality after scope identity
is known, and existing citation/freshness/conflict gates.

The deterministic layer contains no tunable threshold.

Open-world alias guessing, unrestricted predicate paraphrase inference, hierarchy
inference, conditional implication, nontrivial temporal arithmetic, open-ended modal
implication, and contradiction from lexical dissimilarity require a separately bound and
validated learned or specialized component.

At most three learned roles may exist in the future authoritative pipeline:

1. proposition extraction plus decontextualized frame extraction;
2. residual query-conditioned typed-edge alignment;
3. residual polarity on scope-aligned, sufficiently covered evidence.

The same underlying model may serve more than one role, but one future execution must
freeze exactly one authoritative pipeline before any M5 calibration inference occurs.
Model-family comparison and prompt search on the M5 calibration corpus are prohibited.

## Registered future calibration budget

A future binding may expose at most three scalar confidence thresholds:

- `extraction_confidence_min`;
- `alignment_confidence_min`;
- `polarity_confidence_min`.

Each registered threshold may take one of seven values from `0.60` through `0.90` in
increments of `0.05`, for at most `343` joint candidates. If a bound component exposes no
scalar confidence, its threshold may be disabled as one fixed state.

Class-specific thresholds, slot-specific thresholds, and post-result grid expansion are
prohibited. All raw component outputs must be frozen before threshold selection. Low
confidence in residual alignment maps to `UNSPECIFIED`; low confidence in polarity maps
to `UNRESOLVED` rather than being converted into contradiction.

The complete set contains 56 calibration-readiness requirements. No end-to-end metric can
compensate for a failed component requirement, and no floor may be weakened after results
are observed.

## Future independent validity floors

A4.5b-M5 registers stricter floors for a later independent validity stage while constructing
no validation data. Those floors include proposition precision/recall and target F1 at
`0.97`, slot/compatibility/sufficiency/polarity/final-relation core metrics at `0.95` or
higher, per-scope mismatch rejection at `0.98`, same-domain and cross-unit distractor
rejection at `0.99`, refutation recall at `0.97`, claim-category macro accuracy at `0.97`,
`SUPPORTED` precision at `0.99`, and exact registered safety gates.

These floors do not authorize validation.

## Data governance

The M2/M3 calibration corpus may not select TPAG models, prompts, or parameters. M5 may be
used for calibration readiness only after one authoritative TPAG pipeline is frozen under
a separately registered execution protocol. M5 itself is not independent generalization
evidence.

The A4.5a fresh-validation partition remains sealed and unscored. The original 68-query
confirmatory partition remains unopened and unscored. No new validation corpus is
constructed here. A4.5c remains ineligible and is not repurposed.

## Next research step

The registered next checkpoint is **A4.5b-M6: TPAG implementation binding and
calibration-only execution protocol**. It must freeze exactly one authoritative TPAG
pipeline, its deterministic components, any learned component revisions and weights, its
runtime, its raw-output contract, and the registered calibration selection procedure
before any M5 calibration result is exposed.

A4.5b-M6 is not authorized by this registration and requires separate approval.

# Phase 4 A4.5a: AERF measurement and fresh validity protocol

A4.5a registers the replacement grounding measurement after the failed A4.4d semantic verifier. It performs no learned inference and binds no model. Its purpose is to prevent the next implementation from being designed against the already-opened A4.4a/A4.4d validation evidence.

## Scientific starting point

A4.4d failed because the frozen three-way MNLI verifier treated absence or insufficiency of evidence as contradiction. A4.4e therefore selected Atom-Evidence Relation Factorization, or AERF, while keeping every learned component unbound.

AERF separates six operations:

1. align each atom to minimal evidence;
2. decide whether the evidence is relevant;
3. decide whether relevant evidence is sufficient to resolve the atom;
4. evaluate support versus refutation only when relevance and sufficiency are established;
5. map the component state deterministically to `ENTAILED`, `CONTRADICTED`, or `UNKNOWN`;
6. compose atomic relations into the claim verdict while preserving deterministic citation, freshness, and conflict gates.

The construction encodes the principle that missing evidence is not refuting evidence.

## Fresh candidate-independent validity surface

A4.5a creates a new fictional auxiliary support corpus, `helix-aerf-validity-corpus-v1`, before any replacement implementation is selected. It contains 60 independently named policy units and does not reuse an A4.4a or A4.4d row.

The deterministic seed is `20260821`. Forty units are calibration-only and twenty are fresh validation units. The split is unit-level so text variants from the same unit cannot cross between calibration and validation.

The component suite contains 540 atom-evidence pairs: 360 calibration and 180 validation. The fresh validation relation counts are 60 `ENTAILED`, 40 `CONTRADICTED`, and 80 `UNKNOWN`.

The claim-composition suite contains 540 cases: 360 calibration and 180 validation. It includes single supported, single refuted, single unknown, multi-document supported, partially supported multi-document, support-refute conflict, citation-invalid, stale-evidence, and registered-conflict cases.

The validity construction includes exact support, paraphrase support, explicit refutation, attribute refutation, cross-document irrelevance, same-domain irrelevance, relevant-but-insufficient evidence, temporal insufficiency, and context-contamination support. This is intentionally broader than the failure slice observed in A4.4d.

The registered validation hashes are:

- component pairs: `5f6f0294230de5da3af8baaee2403c9497bd42308c96f9d1041f4f88667d1da7`;
- claim cases: `116040d37035e4a43a3bee17ea2d29fe87d85c6148adade770e8c224456e43d6`.

## Component validity requirements

All requirements are conjunctive. A replacement implementation fails validity if any registered hard requirement fails.

Minimal evidence alignment requires precision and recall of at least 0.95. Relevance macro-F1, relevant recall, and irrelevant recall each require at least 0.95.

On relevant pairs only, sufficiency macro-F1, sufficient recall, and insufficient recall each require at least 0.95.

On relevant and sufficient pairs only, polarity macro-F1, support recall, and refutation recall each require at least 0.95.

The final three-way relation requires macro-F1 of at least 0.95 and recall of at least 0.95 for each of `ENTAILED`, `CONTRADICTED`, and `UNKNOWN`.

The failure-directed constraints are stricter. Cross-document irrelevance, same-domain irrelevance, and relevant-but-insufficient evidence may each produce false `CONTRADICTED` decisions at a rate no greater than 0.02. Context-contamination support accuracy must be at least 0.95.

## Claim-level validity and safety

Macro exact-verdict accuracy across the nine registered claim categories must be at least 0.95. `SUPPORTED` precision must be at least 0.98 and `SUPPORTED` recall at least 0.95.

Multi-document supported recall and partially supported multi-document `UNSUPPORTED` accuracy must each be at least 0.95.

Support-refute conflict, citation-invalid, stale-evidence, and registered-conflict categories each require accuracy 1.0. Across those safety-gated categories there may be zero false `SUPPORTED` verdicts.

## Evaluation order

The order is frozen before implementation work:

1. register AERF and this fresh validity construction;
2. bind exactly one authoritative AERF implementation before validation;
3. fit any thresholds or calibration parameters on the 40-unit calibration partition only;
4. freeze implementation identity, revision, weights, thresholds, runtime, and input hashes;
5. execute the 20-unit fresh validation partition exactly once;
6. accept pass or failure without post-validation rescue.

The replacement implementation may be modular, but the authoritative path must be singular before validation. Shopping among implementations after validation is not permitted.

## Independence rules

The opened A4.4a/A4.4d validation evidence is no longer independent. It may be used for descriptive diagnosis and regression testing only. It cannot serve as hard validity evidence for AERF, cannot be used to set thresholds, and cannot choose among models.

The new validation partition is also prohibited from model selection or threshold fitting. Calibration is the only fitting surface.

## Confirmatory boundary

The existing 68-query assistance confirmatory partition remains sealed. A4.5a authorizes zero confirmatory inspection and zero confirmatory scoring. Passing fresh AERF validity would still not itself authorize confirmatory execution; a separate approval is required.

## Literature rationale

The design follows established fact-verification distinctions while adapting them to the operational grounding failure observed in Helix. FEVER explicitly distinguishes `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO` and records evidence for supported/refuted claims. Work on fact-checking with insufficient evidence shows that evidence can be relevant yet insufficient without becoming contradictory. AttributionBench demonstrates that automatic claim-to-citation attribution remains difficult even for strong language models, and recent fact-verification work emphasizes evidence retrieval/alignment as a major source of error.

These sources motivate the factorization, but the Helix AERF thresholds and protocol are project-specific preregistered choices rather than claims inherited from those papers.

## Gate boundary

A4.5a is protocol registration only. It does not authorize model binding, calibration execution, validation execution, candidate scoring, or confirmatory access.

The next proposed checkpoint is A4.5b, AERF implementation binding and calibration-only execution protocol. It requires separate approval.

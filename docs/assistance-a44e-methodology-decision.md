# Phase 4 A4.4e: Post-validation methodology decision

A4.4e is a no-inference methodology checkpoint opened after A4.4d closed with the preregistered scientific status `FAILED_REGISTERED_VALIDATION_NO_RESCUE`.

A4.4e does not rerun A4.4d, search thresholds, refit temperature, bind a replacement model, score assistance candidates, or inspect the 68-query confirmatory partition. Its purpose is to identify what the A4.4d failure invalidates and to freeze the measurement architecture that may be registered next.

## Frozen predecessor

The predecessor state is permanent:

- scientific execution SHA `794562f6d9914bfc36e929c6c9df57e06969665a`
- GitHub Actions run `32508572173`, attempt 1
- immutable artifact `9456245570`
- artifact ZIP SHA256 `e6e31bd269d3da9b435ec6097d7df492dc4d151ee869b97152895861fad1956c`
- permanent closure SHA `291718add7a37681761ce8365d6db8dfbe504151`
- 11 of 17 registered requirements passed
- scientific status `FAILED_REGISTERED_VALIDATION_NO_RESCUE`

The failed result is evidence. It does not authorize a rescue run.

## Failure geometry

The headline failure was atomic `UNKNOWN` recall of `1 / 120 = 0.008333333333333333`. The error structure shows two distinct no-evidence regimes.

### Cross-document non-evidence

One hundred gold `UNKNOWN` atom-document pairs occur in the two multi-document categories because each atom is checked against all cited documents even though a cited document may be evidence only for another atom. These pairs deliberately test whether the verifier can recognize that a document is not evidence for the current proposition.

Among those 100 pairs, the frozen verifier predicts 94 `CONTRADICTED`, five `ENTAILED`, and one `UNKNOWN`. This error then propagates through deterministic composition. All 20 `multi_document_supported` cases become `CONFLICTING_EVIDENCE` rather than `SUPPORTED`, while 18 of 20 `partial_multi_document_unsupported` cases also become `CONFLICTING_EVIDENCE`.

### Same-document insufficient evidence

The remaining 20 gold `UNKNOWN` pairs are `unsupported_approval` claims checked against the correct same-intent policy document. The policy does not establish automatic approval, but it also does not contain the registered refuting proposition. All 20 are classified `CONTRADICTED`.

The failure is therefore not limited to cross-topic irrelevance. The native NLI contradiction class also absorbs cases in which evidence is topically relevant but insufficient to decide the claim.

### Whole-document context effect

The two `stale_current_evidence` atomic pairs are gold `ENTAILED` because the checked atom is the first sentence of the cited archived FAQ. Both are classified `CONTRADICTED` when the complete FAQ is supplied as the premise. The final stale-evidence verdict remains correct because freshness is a deterministic higher-priority gate, but the atomic error indicates that unrelated language elsewhere in a document can contaminate a whole-document relation judgment.

## What A4.4d invalidates

A4.4d rejects the specific proposition that a generic MNLI three-way classifier head can serve as the authoritative atomic grounding primitive through the direct mapping:

- NLI entailment to `ENTAILED`
- NLI contradiction to `CONTRADICTED`
- NLI neutral to epistemic `UNKNOWN`

That mapping is not valid for the Helix evidence-grounding construct.

A4.4d does not invalidate the higher-level A4.4a decomposition into deterministic citation identity, freshness, conflict handling, atomic support, and deterministic aggregation. Those components continued to behave as designed. It also does not establish that the underlying encoder representation is useless. The rejected object is the frozen native three-class decision surface as the authoritative claim-evidence relation instrument.

## Literature basis

The methodology decision is consistent with external work but is not justified by benchmark reputation alone.

FEVER separates `SUPPORTED`, `REFUTED`, and `NOT ENOUGH INFO`, preserving an epistemic distinction between lack of evidence and positive refutation. Nighojkar, Laverghetta Jr., and Licato show that the standard NLI neutral label has important construct-validity and operationalization problems. Mor-Lan and Levi further show that factual support and undermining relations are not interchangeable with ordinary textual NLI entailment and contradiction.

Task-specific factuality work also motivates a different architecture. MiniCheck directly studies grounded claim checking, while AlignScore and FENICE emphasize claim-to-source alignment rather than indiscriminate whole-document classification. Godbole and Jia show that factuality evaluators can disagree substantially and fail differently across domains, which reinforces the need for a new Helix-specific validity gate. Delbari and Pilehvar distinguish classifier-head generalization failure from the usefulness of underlying representations. TRACER provides additional motivation for separating evidence relevance/alignment from downstream verification operations.

The full citation matrix is frozen in `benchmarks/assistance/results/a44e_methodology_decision_v1/literature_matrix.md`. None of these papers establishes that a specific replacement model will satisfy the Helix requirements.

## Selected architecture: Atom-Evidence Relation Factorization

A4.4e selects **Atom-Evidence Relation Factorization (AERF)** as the architecture to register next. AERF is an internal engineering name, not a claim of methodological novelty.

AERF retains the deterministic A4.4a safety gates and atomization while replacing the generic three-way NLI primitive with a factorized claim-evidence relation process.

### Stage 1: atom-to-evidence alignment

For each atom and cited document, first identify the minimal sentence or bounded span that could be evidence for the atom. Whole-document classification is not authoritative because a document may contain the relevant proposition together with unrelated archival, safety, exception, or procedural language.

### Stage 2: relevance and sufficiency

Determine whether the aligned span addresses the proposition strongly enough to bear on its truth. If no relevant evidence exists, or if relevant material is insufficient to resolve the proposition, the relation is `UNKNOWN`.

`UNKNOWN` is therefore an explicit epistemic state. It is not defined as the native neutral class of a generic NLI head.

### Stage 3: support and refutation polarity

Only evidence that passes relevance and sufficiency is eligible for polarity judgment.

- positive supporting evidence maps to `ENTAILED`
- positive refuting evidence maps to `CONTRADICTED`
- no relevant evidence maps to `UNKNOWN`
- relevant but insufficient evidence maps to `UNKNOWN`
- support and refutation both present for an atom produce an unresolved semantic conflict

Absence of support is never itself evidence of contradiction.

### Stage 4: deterministic composition

The existing deterministic precedence is retained. Citation identity, freshness, and registered response-level conflict remain higher-priority gates.

At the atomic semantic layer, irrelevant cited documents do not create contradiction. An atom is supported when at least one eligible evidence span supports it and no eligible span refutes it. If no supporting evidence exists, the atom remains unsupported. If eligible support and eligible refutation coexist, the atom contributes `CONFLICTING_EVIDENCE`.

This preserves the safety behavior that survived A4.4d while removing the mechanism that converted irrelevant evidence into false conflicts.

## Why A4.4e does not select a model

A4.4e chooses a measurement architecture, not a learned implementation. No encoder, cross-encoder, fact-checking model, prompt, threshold, calibration rule, or span selector is bound here.

A direct binary support detector is also not sufficient by itself because Helix must distinguish positive refutation from mere non-support whenever conflict is safety-relevant. A future implementation may use separate components or a task-specific verifier, but the authoritative relation must obey the AERF semantics rather than inherit labels from a generic benchmark.

## Why A4.4a validation cannot be reused as independent validity

The 144 A4.4a validation cases are now opened. Their failure geometry directly influenced AERF. They may be retained as descriptive development evidence and regression evidence, but they cannot serve as independent hard-validity evidence for the replacement architecture.

This is a critical anti-leakage boundary. Any replacement methodology requires a fresh candidate-independent validity construction, frozen before the authoritative model and thresholds see its validation partition.

## Minimum requirements for a fresh validity protocol

The next protocol must test at least the following semantic regimes separately:

1. relevant and sufficient supporting evidence;
2. relevant and sufficient refuting evidence;
3. topically relevant but insufficient evidence;
4. cross-document irrelevant evidence inside a multi-document evidence pack;
5. support distributed across multiple documents;
6. explicit support and refutation coexistence;
7. local evidence embedded in a document containing unrelated negative or archival language;
8. paraphrastic support that is not a lexical copy;
9. unsupported claims whose truth is simply absent from the corpus.

Component-level metrics must expose alignment/relevance errors separately from polarity errors. Claim-level safety metrics must retain the deterministic citation, freshness, conflict, precision, and abstention requirements.

The future protocol must also define the model-binding and calibration order before validation opens. Any parameter selection must occur on a designated development or calibration partition, not on the new untouched validation partition.

## Confirmatory boundary

The 68-query confirmatory partition remains unopened. A4.4e authorizes zero confirmatory query records, zero confirmatory inference, and zero candidate scoring.

The confirmatory partition must not be opened merely because A4.4d failed. A replacement must first clear a new registered independent validity gate. Only a later separately approved checkpoint may determine whether confirmatory evaluation is warranted.

## Decision

A4.4e closes the methodology question as follows:

- reject another raw three-way MNLI argmax swap;
- reject any post-validation rescue of A4.4d;
- retain the deterministic safety gates and atomic composition;
- select AERF as the next measurement architecture;
- leave all learned components, thresholds, and candidate models unbound;
- require a fresh independent validity protocol before a replacement method can approach confirmatory evaluation.

## Gate boundary

A4.4e authorizes methodology selection only.

The next proposed checkpoint is **A4.5a: AERF measurement and fresh validity protocol registration**. A4.5a requires separate approval. It must be completed before any AERF model binding, calibration, validation execution, candidate comparison, confirmatory scoring, or production adoption.

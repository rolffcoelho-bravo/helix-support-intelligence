# Phase 4 A4.5b-M4 post-M3 measurement-architecture decision

## Decision

A4.5b-M4 selects **Typed Proposition Alignment Graph (TPAG)** as the next measurement architecture for the SCEC construct.

TPAG is an internal engineering name. This checkpoint makes **no novelty claim** and binds no parser, model, weights, prompt, threshold, calibration rule, or implementation.

The methodological decision is deliberately narrower than a new end-to-end verifier. SCEC's semantic principle survives: evidence must concern the same target and scope before sufficiency or polarity can be assessed. What is retired is M3's operational assumption that one zero-shot NLI primitive, queried through competing natural-language hypotheses, can authoritatively measure entity, predicate, target-slot, temporal, location, organizational, conditional, modality/quantification, coverage, and polarity relations at once.

## Why M3 is not repairable by threshold tuning

The closed M3 result contains 609 preregistered mismatch/coverage candidates and **zero feasible candidates**. The best-any diagnostic point, mismatch `0.56` and coverage `0.72`, passed only 14 of 42 readiness requirements.

The failure is not concentrated near one decision boundary. Entity, predicate, target-slot, conditional-scope, and modality/quantification mismatch rejection are all `0.0`; organizational mismatch rejection is `0.770833`; temporal mismatch rejection is `0.854167`; unresolved scope-gap insufficiency recall is `0.052083`; REFUTES recall, CONTRADICTED recall, and conflict detection are all `0.0`. Final relation macro F1 is `0.315255`, and false-SUPPORTED safety rate is `0.152778`.

This geometry rejects three tempting repairs inside M3:

1. moving the two global thresholds;
2. introducing per-dimension thresholds over the same semantic outputs;
3. swapping in another generic zero-shot NLI model while retaining the same universal-hypothesis design.

Those options would treat a construct-measurement failure as a calibration problem. The evidence instead indicates that multiple latent relations are being collapsed into one learned scoring mechanism.

## Literature audit

The literature supports moving from holistic relation scoring toward explicit proposition/evidence structure, while also warning against uncontrolled decomposition.

**Hu, Long, and Wang (NAACL 2025), “Decomposition Dilemmas: Does Claim Decomposition Boost or Burden Fact-Checking Performance?”** show that decomposition can improve or degrade verification because decomposition errors introduce downstream noise. DOI: `10.18653/v1/2025.naacl-long.320`. This supports preserving the already frozen AERF atoms rather than introducing adaptive free-form redecomposition after M3.

**Lu et al. (ACL 2025), “Optimizing Decomposition for Optimal Claim Verification”** show that decomposition and verifier behavior interact, and that verifier-preferred atomicity matters. DOI: `10.18653/v1/2025.acl-long.254`. A4.5b-M4 takes the conservative implication: do not let the failed verifier redefine the atoms post hoc; instead make the verifier consume explicit typed structure.

**Metropolitansky and Larson (ACL 2025), “Towards Effective Extraction and Evaluation of Factual Claims”** emphasize that claim extraction quality, coverage, ambiguity, and decontextualization directly affect fact checking. DOI: `10.18653/v1/2025.acl-long.348`. Their examples show that missing contextual qualifiers can transform non-contradictory evidence into apparent contradiction, directly matching the scope problem SCEC is intended to solve.

**Deng, Schlichtkrull, and Vlachos (ACL 2024), “Document-level Claim Extraction and Decontextualisation for Fact-Checking”** report that decontextualizing claims with needed context improves evidence retrieval. DOI: `10.18653/v1/2024.acl-long.645`. This supports explicit qualifier-bearing proposition frames.

**Wanner, Van Durme, and Dredze (EMNLP 2025), “DnDScore”** show that decomposition and decontextualization can work against each other and that verification strategy changes the resulting factuality score. DOI: `10.18653/v1/2025.emnlp-main.1205`. This argues for separating proposition extraction from later relation measurement rather than hiding both inside one NLI decision.

**Canby et al. (Findings ACL 2025), “Benchmarking Query-Conditioned Natural Language Inference”** show that conditioning semantic relation judgments on the relevant query is important for inconsistency detection and fact verification. DOI: `10.18653/v1/2025.findings-acl.765`. TPAG therefore requires any future learned residual relation model to be conditioned on the exact typed slot or proposition relation being adjudicated, not asked for an undifferentiated whole-pair verdict.

**Kim, Rahimi, and Allan (Findings EMNLP 2023), “Conditional Natural Language Inference”** explicitly model contradictory aspects together with the conditions under which they apply. DOI: `10.18653/v1/2023.findings-emnlp.456`. This supports representing conditional scope as a first-class field instead of expecting generic NLI to infer it implicitly.

**Holliday, Mandelkern, and Zhang (EMNLP 2024), “Conditional and Modal Reasoning in Large Language Models”** find basic errors across tested models on conditionals and epistemic modals. DOI: `10.18653/v1/2024.emnlp-main.222`. This directly cautions against treating conditional and modality/quantification scope as interchangeable natural-language labels inside one generic verifier.

**Wang and Zhao (Findings ACL 2024), “TRAM: Benchmarking Temporal Reasoning for Large Language Models”** find substantial gaps to human performance across temporal order, arithmetic, frequency, and duration. DOI: `10.18653/v1/2024.findings-acl.382`. TPAG therefore treats temporal scope as an explicit typed relation that may later use specialized normalization/reasoning rather than universal NLI.

**Li et al. (TrustNLP 2025), “Minimal Evidence Group Identification for Claim Verification”** formalize the need to identify evidence groups that collectively provide complete support and reduce the problem to Set Cover-like evidence aggregation. TPAG adopts the same high-level lesson without claiming their method: sufficiency is a graph/set coverage property, not a property of the single highest-scoring sentence.

**Si et al. (ACL 2024), “CHECKWHY: Causal Fact Verification via Argument Structure”** represent verification through connected evidence and explicit argument structures. DOI: `10.18653/v1/2024.acl-long.835`. This supports making evidence composition inspectable rather than implicit.

**PrimeFacts (LREC 2026), “From Articles to Premises: Building PrimeFacts”** reports that decontextualized evidence premises improve retrieval and verdict prediction, including reported 10–20 macro-F1 point gains over a baseline in their evaluation. Anthology ID: `2026.lrec-1.613`. The relevant lesson for Helix is that proposition-level, decontextualized evidence representations can materially improve downstream verification.

**Magomere et al. (Findings EACL 2026), “Distill and Align Decomposition for Enhanced Claim Verification”** further demonstrate that decomposition quality and verifier alignment are coupled. DOI: `10.18653/v1/2026.findings-eacl.309`. A4.5b-M4 does not adopt their RL approach, because verifier-adaptive decomposition after seeing Helix failures would add an unnecessary adaptive degree of freedom. It takes only the architectural lesson that decomposition and verification interfaces must be deliberately aligned.

## TPAG construct

For each frozen AERF atom, TPAG defines a typed claim proposition frame. Candidate evidence is segmented into evidence propositions and decontextualized only enough to recover information required to interpret those propositions.

The registered frame vocabulary is:

### Core slots

1. entity or subject;
2. predicate or event;
3. target-slot identity;
4. target value.

### Qualifier slots

5. temporal scope;
6. location scope;
7. organizational scope;
8. conditional scope;
9. modality or quantification.

Each claim-slot/evidence-slot comparison has exactly three semantic states:

- `MATCH`;
- `MISMATCH`;
- `UNSPECIFIED`.

The crucial difference from M3 is that these states are **typed relations**, not three candidate sentences competing for probability mass from a universal NLI call.

## Compatibility semantics

Compatibility remains fail-closed on explicit mandatory mismatch.

- If a claim specifies a mandatory slot and aligned evidence explicitly disagrees on that slot, the proposition is incompatible for that claim target.
- If evidence does not specify a claim-required slot, that is `UNSPECIFIED`, not contradiction. The proposition may remain compatible but cannot cover that slot.
- Exact normalized equality may later be handled deterministically where semantic risk is low.
- Learned semantic matching, if needed, is restricted to unresolved typed edges such as predicate paraphrase or entity aliasing. It may not silently decide unrelated slots.

This directly separates the two cases M3 repeatedly confused: **wrong scope** versus **same scope but missing decisive information**.

## Evidence coverage graph

TPAG represents evidence composition as a graph with claim-slot, evidence-proposition, and evidence-slot nodes. Alignment edges encode `MATCH`, `MISMATCH`, or `UNSPECIFIED`; a `COVERS` edge is licensed only by a compatible `MATCH` that contributes decisive claim content.

An evidence set is sufficient only if every decisive claim slot is covered by one or more mutually scope-compatible evidence propositions and no mandatory scope conflict remains unresolved.

This allows complementary multi-span evidence without pretending that the highest-scoring individual sentence must contain the whole answer. It also makes missing coverage auditable: a verdict can identify the exact uncovered slot rather than returning a generic neutral or contradiction score.

## Polarity and conflict

Polarity is evaluated **only after** the proposition target is scope-aligned and sufficiently covered.

Support means the sufficient evidence group establishes the aligned proposition. Refutation means it establishes an incompatible value or polarity for the **same aligned proposition target**, rather than merely being semantically dissimilar.

Explicit numerical/value incompatibility, explicit negation, and certain temporal relations may later be handled by specialized deterministic logic when safe. Any learned polarity component must be bound and validated separately from frame extraction and slot compatibility.

`CONFLICTING_EVIDENCE` requires at least one sufficient support group and one sufficient refutation group for the same aligned claim frame. This makes conflict detection a property of evidence groups, not a by-product of noisy pairwise labels.

## What remains unchanged

A4.5b-M4 does not reopen the underlying scientific semantics merely because M3 failed. The following remain fixed:

- existing AERF atoms are the unit of claim analysis;
- compatibility precedes sufficiency;
- compatible-but-insufficient evidence maps to `UNKNOWN`;
- incompatible evidence maps to `UNKNOWN`;
- compatible+sufficient support maps to `ENTAILED`;
- compatible+sufficient refutation maps to `CONTRADICTED`;
- coexisting sufficient support/refutation maps to `CONFLICTING_EVIDENCE`;
- citation, freshness, and registered-conflict vetoes remain deterministic safety gates.

## Measurement requirements for a future protocol

A future TPAG protocol must expose errors at the level where they occur. At minimum it must separately measure:

1. evidence proposition extraction fidelity;
2. claim/evidence frame slot extraction fidelity;
3. per-slot `MATCH` / `MISMATCH` / `UNSPECIFIED` accuracy;
4. explicit mismatch rejection for every registered scope dimension;
5. preservation of relevant-but-insufficient cases as compatible but incomplete;
6. set-level decisive-slot coverage;
7. complementary multi-span evidence-group recovery;
8. sufficiency classification;
9. polarity only on scope-aligned sufficient cases;
10. support/refute conflict detection;
11. final relation macro metrics;
12. claim-category and false-SUPPORTED safety metrics.

No component floor may be weakened merely to make TPAG pass.

## Data governance

The M2/M3 SCEC calibration corpus has now informed two architecture decisions: the failed M3 implementation and this post-result diagnosis. It is therefore not independent evidence for TPAG.

It may be used only for failure analysis and, after a future TPAG implementation is frozen, descriptive regression checks. A **new fresh TPAG calibration construction** is required before any future validation execution.

The A4.5a fresh-validation partition remains sealed and unscored. The original 68-query confirmatory partition remains unopened. A4.5c remains ineligible and is not repurposed.

## Scope of this checkpoint

Authorized and performed:

- analysis of the immutable M3 negative result;
- current literature audit;
- measurement-architecture selection;
- typed frame, alignment, coverage, polarity, and conflict semantics registration;
- future measurement and data-governance constraints.

Not authorized or performed:

- semantic inference;
- parser or model binding;
- model-family comparison;
- prompt search;
- threshold search;
- calibration fitting;
- M3 rescoring;
- fresh-validation scoring;
- confirmatory inspection or scoring.

## Next locked action

The only admissible next action is **fresh TPAG measurement and calibration protocol registration**. That checkpoint must define the fresh calibration construction, component labels, measurement floors, deterministic versus learned boundaries, and later binding/search budget **before** any parser/model is bound or any new inference is executed.

That next action is not authorized by this document and requires separate approval.

# Data Card

## Purpose

Helix Phase 1 freezes the data semantics required for later routing, retrieval, grounded generation, safety evaluation, and event logging. No Phase 1 artifact is evidence of model performance.

## BANKING77

**Role:** public benchmark for fine-grained banking intent classification.

**Upstream source:** PolyAI-LDN `task-specific-datasets`, pinned to Git commit `57ec275d8078af65b7731c2a98be812d844a6d6b`.

**Licence:** Creative Commons Attribution 4.0 International. The upstream project asks users of the banking dataset to cite Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*.

**Frozen source statistics:** 10,003 upstream training examples, 3,080 upstream test examples, 77 intents.

**Helix transformation:**

1. verify the pinned raw CSV SHA-256 digests;
2. normalize text only for leakage auditing, never to rewrite source examples;
3. quarantine 123 source-training rows identified by the frozen Phase 1 near-duplicate audit;
4. construct a deterministic, intent-stratified validation split from the remaining training pool;
5. leave all 3,080 official test examples untouched.

**Derived counts:** 7,904 train, 1,976 validation, 3,080 test, 123 quarantined.

**Test discipline:** the upstream test split is confirmatory. It cannot select models, hyperparameters, calibration methods, thresholds, prompts, or retrieval settings.

**Known limitations:** BANKING77 is English-only, intent-labelled, and not representative of live Helix traffic. Its queries do not establish real-world queue costs, out-of-scope prevalence, temporal drift, customer outcomes, or production safety.

## HelixBank Policy Corpus v1

**Role:** fictional corpus for retrieval, reranking, citation, refusal, conflict, and recommendation evaluation.

**Origin:** generated entirely from repository-owned deterministic templates. It intentionally does not reproduce a real bank's policy language.

**Version:** `helixbank-policy-v1.0.0`.

**Frozen contents:** 154 documents, 308 queries, 616 graded judgments, spanning all 77 BANKING77 intent labels.

**Evidence states:** current, archived, ambiguous, conflicting, missing, and controlled untrusted-content fixtures.

**Permitted use:** public research, testing, benchmarking, demonstrations, and engineering validation.

**Prohibited interpretation:** synthetic policy behavior is not evidence about a real bank, real customers, real agent acceptance, or real operational impact.

## Privacy and sensitive data

No dataset in Phase 1 contains account credentials, authentication codes, real customer records, or knowingly collected personal data. Deterministic fixtures use fictional UUIDs and support text.

## Versioning

Source revisions, checksums, derived split hashes, corpus hashes, and contract schemas are committed. Any later change that alters benchmark semantics requires a new version and must not silently overwrite Phase 1 evidence.

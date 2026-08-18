# Phase 1 Exit Report

- Phase: Data and evaluation contracts
- Status: Candidate — public CI pending
- Date: 2026-08-18
- Public version: 0.1.0

## Delivered

- BANKING77 source revision, licence, raw checksums, and deterministic materialization code.
- Frozen cross-split leakage quarantine and derived train/validation/test hashes.
- HelixBank Policy Corpus v1 with 154 documents, 308 golden queries, and 616 graded judgments.
- Public schemas for routing, retrieval, generation, safety, ticket events, policy documents, golden queries, and relevance judgments.
- Data card, evaluation contract, provenance/leakage report, public experiment-registry shell, and deterministic fixtures.
- Offline `data-check` integrated into the repository quality command.

## Exit evidence

| Gate | Verification | Result |
|---|---|---:|
| Source provenance frozen | Pinned revision and SHA-256 values | Pass |
| Official test preserved | 3,080 source test rows, no mutation | Pass |
| Leakage control | 123 source-training rows quarantined | Pass |
| Derived split determinism | Counts and canonical JSONL hashes | Pass |
| Intent coverage | 77 intents in train, validation, and test | Pass |
| Fictional corpus determinism | Generator, configuration, manifest hashes agree | Pass |
| Contract completeness | Eight JSON Schema contracts | Pass |
| Offline tests | `pytest` and Phase 1 validator | Pass |
| Public CI | GitHub Actions | Pending |

## Test-set discipline

No model fitting, threshold selection, prompt optimization, or final-test tuning is part of Phase 1. The official BANKING77 test split remains confirmatory.

## Publication boundary

The public experiment registry freezes the registry schema only. Unpublished confirmatory hypotheses, private thresholds, cost weights, and remediation logic remain outside the public repository.

## Decision

Phase 1 remains open until the branch passes public CI. No Phase 2 routing-model implementation is authorized before that gate closes.

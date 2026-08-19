# Phase 3 Exit Report

- Phase: Search and reranking
- Status: **Active — benchmark construction protocol frozen; hash freeze pending; no retrieval score produced yet**
- Date opened: 2026-08-19
- Public version: 0.1.0

## Phase lock

Phase 2 is merged and closed on `main`. Phase 3 is the only active implementation phase. Phase 4 evidence-bound assistance and recommendation have not started.

## Blueprint objective

Produce independently evaluated evidence retrieval through the bounded B0-B3 ladder:

- B0 BM25;
- B1 dense bi-encoder retrieval;
- B2 reciprocal-rank hybrid fusion;
- B3 hybrid fusion plus cross-encoder reranking.

## Required deliverables

- [ ] B0-B3 retrieval ladder completed.
- [x] Retrieval evaluation protocol defined.
- [x] Primary natural-language retrieval benchmark construction protocol frozen before scoring.
- [ ] Primary benchmark hash manifest frozen and audited.
- [ ] B0 BM25 baseline implemented and evaluated on development only.
- [ ] B1 dense bi-encoder registered before scoring and evaluated.
- [ ] B2 RRF hybrid configuration registered before scoring and evaluated.
- [ ] B3 cross-encoder reranker registered before scoring and evaluated.
- [ ] Vespa schema and reproducible index build.
- [ ] Retrieval benchmark and latency report.
- [ ] `/v1/search` integration tests.
- [ ] One final retrieval configuration frozen using the relevance-latency rule.
- [ ] Pre-confirmatory integrity audit.
- [ ] One-shot registered H1/H2 confirmatory result.

## Initial benchmark audit finding

The Phase 1 HelixBank corpus contains 308 deterministic golden queries whose wording is generated directly from the same intent names used in document titles and bodies. That set remains useful for contract and edge-case behavior, but using it as the primary Phase 3 relevance benchmark would create unusually favorable lexical overlap and could overstate BM25 and hybrid-search quality.

This is not treated as a Phase 1 defect requiring the closed phase to be reopened. Phase 3 already owns relevance evaluation under the blueprint. Therefore Phase 3 introduces a primary natural-language retrieval benchmark derived from the pinned BANKING77 **source training file only**.

The construction replays the frozen Phase 1 train quarantine/split contract and then samples only `fit_train` utterances with a new deterministic Phase 3 salt:

- 20 development queries per intent = 1,540;
- 10 confirmatory queries per intent = 770;
- 77 intents represented in both partitions;
- development and confirmatory stable IDs are disjoint;
- official BANKING77 test access is forbidden in the materializer.

The query intent deterministically maps to the corresponding frozen HelixBank governing policy (relevance 3) and current FAQ where eligible (relevance 2).

## Candidate eligibility

Documents are filtered before scoring:

- `corpus_version == helixbank-policy-v1.0.0`;
- `status == current`;
- `permission == public_support`.

The current deterministic corpus yields 147 eligible candidate documents because seven archived FAQ fixtures are excluded.

## Current evidence state

No B0-B3 retrieval metric has been produced in Phase 3 yet. This is intentional: the benchmark hash manifest must be materialized, audited, and frozen before the first retrieval score.

## Next locked action

Run the dependency-light benchmark materializer in CI, inspect only the generated counts and hashes, freeze those values into the Phase 3 benchmark contract, and rerun the release quality gate. Only after that freeze may B0 BM25 development scoring begin.

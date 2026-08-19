# Phase 3 Retrieval Evaluation Protocol

Status: **frozen before evaluation**  
Protocol: `phase3-retrieval-r3.0-v1`  
Checkpoint: `R3.0`  
Freeze date: `2026-08-19`

## Purpose

This protocol fixes the Phase 3 retrieval experiment before any B0, B1, B2, or B3 score is observed. The objective is to compare a bounded retrieval ladder over the already-frozen HelixBank Policy Corpus v1 and to adopt additional retrieval complexity only when it produces reproducible relevance gains within a declared latency budget.

This phase evaluates retrieval only. It does not reopen Phase 2 routing, modify the frozen A2 router, evaluate generated answers, or claim production banking effectiveness.

The machine-readable source of truth is [`configs/models/retrieval_ladder_v1.json`](../configs/models/retrieval_ladder_v1.json).

## Frozen evidence substrate

The retrieval experiment uses the existing fictional HelixBank Policy Corpus. No new corpus is created for Phase 3.

| Quantity | Frozen value |
|---|---:|
| Corpus version | `helixbank-policy-v1.0.0` |
| Generator version | `1.0.0` |
| Documents | 154 |
| Queries | 308 |
| Relevance judgments | 616 |
| Intents | 77 |
| Document SHA-256 | `13572a02eddadbd621a39490238358acccf8c5f01fc7a26dcfe3f4017aad2d8f` |
| Query SHA-256 | `2c54b5353d71a399cb15303fb6d751dd18c9ba814c992c4f676fb19bd865481c` |
| Judgment SHA-256 | `8d5b4b816a807d5976d0102dcdd3d917a16c9c4a916e4776f5023b47043d5a4d` |
| Evaluation date | `2026-08-19` |

The material is repository-generated fictional policy content. Results from this corpus are controlled benchmark evidence, not measurements of a real bank knowledge base or real customer traffic.

## Eligibility before ranking

Eligibility is applied before any candidate assigns a rank. A document is eligible only when all of the following are true:

- `status == current`;
- `permission == public_support`;
- `audience == customer_support`;
- `jurisdiction == fictional-global`;
- `valid_from <= 2026-08-19`;
- `valid_to` is null or `valid_to >= 2026-08-19`.

Archived fixtures are therefore excluded from the active evidence pool. Current conflict fixtures remain eligible because suppressing them would hide the evidence condition that later decision logic must detect. Untrusted-content fixtures remain ordinary data and must never be interpreted as instructions.

The benchmark explicitly forbids candidate filtering with query labels or gold information. Retrieval may not use `intent`, `queue`, `case_type`, `expected_decision`, `gold_citations`, `allowed_resolution_types`, or relevance judgments to reduce the candidate pool or alter scores.

This prevents label-derived shortcuts from being reported as retrieval quality.

## Text representation

For every eligible document, the indexed text is:

```text
title + "\n" + body
```

The query text is exactly the `text` field from the frozen retrieval query record. Intent labels and other benchmark metadata are not appended to either query or document text.

## Bounded retrieval ladder

The ladder is intentionally small. No candidate may be added after scoring begins under this protocol version.

| ID | Candidate | Frozen definition |
|---|---|---|
| B0 | BM25 | Deterministic repository-owned BM25, `k1=1.2`, `b=0.75`, top 50 |
| B1 | Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2`, pinned revision, cosine over normalized embeddings, top 50 |
| B2 | Hybrid RRF | Reciprocal-rank fusion of B0 and B1 top-50 lists, `k=60`, top 50 |
| B3 | Hybrid + reranker | Cross-encoder reranking of B2 ranks 1 to 20, then append B2 ranks 21 to 50 unchanged |

### B0 tokenization

BM25 uses Unicode NFKC normalization followed by lowercase conversion. Tokens match `[A-Za-z0-9]+`. There is no stemming and no stopword removal.

### B1 dense model

- model: `sentence-transformers/all-MiniLM-L6-v2`
- revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`
- maximum sequence length: 256
- pooling: model-native mean pooling
- embedding normalization: enabled
- similarity: cosine

### B2 fusion

Each source ranking is resolved deterministically before fusion. For a document present in one or both source lists:

```text
RRF(document) = sum(1 / (60 + rank_i))
```

The fusion source depth is 50 for both B0 and B1. The fused list is truncated to 50.

### B3 reranker

- model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- revision: `c5f2b386de279a97c53a702dd5189d1c407160dc`
- maximum sequence length: 512
- pair: query text against `title + "\n" + body`
- rerank depth: 20

Only B2 ranks 1 to 20 are rescored. B2 ranks 21 to 50 are appended in their original order so Recall@50 remains measurable on the same candidate support.

## Deterministic ranking

Candidate scores are sorted in descending order. Exact score ties are resolved by `document_id` ascending. Source-ranking ties are resolved before RRF using the same rule. No ranking-stage randomness is permitted.

## Relevance semantics

The frozen judgments use relevance grades 0 through 3. Unjudged eligible documents receive grade 0 for metric computation.

An ineligible document is removed from the qrels for the evaluated evidence pool before computing the ideal ranking. This means an archived FAQ cannot improve or reduce the score of a system that correctly excludes archived evidence.

For nDCG, gain is `2^relevance - 1`.

For MRR and recall diagnostics, direct relevance is defined as grade at least 2.

## Registered metrics

The primary retrieval endpoint is **nDCG@10**.

Registered secondary metrics are:

- **MRR@10**;
- **Recall@20**;
- **Recall@50**.

For MRR@10, a query with no eligible document of relevance at least 2 contributes zero. For Recall@20 and Recall@50, the macro average is computed only across queries with at least one eligible relevance-at-least-2 document, and the applicable-query count must be reported.

Diagnostic slices may be reported by `case_type`, document kind, conflict-fixture status, and untrusted-content-fixture status. Slice metrics cannot replace the primary endpoint after results are observed.

## Registered hypotheses

### H1, hybrid retrieval relevance

Comparison: `B2 - B0`  
Endpoint: `nDCG@10`

H1 asks whether lexical-semantic fusion improves graded top-10 ranking relevance over the lexical baseline.

### H2, reranking first-evidence quality

Comparison: `B3 - B2`  
Endpoint: `MRR@10`

H2 asks whether cross-encoder reranking improves the rank of the first directly relevant evidence item over hybrid retrieval.

These comparisons are fixed before scoring. B1 remains a necessary bounded component comparison and diagnostic candidate, but it does not replace H1 or H2.

## Statistical inference

Inference is paired at the query level.

- resampling method: nonparametric paired bootstrap;
- replicates: 5,000;
- seed: `20260819`;
- interval: two-sided percentile 95%;
- difference orientation: candidate minus comparator.

A registered hypothesis is supported only when the point estimate is positive and the lower confidence bound is above zero. If the interval includes zero, the result is inconclusive. If the point estimate is negative and the upper confidence bound is below zero, the result is adverse.

Negative and inconclusive results remain part of the permanent Phase 3 evidence.

## Latency protocol

Latency is measured on CPU with the corpus index and document embeddings already constructed. The measurement represents warm per-query retrieval, not installation or startup time.

Included in timing as applicable:

- query tokenization;
- query embedding;
- lexical scoring;
- dense similarity;
- RRF fusion;
- cross-encoder scoring.

Excluded from timing:

- model download;
- model loading;
- corpus generation;
- BM25 index construction;
- document embedding construction.

Each candidate receives 30 untimed warm-up queries and five timed passes over the 308 frozen queries in canonical `query_id` order. Results must report mean, p50, p95, p99, and queries per second together with the CPU and software environment.

The frozen p95 latency budgets are:

| Candidate | Maximum p95 warm latency |
|---|---:|
| B0 | 100 ms |
| B1 | 250 ms |
| B2 | 250 ms |
| B3 | 500 ms |

These are experiment-selection budgets for this repository benchmark. They are not service-level commitments for a production bank.

## Complexity-adoption rule

The default winner is B0. B1, B2, and B3 are considered in that order.

A more complex candidate replaces the current simpler winner only when all of the following hold:

1. its mean nDCG@10 improvement is at least `0.010`;
2. the paired 95% bootstrap lower bound for that nDCG@10 difference is above zero;
3. its MRR@10 difference is not below `-0.005`;
4. its p95 latency is within its declared candidate budget.

If any condition fails, the current simpler winner is retained and the negative or inconclusive comparison is preserved.

If multiple candidates satisfy replacement rules and their nDCG@10 values are within `0.005`, the lower-p95 candidate wins. If still tied, the simpler lower-numbered candidate wins.

The selection rule is intentionally stricter than simply choosing the largest observed score.

## Execution guard

At freeze time:

```text
results_opened = false
```

No B0 through B3 score may be treated as an R3.0 result before this protocol is merged. Before the first score, the following are frozen:

- corpus records and qrels;
- candidate identities and model revisions;
- BM25 parameters and tokenization;
- retrieval depths;
- RRF parameter;
- rerank depth;
- eligibility filters;
- metric definitions;
- bootstrap settings;
- latency rules;
- complexity-adoption thresholds.

After scoring, a result-motivated change must create a new protocol version. It cannot overwrite the R3.0 v1 evidence.

## Phase boundary

Phase 2 remains closed. Phase 3 retrieval evidence is evaluated separately from the already-verified routing subsystem.

The next valid checkpoint after this protocol is merged is implementation of the frozen evaluator and B0 through B3 candidates, followed by a protocol-compliance preflight. Scoring begins only after that implementation reproduces the frozen corpus hashes and passes the guard checks.

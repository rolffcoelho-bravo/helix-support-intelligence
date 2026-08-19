# Phase 3 Retrieval Implementation Preflight

Status: **implementation complete, benchmark scoring closed**  
Implementation: `phase3-retrieval-r3.1-v1`  
Protocol: `phase3-retrieval-r3.0-v1`

## Purpose

R3.1 converts the frozen retrieval protocol into executable, typed retrieval components without opening the 308-query retrieval benchmark. The checkpoint tests implementation semantics on small unit fixtures, verifies the immutable corpus manifest and eligibility pool, and confirms that the model identities and revisions remain exactly those declared before scoring.

No Phase 3 retrieval quality or latency result is produced at this checkpoint.

## Implemented surface

| Component | R3.1 implementation |
|---|---|
| B0 | Deterministic repository-owned BM25 with frozen tokenization, `k1=1.2`, `b=0.75`, and deterministic tie-breaking |
| B1 | Dense cosine retriever behind a typed embedding-encoder adapter |
| B2 | Reciprocal-rank fusion with source depth 50, `k=60`, and top 50 output |
| B3 | Cross-encoder head reranking behind a typed pair-scoring adapter, with the B2 tail preserved |
| Metrics | nDCG@10, MRR@10, Recall@20, Recall@50 |
| Inference | Paired bootstrap difference with deterministic seed and percentile interval implementation |
| Latency | Deterministic summary functions for mean, P50, P95, P99, and throughput |
| Selection | Registered complexity-adoption rule implemented as an explicit decision function |

The composed `RetrievalLadder` executes B0 through B3 using injected model adapters. R3.1 unit tests use only tiny deterministic fixtures. They do not pass a frozen HelixBank retrieval query to the ladder.

## Implementation details fixed before scoring

The machine-readable implementation record is [`configs/models/retrieval_implementation_v1.json`](../configs/models/retrieval_implementation_v1.json).

The B0 IDF calculation is fixed as:

```text
ln(1 + (N - df + 0.5) / (df + 0.5))
```

Dense document and query vectors are L2-normalized inside the retrieval core. Their dot product is therefore cosine similarity. All score ties are resolved by ascending `document_id`.

For B3, only the first 20 B2 documents enter the pair scorer. Those 20 are sorted by descending reranker score with `document_id` as the tie-break. B2 ranks 21 through 50 are appended in their existing order.

Bootstrap and latency percentiles use linear interpolation. These details are fixed here before frozen-query retrieval is allowed.

## Preflight checks

`scripts/preflight_phase3_retrieval.py` performs only readiness checks. It verifies:

- the R3.0 protocol identifier and pre-evaluation state;
- the frozen HelixBank version, counts, and SHA-256 values against the deterministic generator;
- the pre-ranking eligibility pool;
- retention of current conflict and untrusted-content fixtures;
- exact B0 through B3 ladder identity;
- exact dense and reranker model commit pins;
- zero benchmark-query scores computed by the preflight.

The expected eligible evidence pool is 147 documents. Seven current conflict fixtures and five current untrusted-content fixtures remain eligible. Archived documents remain excluded.

## Model execution boundary

The core Python package deliberately does not add a heavyweight model runtime dependency. B1 and B3 use typed adapter interfaces. The later benchmark execution environment must bind those interfaces to the exact revisions already registered in the R3.0 protocol:

- `sentence-transformers/all-MiniLM-L6-v2` at `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`;
- `cross-encoder/ms-marco-MiniLM-L6-v2` at `c5f2b386de279a97c53a702dd5189d1c407160dc`.

A different model or revision cannot replace either pin because of an observed retrieval result. Such a change would require a separate versioned evaluation.

## Boundary of this checkpoint

R3.1 establishes implementation readiness only. It does not report nDCG, MRR, recall, latency, bootstrap comparisons, or a selected retrieval winner. Those fields remain genuinely unknown until the registered retrieval execution is performed.

# Phase 3 B3 Cross-Encoder Development Protocol

Status: **frozen before the first B3 development score**

## Purpose

B3 asks one bounded question: does a fixed cross-encoder reranker add enough top-rank relevance
to justify its extra inference stage over the already-strong B1 dense retriever?

This checkpoint does not authorize a reranker model search, candidate-depth search, fine-tuning,
hybrid rescue, or access to the sealed retrieval confirmatory partition.

## Frozen parent

B3 uses B1 as its only parent retriever.

- B1 configuration: `retrieval-b1-dense-v1`
- accepted B1 complete-ranking SHA-256:
  `b51d2649453e2ddedfde7d76525f11d41f37fe364077d840c048378d4a33fe20`
- candidate pool: the first **50** B1 documents per query
- ranks 51 through 147 remain in the original B1 order after reranking

The top-50 depth is fixed before B3 scoring. B1 development Recall@50 is 0.9881, so this pool
captures nearly all judged relevance while still making the added cross-encoder cost explicit.
Reranking top 50 also permits relevant evidence originally below rank 20 to move into the
top-20 evaluation window.

## Frozen reranker

- family: cross-encoder sequence-classification reranker
- model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- immutable revision: `fbf9045f293a58fa68636213c5e0cb8a2de5d45e`
- licence: Apache-2.0
- `model.safetensors` SHA-256:
  `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`
- maximum pair length: 512 tokens
- score: raw sequence-classification logit
- activation: identity
- batch size: 32
- device: CPU
- threads: 1
- fine-tuning: none
- trust remote code: false

Each pair is:

`raw BANKING77 query text` + `HelixBank title + newline + body`

The BGE query instruction used by B1 is not injected into B3 because B3 is a separate
MS-MARCO-trained pairwise reranker.

## Ranking rule

For each query:

1. reconstruct the frozen B1 full ranking;
2. fail closed if its complete ranking SHA differs from the accepted B1 hash;
3. score exactly the first 50 B1 documents with the cross-encoder;
4. sort descending by raw cross-encoder score;
5. use original B1 candidate rank, then document ID, as deterministic tie breakers;
6. append B1 ranks 51 through 147 unchanged.

No post-score calibration, score blending, RRF, learned fusion, or query-specific rule is allowed.

## Evaluation and stopping rule

The metric contract remains the frozen Phase 3 retrieval contract:

- nDCG@10
- MRR@10
- Recall@20
- Recall@50
- Success@1
- governing-policy recall@20

B3 replaces B1 only if **all** registered conditions hold:

- nDCG@10 improves by at least **+0.020 absolute**;
- MRR@10 does not decrease;
- Success@1 does not decrease;
- Recall@20 does not decrease;
- governing-policy recall@20 does not decrease;
- Recall@50 remains exactly equal to B1, as required by the fixed top-50 candidate pool.

The +0.020 nDCG@10 materiality floor is frozen before B3 scoring as the minimum relevance gain
required to justify adding a second neural scoring stage. If B3 fails any condition, it is
rejected as the selected retrieval reference. The result is preserved; there is no rescue tuning.

## Latency accounting

The evaluator reports separately:

- B1 parent reconstruction time;
- cross-encoder model-load time;
- total reranking time;
- scored pair count;
- reranking pairs per second;
- mean reranking time per development query.

The registered pair count is **69,300** = 1,386 queries × 50 candidates.

GitHub-hosted CPU timing is descriptive development evidence only. It is not a production SLA
and is not generalized to deployment hardware.

## Leakage and confirmatory boundary

The B3 evaluator may read only:

- frozen candidate documents;
- development queries;
- development qrels;
- already-public B1 development evidence.

It must refuse materialized confirmatory query or qrel files. The 616-query retrieval
confirmatory partition remains sealed, and the official BANKING77 test split must not be accessed.

## Scientific decision boundary

Only this one B3 candidate is authorized. Once the first valid B3 development result exists:

- preserve positive, negative, or inconclusive evidence;
- do not change model family, revision, depth, input format, batch semantics, activation,
  tie breaking, metric definitions, or the selection rule to improve the result;
- perform the standard hostile audit before any next Phase 3 action.

# Phase 3 R3.2 Validated Retrieval Result

Status: **validated after independent and manual post-execution audit**  
Execution: `phase3-retrieval-r3.2-v1`  
Protocol: `phase3-retrieval-r3.0-v1`  
Implementation: `phase3-retrieval-r3.1-v1`

## Provenance

The registered execution ran once from commit `d13bc04383c8cc59b2cc418fe48c9ef0756b2a6a` in GitHub Actions run `32302009548`, job `96226418119`. The uploaded evidence artifact is `9384078230`; its ZIP SHA-256 is `7e0b8106b4e44ffcf6046996a06cef6360013f0d2ac424830bf6a2784d88c6c3`.

The workflow verified all registered scientific input hashes before the first ranking and verified them again after execution. The artifact-level post-execution reconstruction and the separate diagnostic reconstruction both passed with no recorded failure. A subsequent manual code/result audit found no defect requiring invalidation or rerun.

The benchmark contains 308 frozen queries over 147 eligible documents from the fictional HelixBank Policy Corpus v1.

## Registered retrieval evidence

| Candidate | nDCG@10 | MRR@10 | Recall@20* | Recall@50* | P95 latency |
|---|---:|---:|---:|---:|---:|
| **B0 BM25** | **0.9009** | **0.7727** | **1.0000** | **1.0000** | **0.49 ms** |
| B1 dense | 0.8867 | 0.7668 | 1.0000 | 1.0000 | 17.66 ms |
| B2 hybrid RRF | 0.8878 | 0.7711 | 1.0000 | 1.0000 | 18.44 ms |
| B3 hybrid + reranker | **0.9325** | 0.7727 | 1.0000 | 1.0000 | **838.78 ms** |

\* Recall@20 and Recall@50 are macro-averaged over the 238 queries with at least one eligible relevance >= 2 item. The remaining 70 missing-evidence queries are not recall-applicable under the frozen protocol.

## Registered hypotheses

**H1, B2 minus B0 on nDCG@10: ADVERSE.** The mean difference is `-0.01309`, with registered paired-bootstrap 95% CI `[-0.01914, -0.00737]`. Hybrid RRF therefore does not improve the registered primary endpoint over BM25 on this benchmark; it significantly lowers it under the registered query-level inference.

**H2, B3 minus B2 on MRR@10: INCONCLUSIVE.** The mean difference is `+0.00162`, with registered 95% CI `[0.00000, 0.00487]`. The lower bound is exactly zero, so the registered support rule is not met. The result cannot be described as evidence that cross-encoder reranking improves MRR@10.

## Complexity-selection decision

The frozen complexity rule selects **B0**.

B1 and B2 fail because they do not achieve the required nDCG@10 improvement over the current winner. B3 has the strongest graded-ranking score, with nDCG@10 `0.9325`, and its B3-minus-B0 nDCG@10 difference is `+0.03156` with a positive 95% interval. However, its P95 latency is `838.78 ms`, which exceeds the registered `500 ms` budget by `338.78 ms`, or approximately `67.76%`.

No complex candidate earned adoption. The selected Phase 3 retrieval configuration is therefore the deterministic B0 BM25 baseline.

## Manual code and result audit

The manual audit checked additional invariants beyond the automated verifier:

- exactly 1,232 query-candidate ranking records, representing 308 queries times four candidates;
- exactly 61,600 ranked rows, with 50 unique eligible documents per ranking;
- exact rank continuity from 1 through 50 and finite stored scores;
- exact B2 reciprocal-rank-fusion reconstruction from B0 and B1;
- B3 top-20 membership identical to B2 top-20 membership, with only reordering inside the head;
- B3 ranks 21 through 50 identical to B2 ranks 21 through 50;
- no deterministic tie-breaking violation;
- 1,540 raw latency measurements per candidate, corresponding to five timed passes of 308 queries;
- B3 P95 latency above the 500 ms budget in every timed pass, with pass-level P95 ranging from approximately `837.59 ms` to `840.20 ms`;
- all internal artifact checksums and the downloaded artifact ZIP digest independently verified.

The H1 difference is zero on 271 queries, adverse on 29, and favorable on 8. H2 changes MRR@10 on only one query: B3 improves that query from reciprocal rank 0.5 to 1.0 while the other 307 queries are unchanged. This concentration is consistent with H2's zero lower confidence bound and the benchmark's strong MRR ceiling.

As a post-hoc audit sensitivity only, not a replacement for the registered inference, the bootstrap was repeated by resampling 77 intent clusters rather than 308 query variants. H1 remained adverse with a 95% interval approximately `[-0.02239, -0.00450]`; H2 remained inconclusive with `[0.00000, 0.00487]`. The official inference remains the predeclared query-level bootstrap.

## Interpretation boundaries

This benchmark is controlled fictional evidence, not a measurement of a real bank knowledge base, real customer traffic, or production service quality.

Recall@20 and Recall@50 are saturated at 1.0 for all four candidates on the 238 recall-applicable queries. They therefore provide little candidate discrimination in R3.2. B0 already places a directly relevant item at rank 1 for every one of those 238 queries; B1 misses rank 1 on three such queries, B2 on one, and B3 restores rank 1 on all 238. This creates a strong ceiling for H2.

The 70 `missing_evidence` queries require separate interpretation. They have no eligible relevance >= 2 item, so they contribute MRR@10 equal to zero and are excluded from Recall@20/50. They can nevertheless receive high nDCG@10 because the frozen graded-gain definition assigns positive gain to relevance-1 partial evidence. High nDCG on this slice therefore means good ordering of weak or partial evidence, not that the request is answerable or sufficiently supported.

Latency was measured on a single-thread CPU configuration on one GitHub-hosted runner. It supports the registered relative complexity decision for this execution environment, not a universal production latency guarantee.

## Evidence files

The complete immutable execution evidence is retained in `raw/`, including top-50 rankings, per-query metrics, raw latency samples, environment metadata, input hashes, checksum manifest, diagnostic slices, and both automated post-execution audits. GitHub Actions run and artifact metadata are retained under `provenance/`.

Negative and inconclusive results are preserved without rescue tuning. R3.2 does not reopen the frozen ladder or authorize a new candidate search.

# Phase 3 B3 Cross-Encoder Development Checkpoint

> **Verdict: rejected as the selected retrieval reference.** B1 remains the leading Phase 3 retriever. The sealed retrieval confirmatory partition remains unopened.

## Registered result

| Metric | B1 dense | B3 cross-encoder | B3 - B1 |
|---|---:|---:|---:|
| nDCG@10 | **0.6537** | 0.6068 | **-0.0468** |
| MRR@10 | **0.7005** | 0.6463 | **-0.0542** |
| Recall@20 | **0.9325** | 0.9149 | **-0.0177** |
| Recall@50 | **0.9881** | **0.9881** | +0.0000 |
| Success@1 | **0.6248** | 0.5678 | **-0.0570** |
| Governing-policy recall@20 | **0.9228** | 0.9098 | **-0.0130** |

B3 fails five of the six frozen selection checks. The only pass is Recall@50 equality, which is expected because B3 reranks only the first 50 B1 documents and preserves the remaining tail.

## Frozen method

B3 was registered before its first valid development result as one fixed two-stage retrieval candidate:

1. reconstruct the frozen B1 dense ranking;
2. verify the accepted B1 complete-ranking hash;
3. take exactly the first 50 B1 candidates;
4. score each query-document pair with `cross-encoder/ms-marco-MiniLM-L6-v2`;
5. sort those 50 by raw cross-encoder logit, with original B1 rank and then document ID as tie breakers;
6. append B1 ranks 51–147 unchanged.

The reranker is pinned to immutable revision `fbf9045f293a58fa68636213c5e0cb8a2de5d45e`. Its `model.safetensors` SHA-256 is `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.

The candidate depth, model revision, input format, activation, batch size, CPU-only execution, metric definitions, and stopping rule were all frozen before the first valid score. No reranker-family search, depth search, score blending, fine-tuning, or post-result rescue is permitted.

## Registered stopping rule

B3 could replace B1 only if all of the following held:

- nDCG@10 improvement at least +0.020 absolute;
- MRR@10 non-decreasing;
- Success@1 non-decreasing;
- Recall@20 non-decreasing;
- governing-policy recall@20 non-decreasing;
- Recall@50 exactly equal to B1.

Observed checks:

| Check | Result |
|---|---|
| Material nDCG@10 gain | **FAIL** |
| MRR@10 non-decrease | **FAIL** |
| Success@1 non-decrease | **FAIL** |
| Recall@20 non-decrease | **FAIL** |
| Governing-policy recall@20 non-decrease | **FAIL** |
| Recall@50 equal to B1 | **PASS** |

The registered verdict is therefore **rejected**.

## Breadth of the regression

The loss is not driven by one isolated intent group.

| Metric | B3 better | Tied | B3 worse |
|---|---:|---:|---:|
| nDCG@10 | 30 | 0 | **47** |
| MRR@10 | 27 | 1 | **49** |
| Recall@20 | 12 | 34 | **31** |
| Recall@50 | 0 | **77** | 0 |
| Success@1 | 20 | 13 | **44** |
| Governing-policy recall@20 | 12 | 41 | **24** |

This is a meaningfully different negative result from B2. B2 showed that naïve equal-weight lexical+dense rank fusion diluted a much stronger dense parent. B3 shows that a standard off-the-shelf MS-MARCO cross-encoder can also degrade this Helix domain-specific ranking even when it is applied only as a second-stage reranker over a high-recall B1 candidate pool.

The result should not be generalized into a claim that cross-encoder reranking is ineffective. It is specific to this frozen model, this candidate depth, this input representation, and this benchmark.

## Latency

The cross-encoder scored **69,300 query-document pairs** on the frozen 1,386-query development benchmark.

- model load: 0.651 s;
- reranking total: 2,336.006 s;
- throughput: 29.666 pairs/s;
- mean incremental reranking time: 1.685 s/query.

These timings are descriptive GitHub-hosted CPU measurements only. They are not production latency claims. Even so, the development comparison is unfavorable in both directions: B3 adds a substantial second-stage compute cost while reducing the registered relevance metrics.

## Integrity and execution-state audit

The evaluator verified:

- accepted B1 ranking SHA-256: `b51d2649453e2ddedfde7d76525f11d41f37fe364077d840c048378d4a33fe20`;
- observed B1 ranking SHA-256: the same accepted hash;
- B1 model weight SHA-256: `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`;
- B3 model weight SHA-256: `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`;
- B3 complete-ranking SHA-256: `6986cce96e2fbac1be3fd33a4e24f1049d1faa5411d9fa8bb98486f383f156a5`;
- confirmatory partition opened: false;
- official BANKING77 test accessed: false.

The associated GitHub Actions job is marked `cancelled`, but the evaluator itself completed and finalized all three scientific output files before cancellation propagated. The artifact exists with SHA-256 `7e52f12e0fd1b5d78be7027f1b96c00dcb019fd1bfec69bc227af1e26387f848`. The raw result, report, and run-output checksums were independently verified after download. This workflow-state caveat is preserved rather than hidden.

Evidence:

- workflow run: `32269695969`;
- artifact: `9373122220`;
- evaluated merge SHA: `249d5767689c38c86632eb40a46b5aaaa320c208`;
- artifact ZIP SHA-256: `7e52f12e0fd1b5d78be7027f1b96c00dcb019fd1bfec69bc227af1e26387f848`;
- raw `results.json` SHA-256: `f9b0851c3abcd3b7a61735b3d6eda45073b85e5c66542ba7d3fb9f62cc1d1548`;
- raw `report.md` SHA-256: `aefbbd45a05da7c172f0b482cbe1f27ba3d0dc07af471982f69d0e2ceb6324c3`;
- raw `run-output.txt` SHA-256: `d6c7de91719f3a6f41f03d64293eaeecd7dc288e54d4476186ce00a84ab73bfc`.

## Interpretation

B1 remains difficult to improve with generic retrieval components. B1 already captures most judged relevant evidence by rank 50, and its semantic ordering is strong enough that the frozen MS-MARCO reranker moves too many useful Helix documents downward rather than improving the top of the list.

A plausible explanation is domain and objective mismatch: B3 was trained for generic MS-MARCO passage ranking, whereas the Helix benchmark asks a reranker to distinguish closely related support-policy documents and preserve governing-policy evidence. That is an interpretation of the observed pattern, not a claim that has been independently proven by this checkpoint.

The industrial implication is useful: adding a reranker because it is a standard modern search component would make this system slower and worse under the registered development test. The evidence therefore supports keeping the simpler B1 retrieval path unless a later method with a genuinely different scientific rationale is explicitly registered and evaluated.

## Limitations

This is development evidence only. The 616-query retrieval confirmatory partition remains sealed.

The benchmark maps natural BANKING77 utterances to a fictional HelixBank policy corpus through deterministic relevance judgments. It is not a sample of live enterprise-search traffic.

Only one fixed off-the-shelf reranker was registered. This result does not establish that all cross-encoders, domain-adapted rerankers, or learned ranking objectives would fail.

GitHub-hosted CPU timings should not be generalized to production hardware.

## Decision

**B3 is rejected. B1 remains the leading Phase 3 retrieval reference.**

No post-result reranker rescue, candidate-depth adjustment, alternate cross-encoder, fine-tuning, or confirmatory retrieval evaluation is authorized by this checkpoint.

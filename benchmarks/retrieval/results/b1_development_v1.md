# Phase 3 B1 Dense Retrieval Development Checkpoint

> **Verdict: accepted as the dense development reference.** The sealed retrieval confirmatory partition remains unopened.

## Result

| Metric | B0 BM25 | B1 dense | B1 - B0 |
|---|---:|---:|---:|
| nDCG@10 | 0.3832 | **0.6537** | **+0.2705** |
| MRR@10 | 0.4208 | **0.7005** | **+0.2797** |
| Recall@20 | 0.7226 | **0.9325** | **+0.2100** |
| Recall@50 | 0.8687 | **0.9881** | **+0.1194** |
| Success@1 | 0.3384 | **0.6248** | **+0.2864** |
| Governing-policy recall@20 | 0.6522 | **0.9228** | **+0.2706** |

B1 materially improves every aggregate retrieval metric on the frozen 1,386-query development benchmark. The complete ranking SHA-256 is:

`b51d2649453e2ddedfde7d76525f11d41f37fe364077d840c048378d4a33fe20`

The 616-query confirmatory retrieval partition was not opened, and the official BANKING77 test split was not accessed.

## Frozen dense retriever

B1 uses `BAAI/bge-small-en-v1.5` at immutable revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` with:

- MIT licence;
- model weight SHA-256 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`;
- 384-dimensional normalized embeddings;
- maximum sequence length 512;
- CPU-only inference;
- batch size 64;
- no fine-tuning;
- the model's documented retrieval query instruction;
- title plus body as the passage text;
- normalized dot-product similarity;
- deterministic document-ID tie breaking.

No alternative dense model, query-instruction variant, pooling rule, similarity metric, or fine-tuned configuration was selected after observing the result.

## Reproducibility

Two independent GitHub-hosted CPU executions reproduced the scientific result exactly.

| Evidence | Run 32256394681 | Run 32257841759 |
|---|---|---|
| Ranking SHA-256 | `b51d2649…fe20` | `b51d2649…fe20` |
| Aggregate metrics | identical | identical |
| Per-intent metrics | identical | identical |
| Similarity diagnostics | identical | identical |
| Markdown report SHA-256 | `9b1453f4…aed5` | `9b1453f4…aed5` |
| Artifact ZIP SHA-256 | `1083c4b9…f9f5` | `9f66f6e3…75e0` |

The `results.json` byte hashes differ because each artifact records wall-clock timing. The retrieval decisions, ranking hash, all aggregate metrics, all per-intent metrics, similarity diagnostics, model hash, and scientific environment fields are identical.

This supports **exact ranking and metric reproducibility with runner-dependent timing**, not byte-identical timing evidence.

## Intent-level behavior

The aggregate gain is broad rather than being driven by a small number of categories.

| Metric | B1 better | Tied | B1 worse |
|---|---:|---:|---:|
| nDCG@10 | **74** | 0 | 3 |
| MRR@10 | **73** | 0 | 4 |
| Recall@20 | **68** | 7 | 2 |
| Recall@50 | **58** | 16 | 3 |
| Success@1 | **68** | 5 | 4 |
| Governing-policy recall@20 | **70** | 6 | 1 |

The three nDCG@10 regressions relative to B0 are:

| Intent | B0 | B1 | Change |
|---|---:|---:|---:|
| `supported_cards_and_currencies` | 0.3333 | 0.1942 | -0.1391 |
| `apple_pay_or_google_pay` | 0.6157 | 0.5310 | -0.0847 |
| `exchange_rate` | 0.8383 | 0.8107 | -0.0277 |

The only governing-policy recall@20 regression is `declined_transfer`, from 0.9444 under B0 to 0.8889 under B1.

These regressions matter because they show that the lexical and dense systems retain complementary strengths. B1 is much stronger overall, but BM25 still preserves useful signal in a small number of intent groups.

## Remaining weak areas

The ten weakest B1 intent groups by nDCG@10 are:

| Intent | nDCG@10 |
|---|---:|
| `get_physical_card` | 0.1130 |
| `supported_cards_and_currencies` | 0.1942 |
| `card_linking` | 0.2780 |
| `card_arrival` | 0.2921 |
| `card_delivery_estimate` | 0.3230 |
| `order_physical_card` | 0.3457 |
| `card_acceptance` | 0.3667 |
| `transfer_timing` | 0.3764 |
| `card_swallowed` | 0.3887 |
| `getting_spare_card` | 0.4026 |

The strongest residual cluster is the card lifecycle: obtaining a physical card, ordering, arrival, delivery timing, linking, acceptance, and spare-card requests remain semantically close enough to challenge the dense model.

This matters for later fusion and reranking because aggregate retrieval quality is already high; remaining value must come from reducing these local failures rather than merely reproducing B1's broad gains.

## Source-taxonomy fidelity

Two BANKING77 labels have unusual surface forms: `Refund_not_showing_up` and `reverted_card_payment?`. They are preserved from the source taxonomy rather than introduced by Helix. The retrieval benchmark retains source intent strings exactly so its hashes and data lineage remain reproducible.

## Timing

The two successful CI runs show material wall-clock variation:

| Component | Earlier run | Later run |
|---|---:|---:|
| Model load | 0.83 s | 1.64 s |
| Document encoding | 10.30 s | 8.66 s |
| Query encoding | 24.84 s | 20.95 s |
| Similarity + ranking | 0.116 s | 0.124 s |

These are descriptive GitHub-hosted CPU measurements only. They are not production latency claims and are deliberately excluded from the scientific model comparison.

## Interpretation

B1 establishes that semantic retrieval adds substantial value on the harder natural-language Helix retrieval benchmark. Its gains are especially important at the top of the ranking: Success@1 rises from roughly 0.338 to 0.625 and MRR@10 from roughly 0.421 to 0.701. Governing-policy recall@20 also rises from roughly 0.652 to 0.923, directly improving the probability that a later evidence-grounded system has the relevant policy available within its retrieval window.

The result does not establish that dense retrieval is sufficient by itself. B1 still loses to BM25 in a small number of intent groups and remains weak on several card-lifecycle categories. Those residual differences create a real empirical target for reciprocal-rank fusion rather than assuming hybrid retrieval must help.

## Limitations

This remains development evidence. The sealed retrieval confirmatory partition is still untouched.

The benchmark maps natural BANKING77 queries to a fictional HelixBank policy corpus through deterministic intent-based relevance judgments. That is stronger than the original templated-query benchmark for natural-language retrieval testing, but it is not a sample of live enterprise-search traffic and does not capture every form of multi-document or partially relevant evidence found in production knowledge bases.

The model is evaluated as a fixed off-the-shelf bi-encoder. The result does not establish that BGE is globally optimal, and no such claim is needed for this bounded comparison.

## Decision

**B1 is accepted as the frozen dense development reference.** It materially exceeds B0 on every aggregate relevance metric, reproduces the exact ranking and metrics across independent CPU runs, and retains visible intent-level weaknesses rather than hiding them behind the aggregate score.

The next evaluation step is **B2 reciprocal-rank fusion of the already-frozen B0 and B1 rankings** on the same development benchmark. B2 must demonstrate value against the now-strong B1 reference rather than merely outperforming BM25.

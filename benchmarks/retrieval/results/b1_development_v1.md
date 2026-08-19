# Phase 3 B1 Dense Retrieval Development Checkpoint

> **Verdict: accepted as the leading dense development reference.** The sealed retrieval confirmatory partition remains unopened.

## Result

| Metric | B0 BM25 | B1 dense | B1 - B0 |
|---|---:|---:|---:|
| nDCG@10 | 0.3832 | **0.6537** | **+0.2705** |
| MRR@10 | 0.4208 | **0.7005** | **+0.2797** |
| Recall@20 | 0.7226 | **0.9325** | **+0.2100** |
| Recall@50 | 0.8687 | **0.9881** | **+0.1194** |
| Success@1 | 0.3384 | **0.6248** | **+0.2864** |
| Governing-policy recall@20 | 0.6522 | **0.9228** | **+0.2706** |

B1 materially improves every aggregate retrieval metric on the frozen 1,386-query development benchmark. The accepted complete-ranking SHA-256 is:

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
- title plus body as passage text;
- normalized dot-product similarity;
- deterministic document-ID tie breaking.

No alternative dense model, query-instruction variant, pooling rule, similarity metric, or fine-tuned configuration was selected after observing the result.

## Reproducibility evidence and correction

The original accepted B1 run and an independent replication reproduced the accepted ranking hash, aggregate metrics, per-intent metrics, and similarity diagnostics exactly. A later standalone B1 run on the current Phase 3 branch again reproduced the same accepted ranking hash and all metrics exactly.

A subsequent **B2 parent-reconstruction audit** exposed an important boundary to the earlier wording. One fresh hosted CPU reconstruction produced a different complete B1 ranking SHA-256:

`4161dc01a108f067678ae116ce11a59f21704d81ea7f6933c545a6cabb78964d`

instead of the accepted:

`b51d2649453e2ddedfde7d76525f11d41f37fe364077d840c048378d4a33fe20`

The B2 evaluator failed closed before fusion, so that alternative complete ranking was not silently accepted and no B2 replication metric set was emitted from it.

After that event, a fresh standalone B1 run again reproduced the accepted B1 ranking hash and all aggregate/per-intent metrics exactly.

The correct claim is therefore:

- B1's accepted development metrics are repeatedly reproducible in standalone B1 evaluations;
- the accepted full-ranking hash has also reproduced repeatedly;
- **universal bitwise identity of the complete floating-point ranking across every heterogeneous hosted CPU execution context is not claimed**;
- wall-clock timing remains runner-dependent.

This correction does not change the B1 metric values or its status as the leading measured retrieval reference. It narrows an earlier reproducibility statement that was too broad.

## Intent-level behavior

The aggregate gain remains broad:

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

These local regressions show that lexical retrieval retains useful information, but they do not overturn B1's much stronger aggregate performance.

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

The strongest residual cluster remains the card lifecycle: physical-card acquisition, ordering, arrival, delivery timing, linking, acceptance, and spare-card requests are semantically close enough to challenge the dense model.

## Source-taxonomy fidelity

The unusual BANKING77 labels `Refund_not_showing_up` and `reverted_card_payment?` are preserved from the source taxonomy rather than introduced by Helix. Source intent strings remain unchanged for data-lineage reproducibility.

## Interpretation

B1 establishes that semantic retrieval adds substantial value on the harder natural-language Helix retrieval benchmark. Success@1 rises from roughly 0.338 to 0.625, MRR@10 from roughly 0.421 to 0.701, and governing-policy recall@20 from roughly 0.652 to 0.923.

The later B2 experiment strengthens rather than weakens the selection decision: fixed equal-weight RRF improves some lexical-favored intents but degrades all six aggregate metrics relative to B1. B1 therefore remains the leading Phase 3 retrieval reference.

## Limitations

This remains development evidence. The sealed retrieval confirmatory partition is still untouched.

The benchmark maps natural BANKING77 queries to a fictional HelixBank policy corpus through deterministic intent-based relevance judgments. It is not live enterprise-search traffic.

Hosted CPU timing is descriptive only. Complete floating-point ranking bytes should not be assumed universally bitwise identical across every heterogeneous hosted CPU execution context even when the model weights, software stack, benchmark bytes, and deterministic-algorithm settings are fixed.

The model is evaluated as a fixed off-the-shelf bi-encoder. The result does not establish that BGE is globally optimal.

## Decision

**B1 remains the leading frozen dense development reference.** Its development metrics are unchanged and repeatedly reproduced. The earlier statement implying universal bitwise complete-ranking reproducibility across hosted CPU runners is withdrawn after a later B2 parent-reconstruction counterexample.

B2 is separately preserved as a negative development result. No confirmatory retrieval evaluation is opened by this correction.

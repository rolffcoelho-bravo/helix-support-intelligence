# Phase 3 B2 Reciprocal Rank Fusion Development Checkpoint

> **Verdict: rejected as the selected retrieval reference.** B2 beats B0 but loses materially to B1 on every aggregate metric. The sealed retrieval confirmatory partition remains unopened.

## Frozen method

B2 was registered before its first development score as one fixed Reciprocal Rank Fusion candidate over the already-frozen B0 and B1 complete rankings:

- equal weights: B0 = 1.0, B1 = 1.0;
- `k = 60`;
- one-based ranks;
- full 147-document rank depth;
- no score normalization;
- no learned weights;
- deterministic document-ID tie breaking;
- fail closed if either parent ranking hash differs from the accepted B0/B1 evidence.

The scientific freeze was committed before scoring. Two later pre-score failures were static Ruff/formatter defects only; their repairs did not change `k`, weights, depth, parent hashes, benchmark bytes, or metric definitions.

## First valid development result

| Metric | B0 BM25 | B1 dense | B2 RRF | B2 - B1 |
|---|---:|---:|---:|---:|
| nDCG@10 | 0.3832 | **0.6537** | 0.5172 | **-0.1365** |
| MRR@10 | 0.4208 | **0.7005** | 0.5976 | **-0.1030** |
| Recall@20 | 0.7226 | **0.9325** | 0.8813 | **-0.0512** |
| Recall@50 | 0.8687 | **0.9881** | 0.9740 | **-0.0141** |
| Success@1 | 0.3384 | **0.6248** | 0.4899 | **-0.1349** |
| Governing-policy recall@20 | 0.6522 | **0.9228** | 0.8348 | **-0.0880** |

B2 improves every aggregate metric over B0, but the actual Phase 3 selection problem is whether fusion adds value over the much stronger B1 dense reference. It does not. All six aggregate metrics deteriorate.

The first valid B2 fused ranking SHA-256 is:

`e02f4ccf4358a18cea75c0ee0b26da1cd83e51aab9903ba3554a9154585ece51`

The run reconstructed the frozen benchmark exactly, reproduced both accepted parent ranking hashes, verified the B1 model-weight checksum, kept the 616-query confirmatory retrieval partition sealed, and did not access the official BANKING77 test split.

## Breadth of the regression

The aggregate loss is not caused by one narrow intent group.

| Metric | B2 better than B1 | Tied | B2 worse than B1 |
|---|---:|---:|---:|
| nDCG@10 | 4 | 0 | **73** |
| MRR@10 | 15 | 2 | **60** |
| Recall@20 | 5 | 18 | **54** |
| Recall@50 | 2 | 54 | **21** |
| Success@1 | 9 | 16 | **52** |
| Governing-policy recall@20 | 2 | 23 | **52** |

The frozen equal-weight fusion therefore over-weights the materially weaker lexical parent often enough to damage the dense reference broadly.

This does **not** support a general claim that Reciprocal Rank Fusion is ineffective. It supports the narrower statement that this fixed B0+B1 equal-weight configuration is not the right selected retriever for the Helix development benchmark.

## Local lexical recoveries

The negative aggregate result still reveals useful complementarity. B2 improves B1 nDCG@10 in four intents:

| Intent | B1 | B2 | Change |
|---|---:|---:|---:|
| `exchange_rate` | 0.8107 | 0.8928 | +0.0821 |
| `apple_pay_or_google_pay` | 0.5310 | 0.6082 | +0.0773 |
| `declined_card_payment` | 0.4057 | 0.4762 | +0.0705 |
| `supported_cards_and_currencies` | 0.1942 | 0.2573 | +0.0631 |

It also improves governing-policy recall@20 for `card_acceptance` and `declined_transfer`; the latter returns from B1's 0.8889 to 0.9444.

Those recoveries show why lexical evidence remains diagnostically useful. They do not justify selecting B2 because the cost elsewhere is much larger and much broader.

## Strict reproducibility audit

The first valid B2 run passed the frozen parent-integrity checks. A fresh replication attempt then reproduced the benchmark and static gates but **aborted before fusion** because the reconstructed B1 complete-ranking SHA changed from the accepted:

`b51d2649453e2ddedfde7d76525f11d41f37fe364077d840c048378d4a33fe20`

to:

`4161dc01a108f067678ae116ce11a59f21704d81ea7f6933c545a6cabb78964d`

The evaluator failed closed as designed, so no second B2 metric set was emitted and no alternative parent ranking was silently accepted.

A subsequent standalone B1 execution again reproduced the accepted B1 ranking hash and all B1 aggregate/per-intent metrics exactly. The correct interpretation is therefore narrower than the earlier B1 wording: multiple standalone B1 executions reproduce the accepted scientific result, but universal bitwise identity of a complete floating-point ranking across heterogeneous hosted CPU execution contexts is not guaranteed.

This distinction matters especially for B2 because the frozen RRF candidate consumes the full 147-document parent rankings, including deep rank positions that ordinary top-k relevance metrics do not directly summarize.

## Why no rescue tuning follows

The result does not authorize searching over RRF constants, unequal weights, truncated ranking depths, score normalization, or learned fusion. Doing that after seeing the development result would convert the registered comparison into model shopping.

The correct scientific response is to preserve the negative result and retain B1 as the leading retrieval reference.

## Reproducibility evidence

First valid B2 run:

- workflow run `32266247648`;
- job `96111343774`;
- artifact `9370336043`;
- artifact ZIP SHA-256 `d85624c12623e0bc94ebc623737797aa582920b8b5ec5c9a3ad385aeb7b6e3ca`;
- result JSON SHA-256 `0b1e4376cb89abacefe3a11471b5d69872e156ebf09a9badf6eb9fcfe71df477`;
- report SHA-256 `cf877498a89998db34afe329fbbe33f4a3c414bc93f688630647f1a93285265b`.

Strict replication attempt:

- same workflow run, rerun job `96112253707`;
- benchmark reconstruction and pre-evaluation gates passed;
- B1 parent full-ranking integrity check failed;
- fusion and B2 metric emission were blocked.

Follow-up standalone B1 run:

- workflow run `32266247568`;
- job `96113632455`;
- accepted B1 ranking SHA reproduced exactly;
- accepted B1 aggregate and per-intent metrics reproduced exactly.

## Limitations

This is development evidence only. The sealed retrieval confirmatory partition remains untouched.

Only one valid B2 metric run exists because the strict replication attempt deliberately failed closed rather than accepting a different full B1 parent ranking. Consequently B2 cannot claim exact replicated fused-ranking evidence.

The benchmark maps natural BANKING77 utterances to a fictional HelixBank policy corpus through deterministic intent-based qrels. It is not live enterprise-search traffic.

Hosted CPU timing is descriptive only and is not a production latency claim.

## Decision

**B2 is rejected. B1 remains the leading Phase 3 retrieval reference.**

The B2 result adds value precisely because it is negative: a standard fixed equal-weight rank-fusion step that looks reasonable architecturally does not improve the stronger dense system here. The repository preserves that evidence instead of tuning until fusion wins.

No reranker or confirmatory retrieval evaluation is opened by this checkpoint.

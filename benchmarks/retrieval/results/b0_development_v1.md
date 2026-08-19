# Phase 3 B0 BM25 Development Checkpoint

> **Verdict: PASSED as the frozen lexical development reference.** The Phase 3 confirmatory retrieval partition remains sealed.

## Result

| Metric | B0 development |
|---|---:|
| nDCG@10 | **0.3832** |
| MRR@10 | **0.4208** |
| Recall@20 | **0.7226** |
| Recall@50 | **0.8687** |
| Success@1 | **0.3384** |
| Governing-policy / citation-eligible recall@20 | **0.6522** |

The benchmark contains 1,386 natural-language development queries, 147 eligible candidate documents, and 2,646 graded relevance judgments. The 616-query confirmatory partition was not opened or exported to the B0 evaluator. The official BANKING77 test source was not accessed.

## Frozen B0 configuration

B0 is standard dependency-light Okapi BM25:

- title and body concatenated into one text field;
- Unicode NFKC normalization followed by casefolding;
- deterministic alphanumeric tokenization;
- no stopword removal;
- no stemming;
- `k1 = 1.2`;
- `b = 0.75`;
- positive Robertson/Sparck-Jones IDF `log(1 + (N-df+0.5)/(df+0.5))`;
- deterministic document-ID ascending tie break.

No BM25 parameter was selected after seeing the development result.

## Reproducibility

The accepted workflow is Actions run `32251150616`, artifact `9364444799`, artifact digest `sha256:e1c0955b8ebcb7c690e18e148d6589b826025e3af1db574e6e7207bdef35327e`.

The complete ranking over all 147 candidate documents for all 1,386 development queries has deterministic SHA-256:

`e82372d60779f211b4692709db623f3e7ad2a17922e450582ea7487c92b7e41b`

An earlier workflow (`32251072682`) completed the same scientific computation but the step went red because `tee` attempted to create its log file before the output directory existed. That earlier computation produced the **same ranking hash and exactly the same six relevance metrics**.

A separate post-result reconstruction was then performed from the frozen `documents.jsonl`, `development_queries.jsonl`, and `development_qrels.jsonl` without importing the Helix BM25 or metric implementation. It reproduced the ranking hash exactly and every relevance metric with maximum absolute difference `0.0`.

## Descriptive latency

On the accepted GitHub-hosted CI runner:

- index build: approximately `3.51 ms`;
- per-query score/sort P50: approximately `0.36 ms`;
- per-query score/sort P95: approximately `0.65 ms`.

These values are descriptive CI evidence only. The earlier run produced different timing while preserving identical rankings and relevance metrics, demonstrating why these timings must not be presented as hardware-independent production latency.

## Error structure

B0 is far from uniformly strong. The ten weakest intent groups by development nDCG@10 are:

| Intent | nDCG@10 |
|---|---:|
| `getting_spare_card` | 0.0119 |
| `compromised_card` | 0.0441 |
| `card_arrival` | 0.0556 |
| `get_physical_card` | 0.0640 |
| `declined_cash_withdrawal` | 0.0768 |
| `card_swallowed` | 0.0855 |
| `card_linking` | 0.0924 |
| `card_acceptance` | 0.0928 |
| `transfer_timing` | 0.0974 |
| `topping_up_by_card` | 0.1170 |

Governing-policy recall@20 is especially weak for `card_acceptance`, `card_linking`, `compromised_card`, and `getting_spare_card`, each approximately `0.0556` on the balanced development sample.

These weaknesses are not repaired by tuning B0. They become the registered lexical baseline against which B1 dense retrieval and later hybrid/reranking stages must demonstrate value.

## Interpretation

The harder natural-language benchmark behaves as intended: it does **not** make lexical retrieval look artificially excellent. B0 retrieves at least one relevant item reasonably often at deeper ranks (`Recall@50 ≈ 0.869`), but its first-result usefulness (`Success@1 ≈ 0.338`) and governing-policy recall at the citation-relevant depth (`≈ 0.652`) leave substantial headroom.

This is useful evidence for the blueprint hypothesis that semantic retrieval and hybridization may add value. It is not evidence that H1 is supported; H1 requires the registered B2-vs-B0/B1 confirmatory comparison later in Phase 3.

## Audit findings corrected during this checkpoint

1. Static Ruff and formatting defects blocked early runs before scoring and were corrected without changing science.
2. The first successful scientific calculation exposed an artifact-transport bug: the B0 output directory had not been created before `tee` opened its log path. The directory creation was added and the unchanged benchmark was rerun.
3. The benchmark materializer reproduced all frozen hashes before each B0 computation.
4. No sealed confirmatory file entered the B0 evaluator.

## Decision

B0 is accepted and frozen as the Phase 3 lexical development reference. The next blueprint action is to register the exact B1 dense bi-encoder model, immutable revision, licence, query/document formatting, normalization, and dependency/hardware contract **before B1 receives any development score**.

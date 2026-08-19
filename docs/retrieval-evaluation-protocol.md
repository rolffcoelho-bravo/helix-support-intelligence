# Retrieval and Reranking Evaluation Protocol

## Objective

Helix evaluates retrieval independently from answer generation so that relevance failures remain directly measurable. The comparison covers four bounded retrieval configurations:

- **B0:** Okapi BM25 lexical retrieval;
- **B1:** dense bi-encoder retrieval;
- **B2:** reciprocal-rank fusion of B0 and B1;
- **B3:** B2 candidate generation followed by a cross-encoder reranker.

The comparison is intentionally bounded. A more complex retrieval component is retained only when its measured relevance benefit justifies the added latency and implementation complexity.

## Scientific questions

### H1 — hybrid retrieval

B2 is compared with both B0 and B1. The primary endpoint is `nDCG@10` on the frozen confirmatory retrieval partition.

### H2 — reranking

B3 is compared with B2. The primary endpoint is `MRR@10` on the frozen confirmatory retrieval partition. Reranking is useful only when the relevance gain justifies the additional latency.

## Benchmark hierarchy

### Primary natural-language benchmark

The primary retrieval benchmark is `retrieval-benchmark-v1`. It uses natural BANKING77 utterances from the pinned **source training file only** and maps their intent labels to the frozen fictional HelixBank policy corpus.

Benchmark construction:

1. verify the pinned BANKING77 train-source checksum;
2. replay the leakage quarantine and deterministic validation assignment;
3. use only the resulting `fit_train` partition as the retrieval-query source;
4. use a separate deterministic salt to select 18 development and 8 confirmatory utterances for each of 77 intents;
5. require development and confirmatory query IDs to be disjoint;
6. avoid accessing the official BANKING77 test source during retrieval development.

This produces **1,386 development queries** and **616 confirmatory queries**.

The first real-data feasibility check showed that the smallest eligible intent contains 27 rows. The initially proposed 20+10 allocation was therefore infeasible and was reduced to 18+8 before any retrieval score or benchmark hash was frozen.

The confirmatory query contents are not materialized by the development workflow. Their deterministic hashes are recorded before model comparison and the contents remain unavailable to development evaluators.

### Secondary HelixBank contract suite

The original 308 generated HelixBank queries remain useful for deterministic structural and edge-case checks such as archived evidence, ambiguity, missing evidence, conflicts, and citation eligibility.

They are not used as the primary retrieval-quality comparison because their wording is mechanically derived from the same intent names that appear in the corpus documents, creating unusually high lexical overlap. Separating this contract suite from the natural-language benchmark avoids overstating lexical retrieval quality.

## Candidate-document eligibility

Eligibility is applied before scoring:

- `corpus_version == helixbank-policy-v1.0.0`;
- `status == current`;
- `permission == public_support`.

Archived documents are excluded. Current controlled-conflict fixtures remain eligible because conflict handling is a downstream system responsibility rather than a retrieval-time deletion rule.

The frozen candidate set contains **147 documents**.

## Relevance semantics

For each natural-language query:

- the governing policy for the true intent receives relevance grade `3`;
- the current FAQ for that intent receives relevance grade `2`;
- archived FAQs are not eligible candidates;
- unjudged documents receive grade `0`.

The judgments are deterministic consequences of the frozen corpus contract and are versioned before model scoring.

## Metrics

Required metrics are:

- `nDCG@10` — H1 primary endpoint;
- `MRR@10` — H2 primary endpoint;
- `Recall@20`;
- `Recall@50`;
- success at rank 1;
- governing-policy recall at 20;
- P50 and P95 retrieval latency;
- index build time, index size, and memory where applicable;
- per-intent robustness diagnostics.

Latency measured on GitHub-hosted CI is descriptive development evidence unless tied to a declared reference-hardware configuration. It is not presented as a production latency guarantee.

## B0 — lexical baseline

B0 uses standard Okapi BM25 over concatenated document title and body text.

Configuration:

- Unicode NFKC normalization;
- case folding;
- deterministic alphanumeric tokenization;
- `k1 = 1.2`;
- `b = 0.75`;
- positive Robertson/Sparck-Jones-style IDF `log(1 + (N-df+0.5)/(df+0.5))`;
- deterministic ascending document-ID tie breaking;
- no stopword removal, stemming, or parameter search.

B0 is intentionally simple and dependency-light so it provides a credible lexical reference for later comparisons.

## B1 — dense retrieval

The dense bi-encoder model ID, immutable revision, licence, query/document formatting, normalization, batch configuration, dependency environment, and hardware mode are recorded before its development score is accepted.

B1 uses the same frozen query and candidate-document sets as B0.

## B2 — reciprocal-rank fusion

B2 combines the ranked lists from B0 and B1 through reciprocal-rank fusion. The fusion constant and candidate depths are fixed before the B2 comparison. No result-driven fusion-weight search is used.

## B3 — cross-encoder reranking

B3 reranks the frozen B2 candidate set with a versioned cross-encoder. Model identity, revision, licence, input formatting, candidate depth, and execution environment are recorded before evaluation.

## Development and confirmatory evidence

Development queries may be used to compare B0-B3 and to make the bounded configuration decisions declared above. Confirmatory queries are excluded from model choice, model revision, tokenization, fusion parameters, candidate depth, and relevance/latency trade-off selection.

After one final retrieval configuration is fixed, the confirmatory partition is evaluated once using the registered H1/H2 endpoints.

Negative or inconclusive results remain valid evidence. A simpler system remains preferable when additional complexity does not produce a meaningful relevance benefit.

## Reproducibility and integrity

Each accepted retrieval result must support verification of:

- benchmark and candidate-document hashes;
- metric definitions and deterministic tie behaviour;
- query/qrel isolation between development and confirmatory partitions;
- model and dependency versions;
- hardware interpretation for timing measurements;
- workflow exit-code propagation;
- deterministic ranking hashes where applicable;
- public claims that do not exceed the measured evidence.

The benchmark-freeze and B0 workflows use `pipefail` so failures in scientific commands cannot be hidden by downstream logging commands such as `tee`.

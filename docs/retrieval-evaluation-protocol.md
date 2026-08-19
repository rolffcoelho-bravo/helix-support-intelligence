# Phase 3 Retrieval Evaluation Protocol

## Scope and phase lock

Phase 3 is the only active implementation phase. It implements the blueprint-defined search and reranking workstream only. Evidence-bound generation, recommendation, guardrails, observability, and later-phase features remain unopened.

The retrieval ladder is bounded:

- **B0:** Okapi BM25 lexical baseline;
- **B1:** dense bi-encoder retrieval;
- **B2:** reciprocal-rank fusion of B0 and B1;
- **B3:** B2 candidate generation followed by a cross-encoder reranker.

No additional retrieval family enters Phase 3 merely to rescue a preferred result.

## Scientific hypotheses

### H1 — hybrid retrieval

Hybrid retrieval B2 is compared with both B0 and B1. The primary endpoint is `nDCG@10` on the frozen confirmatory retrieval partition. Development selection occurs before confirmatory access.

### H2 — reranking

B3 is compared with B2. The primary endpoint is `MRR@10` on the frozen confirmatory retrieval partition. B3 ships only when relevance benefit justifies its latency increase.

## Benchmark hierarchy

### Primary natural-language benchmark

The primary Phase 3 scientific benchmark is `retrieval-benchmark-v1`. It uses natural BANKING77 utterances from the already-pinned **source training file only** and maps their frozen intent labels to the already-frozen fictional HelixBank policy corpus.

Before Phase 3 scoring:

1. the pinned BANKING77 train source checksum is verified;
2. the Phase 1 leakage quarantine and deterministic validation assignment are replayed;
3. only the resulting `fit_train` partition is eligible as a Phase 3 query source;
4. a Phase 3-specific salted deterministic ranking selects 20 development and 10 confirmatory utterances per each of 77 intents;
5. the two retrieval partitions are disjoint by stable sample ID;
6. the official BANKING77 test URL is never accessed by the Phase 3 benchmark materializer.

This produces 1,540 development queries and 770 confirmatory queries.

The confirmatory query contents are not materialized by the pre-scoring hash-freeze workflow. Their deterministic hash is frozen before B0 scoring, and they are opened only after B0-B3 selection and a final pre-confirmatory audit.

### Secondary HelixBank contract / edge-case suite

The original 308 generated HelixBank queries remain valuable for deterministic contract cases such as archived evidence, ambiguity, missing evidence, conflicts, and citation eligibility. They are **not** the primary retrieval-quality comparison because their wording is generated directly from intent names that also appear in the corpus documents, creating unusually high lexical overlap.

This separation prevents a templated synthetic query set from overstating general retrieval performance.

## Candidate-document eligibility

Eligibility is applied before scoring:

- corpus version must equal `helixbank-policy-v1.0.0`;
- `status == current`;
- `permission == public_support`.

Archived documents are excluded from the candidate set. Current controlled-conflict fixtures remain retrievable because later phases must detect and handle conflicting evidence rather than silently delete it.

## Relevance semantics

For each natural-language query:

- the governing policy for the true intent receives relevance grade `3`;
- the current FAQ for that intent receives relevance grade `2`;
- an archived FAQ is not an eligible candidate;
- unjudged documents receive grade `0`.

These qrels are deterministic consequences of the Phase 1 corpus contract and are frozen before model scoring.

## Metrics

Required Phase 3 metrics:

- `nDCG@10` — H1 primary endpoint;
- `MRR@10` — H2 primary endpoint;
- `Recall@20`;
- `Recall@50`;
- success at rank 1;
- citation-eligible evidence recall at 20;
- P50 and P95 retrieval latency;
- index build time, index size, and memory where applicable;
- robustness by query/intent group.

Latency produced on GitHub-hosted CI is descriptive development evidence unless explicitly tied to declared reference hardware. No cloud-independent production latency claim is permitted.

## B0 contract

B0 uses standard Okapi BM25 over one concatenated text field containing document title and body.

Frozen defaults before B0 scoring:

- Unicode NFKC + casefold normalization;
- deterministic alphanumeric tokenization;
- `k1 = 1.2`;
- `b = 0.75`;
- Robertson/Sparck-Jones-style positive IDF `log(1 + (N-df+0.5)/(df+0.5))`;
- deterministic tie-breaking by document ID.

B0 is intentionally simple and dependency-light. It exists to provide a credible lexical reference, not to maximize BM25 through an open-ended parameter search.

## Candidate registration

B1 and B3 model IDs, licences, immutable revisions, input formatting, candidate depth, hardware environment, and any query/document prefixes must be frozen and audited **before their first score is produced**.

B2 fusion weights and RRF constant must likewise be frozen before B2 evaluation. The default blueprint form is reciprocal-rank fusion; post-result fusion-weight shopping is prohibited.

## Development versus confirmatory evidence

Development queries may be used for B0-B3 comparison and the bounded configuration decisions declared above. The confirmatory partition may not be inspected for model choice, model revision, tokenization, fusion parameters, reranker depth, or latency/relevance tradeoff selection.

After one retrieval configuration is frozen, a pre-confirmatory integrity audit must pin all selection-critical artifacts. The confirmatory partition is then opened once for registered H1/H2 evaluation.

## Failure policy

If H1 or H2 fails:

1. verify data, qrels, metrics, implementation, candidate filtering, and dependencies;
2. perform only the bounded remediation allowed by the blueprint;
3. preserve the negative or inconclusive result;
4. keep the simpler valid system when the more complex candidate does not justify itself;
5. close Phase 3 rather than extending the model family.

## Mandatory hostile execution audit

Every Phase 3 execution closes only after checking:

- code and calculation correctness;
- metric semantics and ranking tie behavior;
- candidate eligibility and qrel correctness;
- development/confirmatory isolation;
- official BANKING77 test non-access during Phase 3 development;
- reproducibility and deterministic hashes;
- dependency and hardware consistency;
- workflow permissions and triggers;
- public wording and private-information leakage;
- consistency with the normative blueprint.

Material defects are corrected in the same execution. Negative scientific results are never tuned away.

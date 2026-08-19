# Phase 3 Exit Report

- Phase: Search and reranking
- Status: **Active — primary benchmark frozen; B0 audited development baseline complete; B1 registration next**
- Date opened: 2026-08-19
- Public version: 0.1.0

## Phase lock

Phase 2 is merged and closed on `main`. Phase 3 is the only active implementation phase. Phase 4 evidence-bound assistance and recommendation have not started.

## Blueprint objective

Produce independently evaluated evidence retrieval through the bounded B0-B3 ladder:

- B0 BM25;
- B1 dense bi-encoder retrieval;
- B2 reciprocal-rank hybrid fusion;
- B3 hybrid fusion plus cross-encoder reranking.

## Required deliverables

- [ ] B0-B3 retrieval ladder completed.
- [x] Retrieval evaluation protocol defined.
- [x] Primary natural-language retrieval benchmark construction protocol frozen before scoring.
- [x] Primary benchmark hash manifest frozen and audited.
- [x] B0 BM25 baseline implemented, development-evaluated, and hostile-audited.
- [ ] B1 dense bi-encoder registered before scoring and evaluated.
- [ ] B2 RRF hybrid configuration registered before scoring and evaluated.
- [ ] B3 cross-encoder reranker registered before scoring and evaluated.
- [ ] Vespa schema and reproducible index build.
- [ ] Retrieval benchmark and latency report.
- [ ] `/v1/search` integration tests.
- [ ] One final retrieval configuration frozen using the relevance-latency rule.
- [ ] Pre-confirmatory integrity audit.
- [ ] One-shot registered H1/H2 confirmatory result.

## Primary benchmark audit and freeze

The Phase 1 HelixBank corpus contains 308 deterministic golden queries whose wording is generated directly from the same intent names used in document titles and bodies. That set remains useful for contract and edge-case behavior, but using it as the primary Phase 3 relevance benchmark would create unusually favorable lexical overlap and could overstate BM25 and hybrid-search quality.

This is not treated as a Phase 1 defect requiring the closed phase to be reopened. Phase 3 already owns relevance evaluation under the blueprint. Therefore Phase 3 introduces a primary natural-language retrieval benchmark derived from the pinned BANKING77 **source training file only**.

The construction replays the frozen Phase 1 train quarantine/split contract and samples only `fit_train` utterances with a new deterministic Phase 3 salt:

- 18 development queries per intent = 1,386;
- 8 confirmatory queries per intent = 616;
- 77 intents represented in both partitions;
- development and confirmatory stable IDs are disjoint;
- official BANKING77 test access is forbidden in the materializer.

The first pre-scoring materialization established that the smallest frozen `fit_train` intent, `contactless_not_working`, has only 27 eligible rows. The initially proposed 20+10 allocation was infeasible and was reduced to 18+8 before any retrieval score or benchmark hash freeze.

Frozen benchmark hashes:

- candidate documents: `4149646be5507c1f2aeeef2ea19249b26f03db017cdabe9b3b891c52be3b0637`;
- development queries: `96ef219af4cb3b20b231fc6453119950f84beef7262d7ac6d98cb3604032992a`;
- sealed confirmatory queries: `876c50f177b5f426904ee1662ce72643adddd90d5527577286348eed613c41a2`;
- development qrels: `d13e7af883e9185a701703f8090773a2774e791bff4d7fd302fba0d3d3d90aae`;
- sealed confirmatory qrels: `2e10dec72cb84f4bca09a5117d0ee78824e7168d7310a38984ee3c023fcd57d0`.

The materializer now regenerates and verifies these frozen values before later development scoring. Permanent evidence: `benchmarks/retrieval/results/benchmark_freeze_v1.{json,md}`.

## Candidate eligibility

Documents are filtered before scoring:

- `corpus_version == helixbank-policy-v1.0.0`;
- `status == current`;
- `permission == public_support`.

The deterministic corpus yields 147 eligible candidate documents because seven archived FAQ fixtures are excluded.

## B0 BM25 development checkpoint

B0 was frozen before its first score as standard Okapi BM25 over concatenated document title and body:

- Unicode NFKC normalization + casefold;
- deterministic alphanumeric tokenization;
- no stopword removal or stemming;
- `k1 = 1.2`;
- `b = 0.75`;
- positive RSJ-style IDF;
- deterministic ascending document-ID tie break.

Metric semantics were frozen in `configs/retrieval/b0_bm25_v1.json` before scoring.

| Metric | B0 development |
|---|---:|
| nDCG@10 | **0.3832** |
| MRR@10 | **0.4208** |
| Recall@20 | **0.7226** |
| Recall@50 | **0.8687** |
| Success@1 | **0.3384** |
| Governing-policy / citation-eligible recall@20 | **0.6522** |

The deterministic full-ranking hash is:

`e82372d60779f211b4692709db623f3e7ad2a17922e450582ea7487c92b7e41b`.

An earlier workflow completed the same scientific computation but failed during artifact logging because `tee` tried to open its output path before the B0 directory existed. After fixing the transport path only, the successful rerun reproduced the exact ranking hash and all six relevance metrics.

A separate post-result reconstruction from the frozen benchmark files, without importing the Helix BM25 or metric code, also reproduced the ranking hash and all relevance metrics exactly with maximum absolute metric difference `0.0`.

Permanent evidence: `benchmarks/retrieval/results/b0_development_v1.{json,md}`.

### B0 error structure

B0 is strongly heterogeneous across intents. The weakest development groups include `getting_spare_card`, `compromised_card`, `card_arrival`, `get_physical_card`, `declined_cash_withdrawal`, `card_swallowed`, `card_linking`, `card_acceptance`, `transfer_timing`, and `topping_up_by_card`.

This weakness is preserved as the registered lexical reference rather than repaired through BM25 parameter tuning. It provides genuine headroom for the already-blueprinted semantic B1/B2/B3 ladder without pre-judging H1 or H2.

## Workflow audit corrections

1. The first real-data benchmark feasibility failure exposed missing shell `pipefail`; scientific pipelines now propagate Python failures correctly through `tee`.
2. The initial 20+10 balanced query allocation was corrected to 18+8 before any score/hash freeze.
3. Static Ruff/format defects blocked early B0 runs before scoring and were fixed without scientific changes.
4. The first B0 scientific computation exposed a missing artifact-directory creation before `tee`; that transport defect was fixed, and deterministic scientific outputs reproduced exactly.

## Isolation status

- Phase 3 confirmatory query contents remain unexported to B0.
- No confirmatory query/qrel file entered the B0 evaluator.
- The official BANKING77 test source has not been accessed by Phase 3 development.
- No Phase 4 implementation has started.

## Next locked action

Register B1's exact dense bi-encoder model, immutable revision, licence, query/document formatting, normalization rule, batch/dependency/hardware contract, and evaluation depth **before its first development score**. Then evaluate B1 on the same frozen 1,386-query development benchmark and hostile-audit the result before any B2 fusion experiment begins.

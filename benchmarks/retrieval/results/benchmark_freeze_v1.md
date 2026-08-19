# Phase 3 Retrieval Benchmark Freeze v1

> **Verdict: PASSED.** The primary Phase 3 retrieval benchmark is frozen before any B0-B3 retrieval score is produced.

## Frozen benchmark

| Surface | Frozen value |
|---|---:|
| Eligible candidate documents | 147 |
| Development queries | 1,386 |
| Sealed confirmatory queries | 616 |
| Development qrels | 2,646 |
| Sealed confirmatory qrels | 1,176 |
| Intents | 77 |
| Development queries per intent | 18 |
| Confirmatory queries per intent | 8 |
| Official BANKING77 test accessed | **No** |
| Confirmatory content exported | **No** |
| Retrieval score produced before freeze | **No** |

The benchmark-freeze workflow was GitHub Actions run `32250295522`; its uploaded artifact is `9364137851` with ZIP digest `sha256:00ca1f4cc24ea7fc9311b92811e8b91cda6c50b5a68728ae2bb4ac28f6e3cd49`.

## Frozen source and content hashes

- pinned BANKING77 source-train SHA-256: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`;
- frozen Phase 1 `fit_train` SHA-256: `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3`;
- candidate documents: `4149646be5507c1f2aeeef2ea19249b26f03db017cdabe9b3b891c52be3b0637`;
- development queries: `96ef219af4cb3b20b231fc6453119950f84beef7262d7ac6d98cb3604032992a`;
- sealed confirmatory queries: `876c50f177b5f426904ee1662ce72643adddd90d5527577286348eed613c41a2`;
- development qrels: `d13e7af883e9185a701703f8090773a2774e791bff4d7fd302fba0d3d3d90aae`;
- sealed confirmatory qrels: `2e10dec72cb84f4bca09a5117d0ee78824e7168d7310a38984ee3c023fcd57d0`.

The three exported benchmark files were independently re-hashed from the Actions artifact and matched the manifest exactly. The artifact contained `documents.jsonl`, `development_queries.jsonl`, `development_qrels.jsonl`, `manifest.json`, and `run-output.txt`. It did **not** contain confirmatory query or qrel files.

## Pre-scoring hostile-audit findings

### 1. Templated-query optimism risk

The original 308 generated HelixBank queries directly reuse humanized intent names that also occur in document titles and bodies. They remain useful as a deterministic contract and edge-case suite, but using them as the primary retrieval-quality benchmark could exaggerate lexical retrieval quality.

**Repair:** Phase 3 uses natural BANKING77 utterances from the already-pinned source training file and the frozen Phase 1 `fit_train` partition for its primary retrieval-quality benchmark. The official BANKING77 test is not accessed.

### 2. Infeasible initial balanced split

The first pre-scoring materialization showed that `contactless_not_working`, the smallest frozen `fit_train` intent for this purpose, contains 27 rows. The originally proposed 20-development + 10-confirmatory allocation therefore could not be satisfied for every intent.

**Repair:** the benchmark was reduced to 18 development + 8 confirmatory queries per intent, requiring 26 rows and leaving at least one eligible row unused in every intent. This change occurred before any retrieval score and before benchmark hashes were frozen.

### 3. Scientific workflow exit-code propagation

The first real-data failure was piped through `tee` without shell `pipefail`. The failed Python command therefore did not determine the materialization step exit status, although the workflow still became red when the next step found the missing manifest.

**Repair:** the scientific materialization step now enables `set -o pipefail`, ensuring a Python failure terminates the step directly.

## Interpretation

This checkpoint deliberately produces **no retrieval performance result**. Its purpose is to make later B0-B3 comparisons meaningful and auditable. A retrieval model cannot influence query selection, qrels, candidate-document eligibility, or the sealed confirmatory bytes after this point.

The original HelixBank golden set is not discarded. It remains valuable for archived-document, ambiguity, conflict, missing-evidence, and citation-eligibility contract checks. The new natural-language benchmark serves a different purpose: model-quality comparison under less mechanically favorable lexical overlap.

## Decision

The Phase 3 benchmark is now frozen. B0 BM25 development scoring may begin only after the frozen-manifest verifier and ordinary repository quality gate both pass. The 616-query confirmatory partition remains sealed for the later one-shot H1/H2 evaluation.

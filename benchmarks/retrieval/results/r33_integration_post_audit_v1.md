# Phase 3 R3.3 integration post-audit

**Verdict: PASSED**

R3.3 integrates the already-selected `retrieval-selected-v1` B0 BM25 configuration into `POST /v1/search`. It does not reopen or retune the R3.2 retrieval experiment.

## Quality gate

The validated merge candidate at `fc623b639ef347b5ee8fcb13eefa9beb2418dddc` passed GitHub Actions run `32306778332`, job `96241129523`:

- Ruff check: passed;
- Ruff format check: passed;
- mypy strict check: passed;
- pytest: **108 passed, 0 failed**;
- Phase 1 offline data contracts: passed;
- Phase 3 retrieval preflight: passed with exactly **147 eligible documents**, zero frozen-query scores computed by preflight, and no retrieval result regeneration;
- publication audit: passed.

## Frozen-result reconstruction through HTTP

The integration test replays all **308 registered HelixBank queries** through `POST /v1/search` with depth 50 and compares the returned B0 `(document_id, rank, score)` tuples against the permanent R3.2 raw ranking evidence.

All **15,400 ranked tuples** matched exactly. The HTTP integration therefore preserves the selected R3.2 B0 ranking behavior across the complete frozen query set, not only a representative example.

## Runtime contract checks

The manual review confirmed:

- runtime candidate is B0 BM25;
- `k1 = 1.2`, `b = 0.75`, and top-50 ceiling match `retrieval-selected-v1`;
- evidence eligibility is applied before ranking and produces the expected 147-document index;
- archived documents are not rankable;
- benchmark-only fields such as intent and gold citations are rejected by the request schema;
- zero-score ties use `document_id` ascending;
- repeated identical requests serialize deterministically;
- known backend contract failures return HTTP 503 with `SEARCH_UNAVAILABLE` and do not expose internal exception text;
- Phase 2 scientific files, R3.2 scientific evidence, and `retrieval-selected-v1` are unchanged;
- no post-result retrieval tuning or runtime model substitution was introduced.

## Defects caught before closure

The full repository gate caught three implementation-process defects before merge. First, an old Phase 1 regression test incorrectly required the public experiment registry to remain empty after validated R3.2 evidence had been intentionally published. The invariant was corrected to require validated public experiment entries. Second, the replacement registry test initially failed to recognize YAML list prefixes and was corrected. Third, the strengthened all-query replay assertion required Ruff formatting. None of these repairs changed retrieval science, model selection, ranking policy, or R3.2 evidence.

## Non-blocking maintenance warnings

The locked FastAPI/Starlette test stack currently emits a deprecation warning for the existing httpx-backed TestClient path. All tests pass; this is recorded as dependency maintenance rather than an R3.3 functional failure. GitHub Actions also emits upstream Node compatibility warnings for currently supported actions.

## Boundaries

The corpus is fictional and frozen at the Phase 3 evaluation date. R3.3 exposes retrieval only; generation, answerability decisions, routing composition, authentication, rate limiting, and production observability remain outside this checkpoint. Untrusted-content fixtures are returned as evidence data and must remain non-executable input to later assistance components. R3.2 CPU timing is not relabeled as HTTP production latency.

No R3.2 rerun is required. R3.3 is ready to close once this post-audit artifact itself passes the final repository quality gate.

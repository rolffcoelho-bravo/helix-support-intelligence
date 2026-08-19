# Phase 3 R3.3 integration audit plan

R3.3 integrates the already-selected `retrieval-selected-v1` configuration into the bounded `POST /v1/search` API. It does not reopen or retune the R3.2 retrieval experiment.

The checkpoint closes only after the exact merge candidate passes the full repository quality gate and the post-execution audit confirms:

- the HTTP path is bound to deterministic B0 BM25 with the frozen selected parameters;
- the frozen eligibility policy is applied before ranking;
- benchmark-only labels and gold metadata cannot enter the runtime request;
- the HTTP response reproduces the permanent R3.2 B0 top-50 ranking for the registered Q-001-1 query exactly, including scores;
- repeated identical requests serialize deterministically;
- invalid requests fail schema validation and known backend failures return the stable non-leaking `SEARCH_UNAVAILABLE` response;
- dependency and publication-boundary checks remain clean;
- no Phase 2 or R3.2 scientific artifact is modified.

A successful focused test run is necessary but not sufficient. The final verdict is recorded only after the repository-wide CI gate and manual code/result review both pass.

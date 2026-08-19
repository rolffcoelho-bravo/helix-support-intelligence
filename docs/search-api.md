# Search API Contract

## Endpoint

`POST /v1/search` exposes the frozen `retrieval-selected-v1` configuration over the fictional HelixBank Policy Corpus.

The endpoint is read-only. It does not route tickets, generate answers, perform account actions, or use benchmark labels during retrieval.

## Request

```json
{
  "query": "What should I know about card arrival?",
  "limit": 5
}
```

`query` is required, must contain non-whitespace text, and is limited to 2,000 characters. `limit` is optional, defaults to 10, and must be between 1 and 50. Extra request fields are rejected.

The API intentionally does not accept intent, queue, case type, expected decision, gold citations, allowed resolution types, or relevance judgments. These benchmark fields cannot influence runtime ranking.

## Eligibility

Before the BM25 index is constructed, the Phase 3 evidence policy is applied to the frozen fictional corpus. Eligible records must be:

- `status = current`;
- `permission = public_support`;
- `audience = customer_support`;
- `jurisdiction = fictional-global`;
- valid at the frozen corpus evaluation date `2026-08-19`.

The resulting selected index contains 147 documents. Archived records are excluded before ranking.

## Ranking

The endpoint uses deterministic repository-owned BM25 with the selected Phase 3 parameters:

- `k1 = 1.2`;
- `b = 0.75`;
- Unicode NFKC normalization followed by lowercase;
- token pattern `[A-Za-z0-9]+`;
- no stopword removal;
- no stemming;
- descending score with `document_id` ascending as the exact tie break;
- maximum returned depth of 50.

This is the same B0 configuration selected by the validated R3.2 evidence. The endpoint does not silently substitute B1, B2, or B3.

## Response

```json
{
  "retrieval_version": "retrieval-selected-v1",
  "corpus_version": "helixbank-policy-v1.0.0",
  "query": "What should I know about card arrival?",
  "result_count": 5,
  "results": [
    {
      "document_id": "POLICY-001",
      "rank": 1,
      "score": 0.0,
      "title": "Card Arrival",
      "body": "...",
      "kind": "policy",
      "resolution_type": "provide_policy_guidance"
    }
  ]
}
```

The score shown above is illustrative only. Authoritative scores are produced at runtime from the submitted query and selected corpus.

The response contains no timestamp or random identifier. Identical requests against the same selected configuration serialize deterministically.

## Failure behaviour

Invalid request bodies return HTTP 422 through schema validation.

A known retrieval-backend contract failure returns HTTP 503 with this stable public body:

```json
{
  "detail": {
    "code": "SEARCH_UNAVAILABLE",
    "message": "Search is temporarily unavailable."
  }
}
```

Internal exception details are not returned to the caller.

## Evidence boundary

Search relevance is not equivalent to answerability. In particular, the Phase 3 benchmark contains missing-evidence cases where no eligible relevance-2-or-higher document exists even though partial grade-1 evidence may still rank. The later assistance and decision layers remain responsible for deciding whether retrieved material is sufficient to answer, clarify, or escalate.

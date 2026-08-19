"""Contract and integration tests for the selected Phase 3 search endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from helix_support_intelligence.api.app import create_app
from helix_support_intelligence.api.search import (
    BM25_B,
    BM25_K1,
    EVALUATION_DATE,
    MAX_RESULTS,
    RETRIEVAL_VERSION,
    SearchBackendError,
    SearchHit,
)
from helix_support_intelligence.data.helixbank import CORPUS_VERSION, generate_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_search_endpoint_returns_selected_retrieval_contract() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/search",
        json={"query": "What should I know about card arrival?", "limit": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_version"] == RETRIEVAL_VERSION
    assert payload["corpus_version"] == CORPUS_VERSION
    assert payload["query"] == "What should I know about card arrival?"
    assert payload["result_count"] == 5
    assert [row["rank"] for row in payload["results"]] == [1, 2, 3, 4, 5]
    assert set(payload["results"][0]) == {
        "document_id",
        "rank",
        "score",
        "title",
        "body",
        "kind",
        "resolution_type",
    }


def test_search_response_is_byte_for_byte_deterministic_for_same_request() -> None:
    client = TestClient(create_app())
    request = {"query": "Has the old guidance for card arrival changed?", "limit": 20}

    first = client.post("/v1/search", json=request)
    second = client.post("/v1/search", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.content == second.content


def test_search_excludes_archived_evidence_before_ranking() -> None:
    client = TestClient(create_app())
    bundle = generate_bundle()
    archived_ids = {
        str(record["document_id"]) for record in bundle.documents if record["status"] == "archived"
    }

    response = client.post(
        "/v1/search",
        json={"query": "What should I know about card arrival?", "limit": MAX_RESULTS},
    )

    assert response.status_code == 200
    returned_ids = {row["document_id"] for row in response.json()["results"]}
    assert len(returned_ids) == MAX_RESULTS
    assert returned_ids.isdisjoint(archived_ids)
    assert "FAQ-001" not in returned_ids


def test_zero_score_ties_use_document_id_ascending() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/search",
        json={"query": "zzzxxyyqqq", "limit": 12},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    ids = [row["document_id"] for row in results]
    assert ids == sorted(ids)
    assert all(row["score"] == 0.0 for row in results)


def test_search_rejects_invalid_extra_and_benchmark_metadata_fields() -> None:
    client = TestClient(create_app())

    assert client.post("/v1/search", json={"query": "   "}).status_code == 422
    assert client.post("/v1/search", json={"query": "card", "limit": 0}).status_code == 422
    assert client.post("/v1/search", json={"query": "card", "limit": 51}).status_code == 422
    assert (
        client.post(
            "/v1/search",
            json={"query": "card", "unexpected": "not allowed"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/search",
            json={"query": "card", "intent": "card_arrival"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/search",
            json={"query": "card", "gold_citations": ["POLICY-001"]},
        ).status_code
        == 422
    )


def test_search_backend_failure_has_stable_non_leaking_503_contract() -> None:
    class BrokenSearch:
        def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
            del query, limit
            raise SearchBackendError("private backend diagnostic")

    client = TestClient(create_app(BrokenSearch()))

    response = client.post("/v1/search", json={"query": "card arrival"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "SEARCH_UNAVAILABLE",
            "message": "Search is temporarily unavailable.",
        }
    }
    assert "private backend diagnostic" not in response.text


def test_runtime_constants_match_frozen_selected_and_integration_configuration() -> None:
    selected = json.loads(
        (ROOT / "configs/models/retrieval_selected_v1.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (ROOT / "configs/models/retrieval_ladder_v1.json").read_text(encoding="utf-8")
    )
    integration = json.loads(
        (ROOT / "configs/models/retrieval_integration_r33_v1.json").read_text(encoding="utf-8")
    )

    assert selected["retrieval_version"] == RETRIEVAL_VERSION
    assert selected["selected_candidate"] == "B0"
    assert selected["configuration"]["k1"] == BM25_K1
    assert selected["configuration"]["b"] == BM25_B
    assert selected["configuration"]["retrieve_k"] == MAX_RESULTS
    assert protocol["corpus"]["version"] == CORPUS_VERSION
    assert protocol["corpus"]["evaluation_date"] == EVALUATION_DATE.isoformat()
    assert integration["retrieval_version"] == RETRIEVAL_VERSION
    assert integration["endpoint"] == {
        "method": "POST",
        "path": "/v1/search",
        "read_only": True,
    }
    assert integration["runtime_retrieval"]["candidate"] == "B0"
    assert integration["runtime_retrieval"]["expected_eligible_documents"] == 147
    assert integration["failure_contract"]["known_backend_failure_status"] == 503


def test_http_ranking_reproduces_frozen_r32_b0_evidence() -> None:
    bundle = generate_bundle()
    query = next(row for row in bundle.queries if row["query_id"] == "Q-001-1")
    expected_results: list[dict[str, Any]] | None = None
    rankings_path = ROOT / "benchmarks/retrieval/results/r32/raw/rankings.jsonl"
    with rankings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["candidate"] == "B0" and row["query_id"] == "Q-001-1":
                expected_results = row["results"]
                break

    assert expected_results is not None
    client = TestClient(create_app())
    response = client.post(
        "/v1/search",
        json={"query": str(query["text"]), "limit": MAX_RESULTS},
    )

    assert response.status_code == 200
    actual = response.json()["results"]
    assert [(row["document_id"], row["rank"], row["score"]) for row in actual] == [
        (row["document_id"], row["rank"], row["score"]) for row in expected_results
    ]


def test_openapi_exposes_only_the_bounded_search_method_at_search_path() -> None:
    application = create_app()
    schema = application.openapi()

    assert set(schema["paths"]["/v1/search"]) == {"post"}

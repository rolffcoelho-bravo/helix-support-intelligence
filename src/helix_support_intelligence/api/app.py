"""FastAPI application exposing the bounded Helix search contract."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from helix_support_intelligence.api.search import (
    MAX_RESULTS,
    RETRIEVAL_VERSION,
    SearchBackendError,
    SearchService,
    SelectedRetrievalSearch,
    corpus_version,
)


class SearchRequest(BaseModel):
    """Validated POST /v1/search request."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=MAX_RESULTS)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        """Reject whitespace-only requests while preserving the submitted query text."""
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value


class SearchHitResponse(BaseModel):
    """One serialized evidence result."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    rank: int
    score: float
    title: str
    body: str
    kind: str
    resolution_type: str


class SearchResponse(BaseModel):
    """Deterministic response envelope for selected retrieval."""

    model_config = ConfigDict(extra="forbid")

    retrieval_version: str
    corpus_version: str
    query: str
    result_count: int
    results: list[SearchHitResponse]


def create_app(search_service: SearchService | None = None) -> FastAPI:
    """Create the bounded API with an injectable search backend for contract testing."""
    service = search_service if search_service is not None else SelectedRetrievalSearch()
    application = FastAPI(
        title="Helix Support Intelligence",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    @application.post("/v1/search", response_model=SearchResponse)
    def search(payload: SearchRequest) -> SearchResponse:
        """Retrieve eligible HelixBank evidence using retrieval-selected-v1."""
        try:
            hits = service.search(payload.query, limit=payload.limit)
        except SearchBackendError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "SEARCH_UNAVAILABLE",
                    "message": "Search is temporarily unavailable.",
                },
            ) from exc

        results = [
            SearchHitResponse(
                document_id=hit.document_id,
                rank=hit.rank,
                score=hit.score,
                title=hit.title,
                body=hit.body,
                kind=hit.kind,
                resolution_type=hit.resolution_type,
            )
            for hit in hits
        ]
        return SearchResponse(
            retrieval_version=RETRIEVAL_VERSION,
            corpus_version=corpus_version(),
            query=payload.query,
            result_count=len(results),
            results=results,
        )

    return application


app = create_app()

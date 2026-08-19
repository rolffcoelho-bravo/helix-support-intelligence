"""Runtime search service bound to the validated Phase 3 retrieval selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from helix_support_intelligence.data.helixbank import CORPUS_VERSION, generate_bundle
from helix_support_intelligence.retrieval import (
    BM25Retriever,
    EligibilityPolicy,
    document_from_record,
    filter_eligible_documents,
)

RETRIEVAL_VERSION = "retrieval-selected-v1"
EVALUATION_DATE = date(2026, 8, 19)
EXPECTED_ELIGIBLE_DOCUMENTS = 147
MAX_RESULTS = 50
BM25_K1 = 1.2
BM25_B = 0.75


class SearchBackendError(RuntimeError):
    """Raised when the selected retrieval backend cannot satisfy its runtime contract."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One deterministic search result returned by the selected retriever."""

    document_id: str
    rank: int
    score: float
    title: str
    body: str
    kind: str
    resolution_type: str


class SearchService(Protocol):
    """Typed boundary used by the HTTP layer."""

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        """Return deterministic eligible evidence for one query."""
        ...


def _required_text(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise SearchBackendError(f"corpus record has invalid {key}")
    return value


class SelectedRetrievalSearch:
    """Production-shaped adapter for the validated B0 BM25 configuration."""

    def __init__(self) -> None:
        bundle = generate_bundle()
        documents = tuple(document_from_record(record) for record in bundle.documents)
        policy = EligibilityPolicy(
            evaluation_date=EVALUATION_DATE,
            statuses=frozenset({"current"}),
            permissions=frozenset({"public_support"}),
            audiences=frozenset({"customer_support"}),
            jurisdictions=frozenset({"fictional-global"}),
        )
        eligible_documents = filter_eligible_documents(documents, policy)
        if len(eligible_documents) != EXPECTED_ELIGIBLE_DOCUMENTS:
            raise SearchBackendError(
                "eligible document count does not match retrieval-selected-v1"
            )

        records_by_id: dict[str, dict[str, object]] = {}
        eligible_ids = {document.document_id for document in eligible_documents}
        for record in bundle.documents:
            document_id = _required_text(record, "document_id")
            if document_id in eligible_ids:
                records_by_id[document_id] = record

        if len(records_by_id) != EXPECTED_ELIGIBLE_DOCUMENTS:
            raise SearchBackendError("eligible document metadata is incomplete")

        self._records_by_id = records_by_id
        self._retriever = BM25Retriever(
            eligible_documents,
            k1=BM25_K1,
            b=BM25_B,
        )

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        """Search the frozen eligible corpus with deterministic B0 ranking."""
        if not query.strip():
            raise ValueError("query must contain non-whitespace text")
        if not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

        ranked = self._retriever.search(query, k=limit)
        hits: list[SearchHit] = []
        for item in ranked:
            record = self._records_by_id.get(item.document_id)
            if record is None:
                raise SearchBackendError("ranked document metadata is unavailable")
            hits.append(
                SearchHit(
                    document_id=item.document_id,
                    rank=item.rank,
                    score=item.score,
                    title=_required_text(record, "title"),
                    body=_required_text(record, "body"),
                    kind=_required_text(record, "kind"),
                    resolution_type=_required_text(record, "resolution_type"),
                )
            )
        return tuple(hits)


def corpus_version() -> str:
    """Expose the exact corpus version bound to the selected search service."""
    return CORPUS_VERSION


def result_ids(hits: Sequence[SearchHit]) -> tuple[str, ...]:
    """Return result IDs for deterministic contract assertions."""
    return tuple(hit.document_id for hit in hits)

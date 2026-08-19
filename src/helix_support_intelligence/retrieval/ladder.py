"""Composition of the frozen B0-B3 Phase 3 retrieval ladder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from helix_support_intelligence.retrieval.core import (
    BM25Retriever,
    DenseRetriever,
    Document,
    EmbeddingEncoder,
    PairScorer,
    RankedDocument,
    reciprocal_rank_fusion,
    rerank_top,
)

CandidateId = Literal["B0", "B1", "B2", "B3"]


class RetrievalLadder:
    """Execute the frozen ladder with injected dense and reranking model adapters."""

    def __init__(
        self,
        documents: Sequence[Document],
        encoder: EmbeddingEncoder,
        reranker: PairScorer,
        *,
        retrieve_k: int = 50,
        source_depth: int = 50,
        rrf_k: int = 60,
        rerank_depth: int = 20,
    ) -> None:
        if not documents:
            raise ValueError("retrieval ladder requires at least one eligible document")
        self._documents = {document.document_id: document for document in documents}
        if len(self._documents) != len(documents):
            raise ValueError("retrieval documents must have unique document ids")
        self._bm25 = BM25Retriever(documents, k1=1.2, b=0.75)
        self._dense = DenseRetriever(documents, encoder)
        self._reranker = reranker
        self._retrieve_k = retrieve_k
        self._source_depth = source_depth
        self._rrf_k = rrf_k
        self._rerank_depth = rerank_depth

    def rank_all(self, query: str) -> dict[CandidateId, tuple[RankedDocument, ...]]:
        """Return all B0-B3 rankings while reusing the same B0/B1 source rankings."""
        b0 = self._bm25.search(query, k=self._retrieve_k)
        b1 = self._dense.search(query, k=self._retrieve_k)
        b2 = reciprocal_rank_fusion(
            (b0, b1),
            rrf_k=self._rrf_k,
            source_depth=self._source_depth,
            retrieve_k=self._retrieve_k,
        )
        b3 = rerank_top(
            query,
            b2,
            self._documents,
            self._reranker,
            depth=self._rerank_depth,
            retrieve_k=self._retrieve_k,
        )
        rankings: dict[CandidateId, tuple[RankedDocument, ...]] = {
            "B0": b0,
            "B1": b1,
            "B2": b2,
            "B3": b3,
        }
        return rankings

    def rank(self, candidate: CandidateId, query: str) -> tuple[RankedDocument, ...]:
        """Return one requested candidate ranking without changing ladder semantics."""
        if candidate == "B0":
            return self._bm25.search(query, k=self._retrieve_k)
        if candidate == "B1":
            return self._dense.search(query, k=self._retrieve_k)

        b0 = self._bm25.search(query, k=self._retrieve_k)
        b1 = self._dense.search(query, k=self._retrieve_k)
        b2 = reciprocal_rank_fusion(
            (b0, b1),
            rrf_k=self._rrf_k,
            source_depth=self._source_depth,
            retrieve_k=self._retrieve_k,
        )
        if candidate == "B2":
            return b2
        return rerank_top(
            query,
            b2,
            self._documents,
            self._reranker,
            depth=self._rerank_depth,
            retrieve_k=self._retrieve_k,
        )

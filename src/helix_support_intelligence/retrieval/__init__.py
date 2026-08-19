"""Information-retrieval primitives for Helix Phase 3."""

from helix_support_intelligence.retrieval.benchmark import (
    RetrievalBenchmarkSpec,
    RetrievalQuery,
    build_manifest,
    build_qrels,
    eligible_documents,
    select_queries,
)

__all__ = [
    "RetrievalBenchmarkSpec",
    "RetrievalQuery",
    "build_manifest",
    "build_qrels",
    "eligible_documents",
    "select_queries",
]

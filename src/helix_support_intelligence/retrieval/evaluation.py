"""Metric aggregation for the frozen Phase 3 retrieval protocol."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from helix_support_intelligence.retrieval.core import (
    RankedDocument,
    macro_average,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Registered retrieval metrics for one query and one candidate."""

    ndcg_at_10: float
    mrr_at_10: float
    recall_at_20: float | None
    recall_at_50: float | None


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Macro retrieval metrics with recall applicability counts."""

    ndcg_at_10: float
    mrr_at_10: float
    recall_at_20: float
    recall_at_20_queries: int
    recall_at_50: float
    recall_at_50_queries: int
    query_count: int


def evaluate_ranking(
    ranking: Sequence[RankedDocument], qrels: Mapping[str, int]
) -> QueryMetrics:
    """Evaluate a ranking using the registered Phase 3 endpoints."""
    return QueryMetrics(
        ndcg_at_10=ndcg_at_k(ranking, qrels, k=10),
        mrr_at_10=mrr_at_k(ranking, qrels, k=10, relevance_threshold=2),
        recall_at_20=recall_at_k(ranking, qrels, k=20, relevance_threshold=2),
        recall_at_50=recall_at_k(ranking, qrels, k=50, relevance_threshold=2),
    )


def aggregate_metrics(rows: Sequence[QueryMetrics]) -> AggregateMetrics:
    """Macro-average registered metrics over a non-empty query collection."""
    if not rows:
        raise ValueError("aggregate metrics require at least one query")
    recall20, count20 = macro_average([row.recall_at_20 for row in rows])
    recall50, count50 = macro_average([row.recall_at_50 for row in rows])
    return AggregateMetrics(
        ndcg_at_10=statistics.fmean(row.ndcg_at_10 for row in rows),
        mrr_at_10=statistics.fmean(row.mrr_at_10 for row in rows),
        recall_at_20=recall20,
        recall_at_20_queries=count20,
        recall_at_50=recall50,
        recall_at_50_queries=count50,
        query_count=len(rows),
    )

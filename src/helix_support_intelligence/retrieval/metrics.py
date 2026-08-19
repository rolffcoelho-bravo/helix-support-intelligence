"""Frozen retrieval metrics for Helix Phase 3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Per-query retrieval metrics under the Phase 3 contract."""

    ndcg_at_10: float
    mrr_at_10: float
    recall_at_20: float
    recall_at_50: float
    success_at_1: float
    citation_eligible_recall_at_20: float


def _dcg(relevances: Sequence[int], k: int) -> float:
    total = 0.0
    for rank, relevance in enumerate(relevances[:k], start=1):
        if relevance <= 0:
            continue
        gain = (2.0**relevance) - 1.0
        total += gain / math.log2(rank + 1.0)
    return total


def evaluate_query(
    ranked_document_ids: Sequence[str],
    qrels: Mapping[str, int],
) -> QueryMetrics:
    """Evaluate one ranked list using the frozen Phase 3 semantics."""

    if not qrels:
        raise ValueError("retrieval query must have at least one judged document")
    if any(value < 0 for value in qrels.values()):
        raise ValueError("retrieval relevance grades must be non-negative")

    ranked_relevance = [qrels.get(document_id, 0) for document_id in ranked_document_ids]
    ideal_relevance = sorted(qrels.values(), reverse=True)
    ideal_dcg = _dcg(ideal_relevance, 10)
    ndcg = _dcg(ranked_relevance, 10) / ideal_dcg if ideal_dcg > 0 else 0.0

    reciprocal_rank = 0.0
    for rank, relevance in enumerate(ranked_relevance[:10], start=1):
        if relevance > 0:
            reciprocal_rank = 1.0 / rank
            break

    relevant = {document_id for document_id, relevance in qrels.items() if relevance > 0}
    if not relevant:
        raise ValueError("retrieval query must have at least one positive relevance judgment")

    def recall_at(k: int) -> float:
        retrieved = set(ranked_document_ids[:k])
        return len(relevant & retrieved) / len(relevant)

    success_at_1 = float(bool(ranked_relevance) and ranked_relevance[0] > 0)
    citation_eligible = {
        document_id for document_id, relevance in qrels.items() if relevance >= 3
    }
    if not citation_eligible:
        raise ValueError("retrieval query must identify a governing grade-3 policy")
    citation_recall = float(bool(citation_eligible & set(ranked_document_ids[:20])))

    return QueryMetrics(
        ndcg_at_10=ndcg,
        mrr_at_10=reciprocal_rank,
        recall_at_20=recall_at(20),
        recall_at_50=recall_at(50),
        success_at_1=success_at_1,
        citation_eligible_recall_at_20=citation_recall,
    )


def mean_metrics(metrics: Sequence[QueryMetrics]) -> dict[str, float]:
    """Macro-average the registered retrieval metrics over queries."""

    if not metrics:
        raise ValueError("cannot aggregate an empty retrieval metric sequence")
    fields = (
        "ndcg_at_10",
        "mrr_at_10",
        "recall_at_20",
        "recall_at_50",
        "success_at_1",
        "citation_eligible_recall_at_20",
    )
    return {
        field: sum(getattr(item, field) for item in metrics) / len(metrics) for field in fields
    }

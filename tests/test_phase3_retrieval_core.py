"""No-result tests for the Phase 3 retrieval implementation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from helix_support_intelligence.data.helixbank import generate_bundle
from helix_support_intelligence.retrieval.core import (
    BM25Retriever,
    Document,
    EligibilityPolicy,
    RankedDocument,
    candidate_earns_complexity,
    document_from_record,
    filter_eligible_documents,
    paired_bootstrap_difference,
    reciprocal_rank_fusion,
    rerank_top,
    summarize_latency,
    tokenize,
)
from helix_support_intelligence.retrieval.evaluation import aggregate_metrics, evaluate_ranking
from helix_support_intelligence.retrieval.ladder import RetrievalLadder


class KeywordEncoder:
    """Tiny deterministic encoder used only for unit fixtures."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            alpha = 1.0 if "alpha" in lowered else 0.0
            beta = 1.0 if "beta" in lowered else 0.0
            if alpha == 0.0 and beta == 0.0:
                vectors.append([0.5, 0.5])
            else:
                vectors.append([alpha, beta])
        return vectors


class BetaFirstScorer:
    """Tiny deterministic pair scorer used only for unit fixtures."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return [1.0 if "beta" in document.lower() else 0.0 for _, document in pairs]


def _document(document_id: str, title: str, body: str) -> Document:
    return Document(
        document_id=document_id,
        title=title,
        body=body,
        status="current",
        valid_from=date(2026, 2, 1),
        valid_to=None,
        permission="public_support",
        audience="customer_support",
        jurisdiction="fictional-global",
    )


def test_real_corpus_document_filter_is_pre_ranking_and_query_independent() -> None:
    bundle = generate_bundle()
    documents = tuple(document_from_record(record) for record in bundle.documents)
    policy = EligibilityPolicy(
        evaluation_date=date(2026, 8, 19),
        statuses=frozenset({"current"}),
        permissions=frozenset({"public_support"}),
        audiences=frozenset({"customer_support"}),
        jurisdictions=frozenset({"fictional-global"}),
    )

    eligible = filter_eligible_documents(documents, policy)

    assert len(eligible) == 147
    assert all(document.status == "current" for document in eligible)
    assert sum(document.conflict_fixture for document in eligible) == 7
    assert sum(document.untrusted_content_fixture for document in eligible) == 5


def test_bm25_tokenization_and_tie_breaking_are_deterministic() -> None:
    documents = (
        _document("DOC-002", "Alpha", "same text"),
        _document("DOC-001", "Alpha", "same text"),
    )
    retriever = BM25Retriever(documents)

    assert tokenize("Café ALPHA-2") == ("caf", "alpha", "2")
    ranking = retriever.search("not-present", k=2)
    assert [item.document_id for item in ranking] == ["DOC-001", "DOC-002"]


def test_rrf_uses_rank_only_and_document_id_for_ties() -> None:
    left = (
        RankedDocument("DOC-002", 1, 99.0),
        RankedDocument("DOC-001", 2, 1.0),
    )
    right = (
        RankedDocument("DOC-001", 1, -5.0),
        RankedDocument("DOC-002", 2, 1000.0),
    )

    fused = reciprocal_rank_fusion((left, right), rrf_k=60, source_depth=50, retrieve_k=2)

    assert [item.document_id for item in fused] == ["DOC-001", "DOC-002"]
    assert fused[0].score == pytest.approx(fused[1].score)


def test_reranker_changes_only_the_declared_head() -> None:
    documents = {
        "DOC-001": _document("DOC-001", "Alpha", "alpha evidence"),
        "DOC-002": _document("DOC-002", "Beta", "beta evidence"),
        "DOC-003": _document("DOC-003", "Tail", "tail evidence"),
    }
    base = (
        RankedDocument("DOC-001", 1, 3.0),
        RankedDocument("DOC-002", 2, 2.0),
        RankedDocument("DOC-003", 3, 1.0),
    )

    reranked = rerank_top(
        "query",
        base,
        documents,
        BetaFirstScorer(),
        depth=2,
        retrieve_k=3,
    )

    assert [item.document_id for item in reranked] == ["DOC-002", "DOC-001", "DOC-003"]
    assert reranked[2].score == 1.0


def test_full_b0_b3_ladder_executes_only_on_tiny_fixture() -> None:
    documents = (
        _document("DOC-001", "Alpha", "alpha policy"),
        _document("DOC-002", "Beta", "beta policy"),
        _document("DOC-003", "Mixed", "alpha beta reference"),
    )
    ladder = RetrievalLadder(
        documents,
        KeywordEncoder(),
        BetaFirstScorer(),
        retrieve_k=3,
        source_depth=3,
        rerank_depth=2,
    )

    rankings = ladder.rank_all("alpha")

    assert list(rankings) == ["B0", "B1", "B2", "B3"]
    assert all(len(ranking) == 3 for ranking in rankings.values())
    assert rankings["B0"][0].document_id in {"DOC-001", "DOC-003"}
    assert rankings["B1"][0].document_id == "DOC-001"


def test_registered_metrics_are_computed_without_metric_substitution() -> None:
    ranking = (
        RankedDocument("DOC-A", 1, 1.0),
        RankedDocument("DOC-B", 2, 0.5),
    )
    qrels = {"DOC-A": 3, "DOC-B": 2}
    row = evaluate_ranking(ranking, qrels)
    aggregate = aggregate_metrics((row,))

    assert row.ndcg_at_10 == pytest.approx(1.0)
    assert row.mrr_at_10 == pytest.approx(1.0)
    assert row.recall_at_20 == pytest.approx(1.0)
    assert row.recall_at_50 == pytest.approx(1.0)
    assert aggregate.query_count == 1
    assert aggregate.recall_at_20_queries == 1
    assert aggregate.recall_at_50_queries == 1


def test_recall_marks_queries_without_directly_relevant_evidence_inapplicable() -> None:
    ranking = (RankedDocument("DOC-A", 1, 1.0),)
    row = evaluate_ranking(ranking, {"DOC-A": 1})
    aggregate = aggregate_metrics((row,))

    assert row.mrr_at_10 == 0.0
    assert row.recall_at_20 is None
    assert row.recall_at_50 is None
    assert aggregate.recall_at_20 == 0.0
    assert aggregate.recall_at_20_queries == 0


def test_bootstrap_latency_and_complexity_gate_are_deterministic() -> None:
    interval = paired_bootstrap_difference(
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        replicates=100,
        seed=20260819,
    )
    latency = summarize_latency((10.0, 20.0, 30.0, 40.0))

    assert interval.point_estimate == 1.0
    assert interval.lower == 1.0
    assert interval.upper == 1.0
    assert latency.mean_ms == 25.0
    assert candidate_earns_complexity(
        delta_ndcg_at_10=0.02,
        ndcg_ci_lower=0.005,
        delta_mrr_at_10=0.0,
        candidate_p95_ms=200.0,
        latency_budget_ms=250.0,
    )
    assert not candidate_earns_complexity(
        delta_ndcg_at_10=0.009,
        ndcg_ci_lower=0.005,
        delta_mrr_at_10=0.0,
        candidate_p95_ms=200.0,
        latency_budget_ms=250.0,
    )

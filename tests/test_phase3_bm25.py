from __future__ import annotations

import math

import pytest

from helix_support_intelligence.retrieval.bm25 import BM25Index, tokenize
from helix_support_intelligence.retrieval.metrics import evaluate_query, mean_metrics


def test_b0_tokenizer_matches_frozen_nfkc_casefold_contract() -> None:
    assert tokenize("Café_CARD １２3") == ("café", "card", "123")


def test_b0_bm25_uses_positive_rsj_idf_and_stable_document_ties() -> None:
    documents = [
        {"document_id": "DOC-B", "title": "alpha", "body": "beta"},
        {"document_id": "DOC-A", "title": "alpha", "body": "beta"},
        {"document_id": "DOC-C", "title": "gamma", "body": "delta"},
    ]
    index = BM25Index.build(documents, k1=1.2, b=0.75)
    ranked = index.score("alpha")

    assert [item.document_id for item in ranked] == ["DOC-A", "DOC-B", "DOC-C"]
    expected_idf = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))
    assert ranked[0].score == pytest.approx(expected_idf)
    assert ranked[1].score == pytest.approx(expected_idf)
    assert ranked[2].score == 0.0


def test_b0_metrics_follow_registered_graded_and_binary_semantics() -> None:
    ranking = ["OTHER", "FAQ", "POLICY", "NOISE"]
    qrels = {"POLICY": 3, "FAQ": 2}
    metrics = evaluate_query(ranking, qrels)

    ideal = 7.0 / math.log2(2) + 3.0 / math.log2(3)
    observed = 3.0 / math.log2(3) + 7.0 / math.log2(4)
    assert metrics.ndcg_at_10 == pytest.approx(observed / ideal)
    assert metrics.mrr_at_10 == 0.5
    assert metrics.recall_at_20 == 1.0
    assert metrics.recall_at_50 == 1.0
    assert metrics.success_at_1 == 0.0
    assert metrics.citation_eligible_recall_at_20 == 1.0


def test_b0_metric_macro_aggregation_is_query_weighted() -> None:
    first = evaluate_query(["POLICY", "FAQ"], {"POLICY": 3, "FAQ": 2})
    second = evaluate_query(["NOISE", "POLICY"], {"POLICY": 3})
    aggregate = mean_metrics([first, second])

    assert aggregate["success_at_1"] == 0.5
    assert aggregate["citation_eligible_recall_at_20"] == 1.0
    assert aggregate["mrr_at_10"] == 0.75

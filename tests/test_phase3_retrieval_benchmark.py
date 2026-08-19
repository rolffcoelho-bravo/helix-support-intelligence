from __future__ import annotations

from pathlib import Path

from helix_support_intelligence.data.banking77 import BankingExample
from helix_support_intelligence.data.helixbank import INTENTS, generate_bundle
from helix_support_intelligence.retrieval.benchmark import (
    RetrievalBenchmarkSpec,
    build_manifest,
    build_qrels,
    eligible_documents,
    select_queries,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "retrieval" / "phase3_benchmark_v1.json"
MATERIALIZER = ROOT / "scripts" / "materialize_retrieval_benchmark.py"


def _synthetic_fit_train(rows_per_intent: int = 40) -> list[BankingExample]:
    rows: list[BankingExample] = []
    source_index = 0
    for intent in INTENTS:
        for position in range(rows_per_intent):
            rows.append(
                BankingExample(
                    source_split="train",
                    source_index=source_index,
                    text=f"natural support utterance {position} for {intent}",
                    intent=intent,
                )
            )
            source_index += 1
    return rows


def test_phase3_benchmark_selection_is_deterministic_and_disjoint() -> None:
    spec = RetrievalBenchmarkSpec.from_json(CONFIG)
    fit_train = _synthetic_fit_train()

    first_dev, first_confirm = select_queries(fit_train, spec, "synthetic-revision")
    second_dev, second_confirm = select_queries(fit_train, spec, "synthetic-revision")

    assert first_dev == second_dev
    assert first_confirm == second_confirm
    assert len(first_dev) == 1540
    assert len(first_confirm) == 770
    assert {item.query_id for item in first_dev}.isdisjoint(
        {item.query_id for item in first_confirm}
    )
    assert {item.intent for item in first_dev} == set(INTENTS)
    assert {item.intent for item in first_confirm} == set(INTENTS)


def test_phase3_candidate_filter_excludes_archived_documents_before_scoring() -> None:
    spec = RetrievalBenchmarkSpec.from_json(CONFIG)
    documents = eligible_documents(generate_bundle().documents, spec)

    assert len(documents) == 147
    assert all(item["status"] == "current" for item in documents)
    assert all(item["permission"] == "public_support" for item in documents)
    assert all(item["corpus_version"] == "helixbank-policy-v1.0.0" for item in documents)


def test_phase3_qrels_and_manifest_match_frozen_semantics() -> None:
    spec = RetrievalBenchmarkSpec.from_json(CONFIG)
    development, confirmatory = select_queries(_synthetic_fit_train(), spec, "synthetic-revision")
    documents = eligible_documents(generate_bundle().documents, spec)
    development_qrels = build_qrels(development, documents, spec)
    confirmatory_qrels = build_qrels(confirmatory, documents, spec)
    manifest = build_manifest(
        development,
        confirmatory,
        development_qrels,
        confirmatory_qrels,
        documents,
        spec,
    )

    assert len(development_qrels) == 2940
    assert len(confirmatory_qrels) == 1470
    assert manifest["candidate_documents"] == 147
    assert manifest["development_queries"] == 1540
    assert manifest["confirmatory_queries"] == 770
    assert manifest["development_qrels"] == 2940
    assert manifest["confirmatory_qrels"] == 1470
    assert manifest["confirmatory_content_logged"] is False


def test_phase3_materializer_cannot_access_official_test_url() -> None:
    source = MATERIALIZER.read_text(encoding="utf-8")

    assert "_download_train_only(banking.train_url" in source
    assert "banking.test_url" not in source
    assert "test.csv" not in source
    assert 'manifest["official_test_accessed"] = False' in source

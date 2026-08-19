"""Phase 3 retrieval-protocol invariants before any B0-B3 scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.helixbank import manifest

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_PROTOCOL = ROOT / "configs" / "models" / "retrieval_ladder_v1.json"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _load() -> dict[str, Any]:
    payload = json.loads(RETRIEVAL_PROTOCOL.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_phase3_retrieval_corpus_is_exactly_the_frozen_phase1_corpus() -> None:
    payload = _load()
    corpus = cast(dict[str, Any], payload["corpus"])
    current_manifest = manifest()

    assert corpus["version"] == current_manifest["corpus_version"]
    assert corpus["generator_version"] == current_manifest["generator_version"]
    assert corpus["counts"] == current_manifest["counts"]
    assert corpus["sha256"] == current_manifest["sha256"]


def test_phase3_retrieval_ladder_is_bounded_and_pinned() -> None:
    payload = _load()
    ladder = cast(list[dict[str, Any]], payload["ladder"])
    by_id = {str(candidate["id"]): candidate for candidate in ladder}

    assert list(by_id) == ["B0", "B1", "B2", "B3"]
    assert by_id["B0"]["parameters"] == {"k1": 1.2, "b": 0.75}
    assert by_id["B0"]["retrieve_k"] == 50

    b1_model = cast(dict[str, Any], by_id["B1"]["model"])
    assert b1_model["id"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert FULL_SHA_RE.fullmatch(str(b1_model["revision"])) is not None
    assert by_id["B1"]["retrieve_k"] == 50

    b2_rrf = cast(dict[str, Any], by_id["B2"]["rrf"])
    assert by_id["B2"]["inputs"] == ["B0", "B1"]
    assert by_id["B2"]["source_depth"] == 50
    assert b2_rrf["k"] == 60
    assert by_id["B2"]["retrieve_k"] == 50

    b3_model = cast(dict[str, Any], by_id["B3"]["model"])
    assert by_id["B3"]["input"] == "B2"
    assert by_id["B3"]["rerank_depth"] == 20
    assert by_id["B3"]["retrieve_k"] == 50
    assert b3_model["id"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert FULL_SHA_RE.fullmatch(str(b3_model["revision"])) is not None
    assert "ranks 21-50 unchanged" in str(by_id["B3"]["tail_policy"])


def test_phase3_filters_cannot_use_query_labels_or_gold_data() -> None:
    payload = _load()
    corpus = cast(dict[str, Any], payload["corpus"])
    filters = cast(dict[str, Any], corpus["eligibility_filters"])
    forbidden = set(cast(list[str], filters["explicitly_forbidden_query_derived_filters"]))

    assert filters["apply_before_ranking"] is True
    assert filters["status"] == ["current"]
    assert filters["permission"] == ["public_support"]
    assert filters["audience"] == ["customer_support"]
    assert filters["jurisdiction"] == ["fictional-global"]
    assert forbidden == {
        "intent",
        "queue",
        "case_type",
        "expected_decision",
        "gold_citations",
        "allowed_resolution_types",
        "relevance_judgments",
    }


def test_phase3_metrics_hypotheses_and_inference_are_predeclared() -> None:
    payload = _load()
    metrics = cast(dict[str, Any], payload["metrics"])
    hypotheses = cast(dict[str, dict[str, Any]], payload["registered_hypotheses"])
    inference = cast(dict[str, Any], payload["inference"])

    assert metrics["primary"] == "nDCG@10"
    assert metrics["registered_secondary"] == ["MRR@10", "Recall@20", "Recall@50"]
    assert hypotheses["H1"]["comparison"] == "B2 - B0"
    assert hypotheses["H1"]["endpoint"] == "nDCG@10"
    assert hypotheses["H2"]["comparison"] == "B3 - B2"
    assert hypotheses["H2"]["endpoint"] == "MRR@10"
    assert inference["method"] == "paired nonparametric bootstrap"
    assert inference["replicates"] == 5000
    assert inference["seed"] == 20260819


def test_phase3_latency_and_complexity_rules_are_locked_before_scoring() -> None:
    payload = _load()
    latency = cast(dict[str, Any], payload["latency"])
    adoption = cast(dict[str, Any], payload["complexity_adoption"])
    candidate_rule = cast(dict[str, Any], adoption["candidate_rule"])
    guard = cast(dict[str, Any], payload["execution_guard"])

    assert latency["device"] == "CPU only"
    assert latency["warmup_queries_per_candidate"] == 30
    assert latency["timed_passes"] == 5
    assert latency["selection_budgets_ms"] == {"B0": 100, "B1": 250, "B2": 250, "B3": 500}
    assert adoption["starting_winner"] == "B0"
    assert adoption["evaluation_order"] == ["B1", "B2", "B3"]
    assert candidate_rule["minimum_delta_ndcg_at_10"] == 0.01
    assert candidate_rule["require_paired_95_ci_lower_bound_above_zero"] is True
    assert candidate_rule["require_mrr_at_10_delta_not_below"] == -0.005
    assert adoption["no_post_score_parameter_tuning"] is True
    assert guard["results_opened"] is False
    assert guard["scoring_allowed_only_after_protocol_merge"] is True

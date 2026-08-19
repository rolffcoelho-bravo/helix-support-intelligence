"""R3.1 implementation invariants that must hold before frozen-query scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "configs" / "models" / "retrieval_implementation_v1.json"


def _load() -> dict[str, Any]:
    payload = json.loads(IMPLEMENTATION.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_r31_is_bound_to_the_merged_r30_protocol_without_results() -> None:
    payload = _load()

    assert payload["implementation_id"] == "phase3-retrieval-r3.1-v1"
    assert payload["protocol_id"] == "phase3-retrieval-r3.0-v1"
    assert payload["status"] == "implemented_preflight_no_results"
    assert payload["frozen_query_scores_computed"] == 0


def test_r31_implementation_details_are_fixed_before_scoring() -> None:
    payload = _load()
    b0 = cast(dict[str, Any], payload["B0"])
    b1 = cast(dict[str, Any], payload["B1"])
    b2 = cast(dict[str, Any], payload["B2"])
    b3 = cast(dict[str, Any], payload["B3"])
    evaluation = cast(dict[str, Any], payload["evaluation"])

    assert b0["idf_formula"] == "ln(1 + (N - df + 0.5) / (df + 0.5))"
    assert b1["revision"] == "c315f904dfc467d8b9c40ab4ed50b3a8d0866c15"
    assert b1["similarity"] == "dot product over normalized vectors, equivalent to cosine"
    assert b2["rrf_k"] == 60
    assert b2["source_depth"] == 50
    assert b2["retrieve_k"] == 50
    assert b3["revision"] == "c5f2b386de279a97c53a702dd5189d1c407160dc"
    assert b3["rerank_depth"] == 20
    assert evaluation["mrr_and_recall_relevance_threshold"] == 2
    assert evaluation["bootstrap_percentile_interpolation"].startswith("linear interpolation")


def test_r31_preflight_keeps_model_execution_and_frozen_query_ranking_closed() -> None:
    payload = _load()
    execution = cast(dict[str, Any], payload["execution"])
    latency = cast(dict[str, Any], payload["latency"])

    assert execution["unit_fixture_only_in_r3.1"] is True
    assert execution["benchmark_query_ranking_allowed_in_r3.1"] is False
    assert "R3.2" in str(execution["concrete_model_runtime_binding"])
    assert "R3.2" in str(latency["measurement_execution"])

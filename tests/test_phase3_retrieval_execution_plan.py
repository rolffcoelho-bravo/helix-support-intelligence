"""Guard the registered R3.2 retrieval execution before result opening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/models/retrieval_ladder_v1.json"
IMPLEMENTATION = ROOT / "configs/models/retrieval_implementation_v1.json"
EXECUTION = ROOT / "configs/models/retrieval_execution_r32_v1.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_r32_is_bound_to_frozen_protocol_and_implementation() -> None:
    protocol = _load(PROTOCOL)
    implementation = _load(IMPLEMENTATION)
    execution = _load(EXECUTION)

    assert execution["execution_id"] == "phase3-retrieval-r3.2-v1"
    assert execution["protocol_id"] == protocol["protocol_id"]
    assert execution["implementation_id"] == implementation["implementation_id"]
    assert execution["results_opened"] is False
    assert execution["base_main_sha"] == "c2d884410fb5f703699dbf07dbde5968bdc71d37"


def test_r32_runtime_and_model_pins_are_exact() -> None:
    protocol = _load(PROTOCOL)
    execution = _load(EXECUTION)
    ladder = {row["id"]: row for row in protocol["ladder"]}

    assert execution["runtime"]["device"] == "cpu"
    assert execution["runtime"]["torch_num_threads"] == 1
    assert execution["runtime"]["torch_num_interop_threads"] == 1
    assert execution["models"]["B1"]["model_id"] == ladder["B1"]["model"]["id"]
    assert execution["models"]["B1"]["revision"] == ladder["B1"]["model"]["revision"]
    assert execution["models"]["B3"]["model_id"] == ladder["B3"]["model"]["id"]
    assert execution["models"]["B3"]["revision"] == ladder["B3"]["model"]["revision"]
    assert execution["models"]["B1"]["batch_size"] == 32
    assert execution["models"]["B3"]["batch_size"] == 32


def test_r32_query_latency_and_inference_rules_match_registration() -> None:
    execution = _load(EXECUTION)

    assert execution["execution_partition"] == {
        "corpus": "helixbank-policy-v1.0.0",
        "queries": 308,
        "eligible_documents_expected": 147,
        "query_order": "query_id ascending",
        "candidate_order": ["B0", "B1", "B2", "B3"],
    }
    quality = execution["quality_execution"]
    assert quality["retrieve_k"] == 50
    assert quality["store_top_k"] == 50
    assert quality["paired_bootstrap_replicates"] == 5000
    assert quality["paired_bootstrap_seed"] == 20260819

    latency = execution["latency_execution"]
    assert latency["warmup_queries_per_candidate"] == 30
    assert latency["timed_passes"] == 5
    assert latency["budgets_ms"] == {"B0": 100, "B1": 250, "B2": 250, "B3": 500}


def test_r32_requires_post_execution_reconstruction_before_closure() -> None:
    execution = _load(EXECUTION)
    evidence = execution["evidence"]
    immutability = execution["immutability"]

    assert evidence["independent_post_execution_reconstruction_required"] is True
    assert evidence["run_is_provisional_until_post_audit_passes"] is True
    assert "post_audit.json" in evidence["output_files"]
    assert "checksums.sha256" in evidence["output_files"]
    assert immutability["no_result_motivated_changes"] is True
    assert immutability["any_scientific_change_after_first_ranking_requires_new_execution_id"] is True

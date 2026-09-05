"""Static tests for the A4.5d zero-result validation registration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45d_v1.json"
A45A_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json"
M6_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm6_v1.json"
M6_CLOSURE = ROOT / "benchmarks" / "assistance" / "a45bm6_closure_v1.json"
CORE = ROOT / "benchmarks" / "assistance" / "tpag_core_a45bm6.py"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _core() -> Any:
    spec = importlib.util.spec_from_file_location("tpag_core_a45bm6_a45d_test", CORE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a45d_freezes_exact_m6_implementation_and_threshold() -> None:
    config = _json(CONFIG)
    m6 = _json(M6_CONFIG)
    closure = _json(M6_CLOSURE)
    assert config["source_main_sha"] == "40d6bdb417e798a7c0ead7709bdcec5d8241a989"
    assert closure["scientific_status"] == "PASSED_TPAG_CALIBRATION_READINESS_PARAMETERS_FROZEN"
    assert (
        config["authoritative_implementation"]["implementation_id"]
        == (m6["authoritative_implementation"]["implementation_id"])
    )
    assert (
        config["authoritative_implementation"]["model"]
        == (m6["authoritative_implementation"]["semantic_model"])
    )
    assert config["authoritative_implementation"]["runtime"] == m6["runtime"]
    assert config["authoritative_implementation"]["selected_alignment_confidence_min"] == 0.6


def test_a45d_preserves_all_original_a45a_validity_gates() -> None:
    config = _json(CONFIG)
    a45a = _json(A45A_CONFIG)
    fresh = config["fresh_validation_contract"]
    assert fresh["component_requirements"] == a45a["registered_component_requirements"]
    assert fresh["claim_requirements"] == a45a["registered_claim_requirements"]
    assert len(fresh["component_requirements"]) == 19
    assert len(fresh["claim_requirements"]) == 10
    assert fresh["requirement_count"] == 29
    assert fresh["validation_pairs_sha256"] == (
        "5f6f0294230de5da3af8baaee2403c9497bd42308c96f9d1041f4f88667d1da7"
    )
    assert fresh["validation_claims_sha256"] == (
        "116040d37035e4a43a3bee17ea2d29fe87d85c6148adade770e8c224456e43d6"
    )


def test_frozen_m6_parser_does_not_accept_registered_a45a_text_contract() -> None:
    core = _core()
    queue_probe = core.parse_frame(
        "Orchid case 999 requests are handled by the access_review queue.", {}
    )
    requirement_probe = core.parse_frame(
        "Orchid case 999 review requires transaction reference.", {}
    )
    for probe in (queue_probe, requirement_probe):
        assert probe["entity_or_subject"] is None
        assert probe["predicate_or_event"] is None
        assert probe["target_slot_identity"] is None


def test_a45d_fails_closed_before_validation_or_confirmatory_access() -> None:
    config = _json(CONFIG)
    assert config["status"] == "REGISTERED_ZERO_RESULT_VALIDATION_BLOCKED_INPUT_CONTRACT_MISMATCH"
    assert config["scientific_result_exposed"] is False
    audit = config["pre_execution_contract_audit"]
    assert audit["direct_application_ready"] is False
    assert audit["existing_registered_adapter_found"] is False
    assert audit["adapter_or_parser_extension_would_change_scientific_implementation"] is True
    assert audit["validation_execution_authorized"] is False
    assert all(int(value) == 0 for value in config["execution_scope"].values())
    assert config["fresh_validation_contract"]["validation_records_materialized"] == 0
    assert config["fresh_validation_contract"]["validation_records_inspected"] == 0
    assert config["fresh_validation_contract"]["validation_records_scored"] == 0
    assert config["governance"]["a45a_validation_remains_sealed"] is True
    assert config["governance"]["confirmatory_partition_remains_sealed"] is True
    assert config["governance"]["a45c_repurposed"] is False
    assert config["next_checkpoint"]["authorized"] is False

"""Registration tests for A4.4e post-validation methodology selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a44e_is_no_inference_and_model_unbound() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a44e_v1.json")
    assert config["checkpoint"] == "A4.4e"
    assert config["methodology_decision"]["short_name"] == "AERF"
    assert config["methodology_decision"]["status"] == "ARCHITECTURE_SELECTED_MODEL_UNBOUND"
    assert all(int(value) == 0 for value in config["scope"].values())


def test_a44e_rejects_post_validation_rescue() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a44e_v1.json")
    rejected = config["rejected_paths"]
    assert rejected["another_native_three_way_mnli_argmax_swap"] is True
    assert rejected["post_validation_threshold_rescue"] is True
    assert rejected["post_validation_temperature_refit"] is True
    assert rejected["class_specific_threshold_tuning_on_a44d_validation"] is True
    assert rejected["reuse_a44a_validation_as_independent_hard_validity_evidence"] is True
    assert rejected["open_confirmatory_partition_before_new_validity_gate"] is True


def test_a44e_keeps_confirmatory_sealed_and_next_gate_unapproved() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a44e_v1.json")
    assert config["scope"]["confirmatory_query_rows_authorized"] == 0
    assert config["scope"]["confirmatory_query_records_inspected"] == 0
    future = config["future_binding_constraints"]
    assert future[
        "confirmatory_partition_must_remain_unopened_until_replacement_method_clears_new_registered_validity"
    ] is True
    assert config["next_checkpoint"]["checkpoint"] == "A4.5a"
    assert config["next_checkpoint"]["authorized_by_a44e"] is False
    assert config["next_checkpoint"]["requires_separate_approval"] is True


def test_a44e_preflight_has_no_model_or_case_materialization_surface() -> None:
    source = (ROOT / "scripts" / "preflight_phase4_a44e.py").read_text(encoding="utf-8")
    forbidden_tokens = (
        "AutoModel",
        "AutoTokenizer",
        "snapshot_download",
        "generate_validation_cases",
        "generate_cases(",
        "transformers",
        "torch",
    )
    assert all(token not in source for token in forbidden_tokens)


def test_a44e_failure_geometry_is_descriptive_only() -> None:
    result = _load(
        ROOT
        / "benchmarks"
        / "assistance"
        / "results"
        / "a44e_methodology_decision_v1"
        / "failure_geometry.json"
    )
    assert result["scope"]["post_validation_descriptive_only"] is True
    assert result["scope"]["new_semantic_inference"] == 0
    assert result["scope"]["threshold_search"] == 0
    assert result["scope"]["temperature_refit"] == 0
    assert result["scope"]["confirmatory_queries_scored"] == 0
    assert result["unknown_failure_subtypes"]["cross_document_non_evidence"][
        "gold_unknown_pairs"
    ] == 100
    assert result["unknown_failure_subtypes"]["same_document_insufficient_evidence"][
        "gold_unknown_pairs"
    ] == 20

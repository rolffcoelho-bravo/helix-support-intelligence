"""Registration tests for Phase 4 A4.5a AERF validity protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _generator() -> Any:
    path = ROOT / "benchmarks" / "assistance" / "aerf_validity_a45a.py"
    spec = importlib.util.spec_from_file_location("aerf_validity_a45a", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a45a_fresh_suite_counts_and_hashes() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json")
    observed = _generator().manifest()
    fresh = config["fresh_validity_construction"]
    assert observed["counts"]["validation_units"] == 20
    assert observed["counts"]["pair_rows_by_split"]["validation"] == 180
    assert observed["counts"]["claim_rows_by_split"]["validation"] == 180
    assert observed["counts"]["relation_counts_by_split"]["validation"] == {
        "CONTRADICTED": 40,
        "ENTAILED": 60,
        "UNKNOWN": 80,
    }
    assert observed["sha256"]["validation_pairs"] == fresh["validation_pairs_sha256"]
    assert observed["sha256"]["validation_claims"] == fresh["validation_claims_sha256"]


def test_a45a_is_registration_only_and_model_unbound() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json")
    assert config["architecture"]["short_name"] == "AERF"
    assert config["architecture"]["status"] == "MEASUREMENT_REGISTERED_MODEL_UNBOUND"
    assert all(int(value) == 0 for value in config["scope"].values())
    assert config["next_checkpoint"]["authorized_by_a45a"] is False


def test_a45a_forbids_old_validation_as_hard_validity() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json")
    rules = config["reuse_rules"]
    assert rules["a44d_validation_hard_validity_reuse_forbidden"] is True
    assert rules["a44a_validation_hard_validity_reuse_forbidden"] is True
    assert rules["fresh_validation_may_not_be_used_for_model_or_threshold_selection"] is True


def test_a45a_preserves_confirmatory_seal() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json")
    boundary = config["confirmatory_boundary"]
    assert boundary["confirmatory_queries"] == 68
    assert boundary["confirmatory_query_records_inspected"] == 0
    assert boundary["confirmatory_scoring_authorized"] == 0
    assert (
        boundary["must_remain_sealed_until_fresh_validity_passes_and_separate_approval_is_granted"]
        is True
    )


def test_a45a_failure_directed_constraints_are_registered() -> None:
    config = _load(ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json")
    requirements = config["registered_component_requirements"]
    assert requirements["unknown_recall_min"] == 0.95
    assert requirements["cross_document_irrelevance_false_contradiction_rate_max"] == 0.02
    assert requirements["same_domain_irrelevance_false_contradiction_rate_max"] == 0.02
    assert requirements["relevant_insufficient_false_contradiction_rate_max"] == 0.02
    assert requirements["context_contamination_support_accuracy_min"] == 0.95

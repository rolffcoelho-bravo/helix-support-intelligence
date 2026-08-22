"""Registration tests for Phase 4 A4.5b AERF calibration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"
CASES = ROOT / "benchmarks" / "assistance" / "calibration_cases_a45b.py"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _cases_module() -> Any:
    spec = importlib.util.spec_from_file_location("calibration_cases_a45b", CASES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a45b_calibration_materializer_reproduces_frozen_a45a_partition() -> None:
    config = _json(CONFIG)
    observed = _cases_module().calibration_manifest()
    partition = config["calibration_partition"]
    assert observed["calibration_units"] == 40
    assert observed["calibration_pairs"] == 360
    assert observed["calibration_claims"] == 360
    assert observed["calibration_pairs_sha256"] == partition["pairs_sha256"]
    assert observed["calibration_claims_sha256"] == partition["claims_sha256"]
    assert observed["validation_units_materialized"] == 0
    assert observed["validation_pairs_materialized"] == 0
    assert observed["validation_claims_materialized"] == 0


def test_a45b_materializer_contains_only_registered_calibration_ids() -> None:
    config = _json(CONFIG)
    module = _cases_module()
    assert list(module.CALIBRATION_UNIT_IDS) == config["calibration_partition"]["unit_ids"]
    assert len(set(module.CALIBRATION_UNIT_IDS)) == 40
    source = CASES.read_text(encoding="utf-8")
    assert "build_suite" not in source
    assert "split_units" not in source


def test_a45b_binds_exactly_one_factorized_implementation() -> None:
    config = _json(CONFIG)
    assert config["architecture"]["short_name"] == "AERF"
    assert config["architecture"]["implementation_status"] == "BOUND_PRE_CALIBRATION"
    assert "never directly mapped to UNKNOWN" in config["architecture"]["unknown_construction"]
    binding = config["binding"]
    relevance = binding["alignment_relevance"]
    assert relevance["model_id"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert relevance["revision"] == "c5f2b386de279a97c53a702dd5189d1c407160dc"
    assert relevance["weights_sha256"] == (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    )
    nli = binding["sufficiency_polarity"]
    assert nli["model_id"] == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    assert nli["revision"] == "0e2603d5d3d3ef9b2910814b34eebe1a2101da65"
    assert nli["weights_sha256"] == (
        "06d6fd89edd4f97816831626daafbdb0b029cf63bae8edc0bccab1d64e2e7707"
    )
    assert nli["native_labels"] == {
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
    }


def test_a45b_threshold_search_is_frozen_before_calibration() -> None:
    config = _json(CONFIG)
    setup = config["threshold_calibration"]
    relevance = setup["relevance_grid"]
    sufficiency = setup["sufficiency_grid"]
    relevance_points = len(
        range(
            relevance["integer_start"],
            relevance["integer_stop"] + relevance["integer_step"],
            relevance["integer_step"],
        )
    )
    sufficiency_points = len(
        range(
            sufficiency["integer_start"],
            sufficiency["integer_stop"] + sufficiency["integer_step"],
            sufficiency["integer_step"],
        )
    )
    assert relevance_points == 241
    assert sufficiency_points == 50
    assert relevance_points * sufficiency_points == 12050
    assert setup["joint_candidates"] == 12050
    assert setup["no_validation_feedback"] is True
    assert setup["no_temperature_fit"] is True
    assert setup["no_class_specific_thresholds"] is True


def test_a45b_keeps_validation_and_confirmatory_sealed() -> None:
    config = _json(CONFIG)
    sealed = config["sealed_partitions"]
    assert sealed["fresh_validation_units"] == 20
    assert sealed["validation_scoring_authorized"] == 0
    assert sealed["validation_rows_materialized_by_a45b"] == 0
    assert sealed["confirmatory_queries"] == 68
    assert sealed["confirmatory_query_records_inspected"] == 0
    assert sealed["confirmatory_scoring_authorized"] == 0
    scope = config["scope"]
    assert scope["candidate_model_comparison_authorized"] == 0
    assert scope["validation_model_inference_authorized"] == 0
    assert scope["confirmatory_queries_scored"] == 0
    assert config["next_checkpoint"]["authorized_by_a45b"] is False

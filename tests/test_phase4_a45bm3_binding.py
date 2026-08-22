"""Registration tests for A4.5b-M3 SCEC binding and calibration protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm3_v1.json"
M2_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm2_manifest_v1.json"
M2_BUILDER = ROOT / "benchmarks" / "assistance" / "scec_calibration_a45bm2.py"
CORE = ROOT / "benchmarks" / "assistance" / "scec_calibration_core_a45bm3.py"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m2_manifest_remains_exactly_reproducible() -> None:
    builder = _module(M2_BUILDER, "scec_calibration_a45bm2")
    assert builder.manifest() == _json(M2_MANIFEST)


def test_exactly_one_model_is_bound_before_calibration() -> None:
    config = _json(CONFIG)
    implementation = config["authoritative_implementation"]
    assert implementation["count"] == 1
    model = implementation["semantic_model"]
    assert model["model_id"] == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert model["revision"] == "91562024e753ad76646a2d0dfcbb26801aa945fe"
    assert (
        model["weights_sha256"]
        == "6e8f2af78c828dcbd5243aac40fb87430376f0b8a9c288f4993df3ea3558d557"
    )
    assert model["native_labels"] == {"0": "entailment", "1": "not_entailment"}


def test_registered_parameter_grid_is_finite_and_exact() -> None:
    config = _json(CONFIG)
    calibration = config["calibration"]
    assert calibration["mismatch_threshold_grid"] == [
        round(0.34 + 0.02 * index, 2) for index in range(29)
    ]
    assert calibration["coverage_threshold_grid"] == [
        round(0.50 + 0.02 * index, 2) for index in range(21)
    ]
    assert calibration["joint_candidate_count"] == 609


def test_unspecified_scope_is_not_forced_to_mismatch() -> None:
    core = _module(CORE, "scec_calibration_core_a45bm3")
    scores = {"MATCH": 0.30, "MISMATCH": 0.10, "UNSPECIFIED": 0.60}
    assert core._dimension_label(scores, 0.34) == "UNSPECIFIED"


def test_mismatch_requires_registered_global_threshold() -> None:
    core = _module(CORE, "scec_calibration_core_a45bm3")
    below = {"MATCH": 0.32, "MISMATCH": 0.33, "UNSPECIFIED": 0.35}
    above = {"MATCH": 0.20, "MISMATCH": 0.60, "UNSPECIFIED": 0.20}
    assert core._dimension_label(below, 0.50) == "UNSPECIFIED"
    assert core._dimension_label(above, 0.50) == "MISMATCH"


def test_claim_composition_uses_predicted_set_mapping() -> None:
    config = _json(CONFIG)
    mapping = config["authoritative_implementation"]["claim_source_mapping"]
    assert mapping == {
        "single_supported": "S01",
        "single_refuted": "S02",
        "compatible_insufficient": "S03",
        "complementary_multi_span_supported": "S05",
        "support_refute_conflict": "S07",
        "citation_invalid": "S01",
        "stale_evidence": "S01",
        "registered_conflict": "S01",
    }


def test_validation_and_confirmatory_partitions_remain_sealed() -> None:
    config = _json(CONFIG)
    assert all(int(value) == 0 for value in config["sealed_partitions"].values())
    scope = config["execution_scope"]
    assert scope["validation_scoring_authorized"] == 0
    assert scope["confirmatory_scoring_authorized"] == 0
    assert config["scientific_outcomes"]["fresh_validation_authorized_after_pass"] is False
    assert config["scientific_outcomes"]["next_checkpoint_authorized"] is False


def test_no_post_binding_model_or_prompt_search_is_allowed() -> None:
    config = _json(CONFIG)
    assert all(bool(value) for value in config["forbidden_after_binding"].values())

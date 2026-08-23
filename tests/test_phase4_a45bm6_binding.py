"""Static and deterministic tests for the A4.5b-M6 TPAG binding."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSISTANCE = ROOT / "benchmarks" / "assistance"
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm6_v1.json"
M5_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm5_v1.json"
CORE = ASSISTANCE / "tpag_core_a45bm6.py"
BUILDER = ASSISTANCE / "tpag_calibration_a45bm5.py"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _module(name: str, path: Path) -> Any:
    if str(ASSISTANCE) not in sys.path:
        sys.path.insert(0, str(ASSISTANCE))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_raw_scores(core: Any, suite: dict[str, Any]) -> dict[str, dict[str, float]]:
    aliases = core.make_alias_map(suite["units"])
    requests = core.collect_residual_requests(suite, aliases)
    scores: dict[str, dict[str, float]] = {}
    for request in requests:
        evidence_value = str(request["evidence_value"])
        if evidence_value.startswith("complete ") and evidence_value.endswith(" handling"):
            probabilities = {"contradiction": 0.005, "entailment": 0.99, "neutral": 0.005}
        else:
            probabilities = {"contradiction": 0.99, "entailment": 0.005, "neutral": 0.005}
        scores[str(request["request_id"])] = probabilities
    return scores


def test_m6_binds_exactly_one_pipeline_and_one_active_threshold() -> None:
    config = _json(CONFIG)
    implementation = config["authoritative_implementation"]
    assert implementation["count"] == 1
    assert implementation["short_name"] == "TPAG"
    assert implementation["novelty_claim"] is False
    assert implementation["semantic_model"]["role"] == (
        "residual query-conditioned typed-edge semantic equivalence only"
    )
    grid = config["calibration_parameter_grid"]
    assert grid["active_thresholds"] == ["alignment_confidence_min"]
    assert grid["disabled_thresholds"] == [
        "extraction_confidence_min",
        "polarity_confidence_min",
    ]
    assert grid["candidate_count"] == 7
    assert grid["candidate_count"] <= grid["m5_maximum_joint_candidates"] == 343


def test_bound_model_pin_and_native_labels_are_frozen() -> None:
    model = _json(CONFIG)["authoritative_implementation"]["semantic_model"]
    assert model["model_id"] == "cross-encoder/nli-deberta-v3-base"
    assert model["revision"] == "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
    assert model["weights_sha256"] == (
        "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa"
    )
    assert model["tokenizer_sha256"] == (
        "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"
    )
    assert model["native_labels"] == {
        "0": "contradiction",
        "1": "entailment",
        "2": "neutral",
    }


def test_residual_request_surface_is_small_and_gold_independent() -> None:
    core = _module("tpag_core_a45bm6", CORE)
    builder = _module("tpag_calibration_a45bm5_for_m6", BUILDER)
    suite = builder.build_suite()
    aliases = core.make_alias_map(suite["units"])
    requests = core.collect_residual_requests(suite, aliases)
    assert len(requests) == 128
    assert len({str(row["request_id"]) for row in requests}) == 128
    assert all("gold" not in row for row in requests)
    evidence_values = {str(row["evidence_value"]) for row in requests}
    assert "archive" in evidence_values
    assert any(value.startswith("complete ") for value in evidence_values)


def test_deterministic_proposition_parser_handles_decontextualization() -> None:
    core = _module("tpag_core_a45bm6_prop", CORE)
    builder = _module("tpag_calibration_a45bm5_prop", BUILDER)
    suite = builder.build_suite()
    aliases = core.make_alias_map(suite["units"])
    subtypes = {str(row["subtype"]): row for row in suite["proposition_rows"][:8]}
    for subtype in (
        "single_clean_target",
        "target_after_unrelated_prefix",
        "target_before_unrelated_suffix",
        "alias_requires_decontextualization",
        "pronoun_requires_decontextualization",
        "coordinated_independent_propositions",
        "parenthetical_context_target",
        "no_target_proposition",
    ):
        row = subtypes[subtype]
        prediction = core.predict_proposition_case(row, aliases)
        assert prediction["surface_propositions"] == row["gold"]["surface_propositions"]
        assert prediction["target_proposition_indices"] == row["gold"]["target_proposition_indices"]
        assert prediction["decontextualized_target_propositions"] == row["gold"][
            "decontextualized_target_propositions"
        ]


def test_fake_residual_scores_recover_registered_structural_cases() -> None:
    core = _module("tpag_core_a45bm6_struct", CORE)
    builder = _module("tpag_calibration_a45bm5_struct", BUILDER)
    suite = builder.build_suite()
    scores = _fake_raw_scores(core, suite)
    predictions = core.predict_all(suite, scores, 0.9)
    alignment_by_id = {str(row["alignment_id"]): row for row in predictions["alignments"]}
    for row in suite["alignment_rows"]:
        prediction = alignment_by_id[str(row["alignment_id"])]
        assert prediction["scope_compatibility"] == row["gold"]["scope_compatibility"]
        assert prediction["final_relation"] == row["gold"]["final_relation"]
    group_by_id = {str(row["group_id"]): row for row in predictions["groups"]}
    for row in suite["evidence_group_rows"]:
        prediction = group_by_id[str(row["group_id"])]
        assert prediction["sufficiency"] == row["gold"]["sufficiency"]
        assert prediction["final_relation"] == row["gold"]["final_relation"]


def test_metric_surface_matches_all_56_frozen_requirements() -> None:
    core = _module("tpag_core_a45bm6_metrics", CORE)
    builder = _module("tpag_calibration_a45bm5_metrics", BUILDER)
    suite = builder.build_suite()
    scores = _fake_raw_scores(core, suite)
    predictions = core.predict_all(suite, scores, 0.9)
    metrics = core.evaluate_predictions(suite, predictions)
    requirements = _json(M5_CONFIG)["calibration_readiness_requirements"]
    expected_metrics = {
        str(name).removesuffix("_min").removesuffix("_max") for name in requirements
    }
    assert len(requirements) == 56
    assert set(metrics) == expected_metrics
    checks = core.requirement_results(
        metrics, {str(name): float(value) for name, value in requirements.items()}
    )
    assert len(checks) == 56


def test_execution_contract_keeps_future_evidence_sealed() -> None:
    config = _json(CONFIG)
    execution = config["execution_contract"]
    assert execution["m5_calibration_units_authorized"] == 64
    assert execution["threshold_candidates_authorized"] == 7
    for name in (
        "model_family_comparisons_authorized",
        "prompt_searches_authorized",
        "closed_a45bm2_m3_rows_authorized",
        "a45a_fresh_validation_rows_authorized",
        "confirmatory_queries_authorized",
        "future_validation_construction_authorized",
        "post_result_rescue_authorized",
    ):
        assert execution[name] == 0
    assert config["next_checkpoint"]["authorized"] is False

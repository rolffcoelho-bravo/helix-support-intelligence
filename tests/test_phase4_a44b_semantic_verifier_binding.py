from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"
PREFLIGHT_PATH = ROOT / "scripts" / "preflight_phase4_a44b.py"

EXPECTED_MODEL = "FacebookAI/roberta-large-mnli"
EXPECTED_REVISION = "2a8f12d27941090092df78e4ba6f0928eb5eac98"
EXPECTED_WEIGHTS_SHA = "f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5"


def _config() -> dict[str, object]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_a44b_binds_one_immutable_architecture_independent_model() -> None:
    config = _config()
    verifier = config["semantic_verifier"]
    assert isinstance(verifier, dict)

    assert verifier["model_id"] == EXPECTED_MODEL
    assert verifier["revision"] == EXPECTED_REVISION
    assert verifier["weights_file"] == "model.safetensors"
    assert verifier["weights_sha256"] == EXPECTED_WEIGHTS_SHA
    assert verifier["use_safetensors"] is True
    assert verifier["trust_remote_code"] is False
    assert verifier["license"] == "MIT"
    assert verifier["architecture_family"] == "roberta"

    independence = verifier["independence"]
    assert isinstance(independence, dict)
    assert independence["frozen_g2_runtime_architecture_family"] == "deberta-v3"
    assert independence["rejected_a43a_architecture_family"] == "minilm2-roberta"
    assert independence["selected_architecture_differs_from_g2_runtime"] is True
    assert independence["selected_architecture_differs_from_rejected_a43a_evaluator"] is True


def test_a44b_three_way_relation_mapping_is_exact_and_threshold_free() -> None:
    config = _config()
    verifier = config["semantic_verifier"]
    assert isinstance(verifier, dict)

    assert verifier["native_label_mapping"] == {
        "0": "CONTRADICTION",
        "1": "NEUTRAL",
        "2": "ENTAILMENT",
    }
    assert verifier["a44a_relation_mapping"] == {
        "CONTRADICTION": "CONTRADICTED",
        "NEUTRAL": "UNKNOWN",
        "ENTAILMENT": "ENTAILED",
    }

    decision = verifier["class_decision"]
    assert isinstance(decision, dict)
    assert decision["rule"] == "argmax_raw_logits"
    assert decision["class_thresholds"] is None
    assert decision["margin_threshold"] is None
    assert decision["abstention_threshold"] is None
    assert decision["calibration_may_change_predicted_class"] is False


def test_a44b_future_temperature_calibration_cannot_change_decisions() -> None:
    config = _config()
    calibration = config["calibration_policy"]
    assert isinstance(calibration, dict)

    assert calibration["execution_in_a44b"] is False
    assert calibration["calibration_case_rows_registered"] == 288
    assert calibration["validation_case_rows_available_to_calibration"] == 0
    assert calibration["model_weights_frozen"] is True
    assert calibration["class_decision_frozen_to_raw_argmax"] is True

    fit = calibration["fit"]
    assert fit == {
        "parameter": "single_global_temperature",
        "purpose": "probability calibration diagnostics only",
        "objective": "three_class_negative_log_likelihood",
        "search": "deterministic_grid",
        "grid_start": 0.25,
        "grid_stop": 4.0,
        "grid_step": 0.01,
        "tie_break": "smallest_temperature",
        "may_change_class_labels": False,
        "may_change_claim_verdicts": False,
    }
    assert all(bool(value) for value in calibration["forbidden"].values())


def test_a44b_registration_has_zero_scientific_execution_surface() -> None:
    config = _config()
    guards = config["execution_guards"]
    assert isinstance(guards, dict)
    assert all(int(value) == 0 for value in guards.values())

    boundary = config["future_execution_boundary"]
    assert isinstance(boundary, dict)
    for field in (
        "semantic_inference_authorized_in_a44b",
        "a44a_calibration_execution_authorized_in_a44b",
        "a44a_validation_execution_authorized_in_a44b",
        "candidate_comparison_authorized_in_a44b",
        "confirmatory_execution_authorized_in_a44b",
    ):
        assert boundary[field] is False


def test_a44b_preflight_cannot_materialize_cases_or_load_models() -> None:
    source = PREFLIGHT_PATH.read_text(encoding="utf-8")
    forbidden = (
        "compositional_cases_a44a",
        "generate_cases(",
        "transformers",
        "torch",
        "huggingface_hub",
        "AutoModel",
        "AutoTokenizer",
    )
    for token in forbidden:
        assert token not in source


def test_a44b_external_scores_are_not_internal_validity_evidence() -> None:
    config = _config()
    selection = config["selection_policy"]
    evidence = config["external_model_evidence"]
    assert isinstance(selection, dict)
    assert isinstance(evidence, dict)

    assert selection["external_benchmark_results_are_helixbank_acceptance_evidence"] is False
    assert evidence["documented_external_mnli_score"] == 90.2
    assert evidence["external_score_used_for_helixbank_validity"] is False
    assert evidence["snapshot_date"] == "2026-08-21"

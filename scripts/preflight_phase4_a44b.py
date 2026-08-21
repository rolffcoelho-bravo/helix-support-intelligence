"""Preflight the frozen A4.4b semantic-verifier binding without model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
A44A_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"
A44A_AUDIT = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a44a_protocol_post_audit_v1.json"
)
A44B_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"
A41_CONFIG = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"
A43A_AUDIT = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a43a_validity_postresult_v1"
    / "forensic_audit.json"
)

EXPECTED_BINDING_ID = "phase4-assistance-a4.4b-semantic-verifier-binding-v1"
EXPECTED_SOURCE_SHA = "4aa86f70c1a829c6e90d03414736e750daef7c66"
EXPECTED_SUITE_SHA = "0ad07e9d08678dbc5fa8b625870d2c3140eef83b0dddb013a4ae479c56bdd90c"
EXPECTED_MODEL_ID = "FacebookAI/roberta-large-mnli"
EXPECTED_REVISION = "2a8f12d27941090092df78e4ba6f0928eb5eac98"
EXPECTED_WEIGHTS_SHA = "f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def preflight() -> dict[str, Any]:
    """Validate A4.4b registration while keeping all benchmark records unopened."""
    a44a = _load(A44A_CONFIG)
    a44a_audit = _load(A44A_AUDIT)
    a44b = _load(A44B_CONFIG)
    a41 = _load(A41_CONFIG)
    a43a = _load(A43A_AUDIT)

    if a44b["binding_id"] != EXPECTED_BINDING_ID:
        raise RuntimeError("A4.4b binding identifier changed after registration.")
    if a44b["source_main_sha"] != EXPECTED_SOURCE_SHA:
        raise RuntimeError("A4.4b must remain based on the frozen A4.4a merge SHA.")
    if a44a["validation_suite"]["sha256"] != EXPECTED_SUITE_SHA:
        raise RuntimeError("A4.4a suite SHA changed before A4.4b binding.")
    if a44a_audit["suite"]["sha256"] != EXPECTED_SUITE_SHA:
        raise RuntimeError("A4.4a audit and protocol suite SHA disagree.")
    if a44a_audit["status"] != "PASSED_PRE_EXECUTION_NO_RESULTS":
        raise RuntimeError("A4.4b requires the frozen no-result A4.4a disposition.")
    if a44a_audit["scientific_disposition"]["confirmatory_partition_remains_unopened"] is not True:
        raise RuntimeError("A4.4b requires the confirmatory partition to remain unopened.")
    if a43a["status"] != "CLOSED_FAILED_EVALUATOR_VALIDITY":
        raise RuntimeError("A4.4b requires the closed negative A4.3a evaluator result.")

    selection = a44b["selection_policy"]
    required_false = (
        "a42_candidate_results_used",
        "a43a_validation_errors_used_to_rank_replacement_models",
        "a44a_calibration_cases_opened",
        "a44a_validation_cases_opened",
        "local_model_search_performed",
        "empirical_candidate_bakeoff_performed",
        "external_benchmark_results_are_helixbank_acceptance_evidence",
    )
    for field in required_false:
        if selection[field] is not False:
            raise RuntimeError(f"A4.4b selection guard {field} must remain false.")
    if not all(bool(value) for value in selection["eligibility_criteria"].values()):
        raise RuntimeError("Every A4.4b structural eligibility criterion must remain required.")

    verifier = a44b["semantic_verifier"]
    if verifier["model_id"] != EXPECTED_MODEL_ID:
        raise RuntimeError("A4.4b semantic verifier model id changed after binding.")
    if verifier["revision"] != EXPECTED_REVISION:
        raise RuntimeError("A4.4b semantic verifier revision changed after binding.")
    if verifier["weights_sha256"] != EXPECTED_WEIGHTS_SHA:
        raise RuntimeError("A4.4b safetensors SHA-256 changed after binding.")
    if verifier["weights_file"] != "model.safetensors" or verifier["use_safetensors"] is not True:
        raise RuntimeError("A4.4b must remain bound to safetensors weights.")
    if verifier["trust_remote_code"] is not False:
        raise RuntimeError("A4.4b remote code must remain disabled.")
    if verifier["license"] != "MIT":
        raise RuntimeError("A4.4b selected verifier license changed after registration.")
    if int(verifier["tokenizer_max_length"]) < 512:
        raise RuntimeError("A4.4b tokenizer limit must remain at least 512 tokens.")
    if verifier["premise_field"] != "document.body" or verifier["hypothesis_field"] != "atom.text":
        raise RuntimeError("A4.4b semantic pair field order changed after registration.")

    native = verifier["native_label_mapping"]
    if native != {"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}:
        raise RuntimeError("A4.4b native label mapping changed after registration.")
    relation = verifier["a44a_relation_mapping"]
    if relation != {
        "CONTRADICTION": "CONTRADICTED",
        "NEUTRAL": "UNKNOWN",
        "ENTAILMENT": "ENTAILED",
    }:
        raise RuntimeError("A4.4b A4.4a relation mapping changed after registration.")
    decision = verifier["class_decision"]
    if decision["rule"] != "argmax_raw_logits":
        raise RuntimeError("A4.4b class decision must remain raw-logit argmax.")
    if any(decision[field] is not None for field in ("class_thresholds", "margin_threshold", "abstention_threshold")):
        raise RuntimeError("A4.4b class decision may not introduce thresholds.")
    if decision["calibration_may_change_predicted_class"] is not False:
        raise RuntimeError("A4.4b calibration may not change predicted classes.")

    runtime = a41["runtime_verifier"]
    rejected = a41["evaluation_verifier"]
    if a43a["scientific_disposition"]["rejected_model_id"] != rejected["model_id"]:
        raise RuntimeError("A4.3a rejected model does not match the frozen A4.1 evaluator.")
    selected_family = str(verifier["architecture_family"])
    if selected_family == str(runtime["architecture_family"]):
        raise RuntimeError("A4.4b verifier must remain architecture-independent from G2 runtime.")
    if selected_family == str(rejected["architecture_family"]):
        raise RuntimeError("A4.4b verifier must differ from the rejected A4.3a family.")

    calibration = a44b["calibration_policy"]
    if calibration["execution_in_a44b"] is not False:
        raise RuntimeError("A4.4b may not execute calibration.")
    if int(calibration["calibration_case_rows_registered"]) != 288:
        raise RuntimeError("A4.4b calibration row registration changed.")
    if int(calibration["validation_case_rows_available_to_calibration"]) != 0:
        raise RuntimeError("A4.4b calibration may not access validation rows.")
    fit = calibration["fit"]
    if fit != {
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
    }:
        raise RuntimeError("A4.4b temperature-calibration rule changed after registration.")
    if not all(bool(value) for value in calibration["forbidden"].values()):
        raise RuntimeError("Every A4.4b forbidden calibration action must remain forbidden.")

    boundary = a44b["future_execution_boundary"]
    for field in (
        "semantic_inference_authorized_in_a44b",
        "a44a_calibration_execution_authorized_in_a44b",
        "a44a_validation_execution_authorized_in_a44b",
        "candidate_comparison_authorized_in_a44b",
        "confirmatory_execution_authorized_in_a44b",
    ):
        if boundary[field] is not False:
            raise RuntimeError(f"A4.4b execution boundary {field} must remain false.")
    if boundary["next_calibration_execution_requires_separate_versioned_checkpoint"] is not True:
        raise RuntimeError("A4.4b must not authorize its future calibration execution.")
    if boundary["validation_must_remain_unopened_until_model_and_calibration_rules_are_frozen"] is not True:
        raise RuntimeError("A4.4b validation isolation rule changed.")

    guards = a44b["execution_guards"]
    for name, value in guards.items():
        if int(value) != 0:
            raise RuntimeError(f"A4.4b execution guard {name} must remain zero.")

    evidence = a44b["external_model_evidence"]
    if evidence["snapshot_date"] != "2026-08-21":
        raise RuntimeError("A4.4b external model-evidence snapshot date changed.")
    if evidence["external_score_used_for_helixbank_validity"] is not False:
        raise RuntimeError("External model scores may not become HelixBank validity evidence.")

    return {
        "binding_id": a44b["binding_id"],
        "source_main_sha": a44b["source_main_sha"],
        "a44a_suite_sha256": EXPECTED_SUITE_SHA,
        "selected_model_id": verifier["model_id"],
        "selected_revision": verifier["revision"],
        "selected_architecture_family": verifier["architecture_family"],
        "weights_file": verifier["weights_file"],
        "weights_sha256": verifier["weights_sha256"],
        "native_labels": native,
        "a44a_relation_mapping": relation,
        "class_decision": decision["rule"],
        "future_calibration_parameter": fit["parameter"],
        "future_calibration_grid_points": 376,
        "candidate_calls_made": 0,
        "openai_calls_made": 0,
        "semantic_verifier_downloads_made": 0,
        "semantic_verifier_calls_made": 0,
        "model_family_searches_made": 0,
        "a44a_calibration_case_records_inspected": 0,
        "a44a_validation_case_records_inspected": 0,
        "confirmatory_query_records_inspected": 0,
        "confirmatory_queries_scored": 0,
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))

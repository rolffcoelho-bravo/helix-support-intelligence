"""Fail-closed preflight for A4.4d validation-only execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
A44A = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"
A44B = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"
A44C = ROOT / "configs" / "models" / "assistance_grounding_a44c_v1.json"
A44D = ROOT / "configs" / "models" / "assistance_grounding_a44d_v1.json"
A44C_CLOSURE = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a44c_calibration_postresult_v1"
    / "forensic_audit.json"
)
FROZEN_A44A_SUITE_SHA256 = "0ad07e9d08678dbc5fa8b625870d2c3140eef83b0dddb013a4ae479c56bdd90c"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def main() -> None:
    a44a = _load(A44A)
    a44b = _load(A44B)
    a44c = _load(A44C)
    a44d = _load(A44D)
    closure = _load(A44C_CLOSURE)

    assert a44d["execution_id"] == "phase4-assistance-a4.4d-validation-only-v1"
    assert a44d["source_main_sha"] == "57bb2c81ab2cc2d5b8c1a4928c2600b4a770d110"
    assert a44d["protocol_id"] == a44a["protocol_id"]
    assert a44d["binding_id"] == a44b["binding_id"]
    assert a44d["calibration_execution_id"] == a44c["execution_id"]
    assert a44a["validation_suite"]["sha256"] == FROZEN_A44A_SUITE_SHA256
    assert a44a["validation_suite"]["expected_counts"]["validation"] == 144
    assert a44a["validation_suite"]["partition"]["validation_intents"] == 20
    assert a44d["scope"]["validation_case_rows"] == 144
    assert a44d["scope"]["validation_intents"] == 20
    assert a44d["scope"]["validation_semantic_pair_rows"] == 246
    assert a44d["scope"]["validation_gold_relation_counts"] == {
        "ENTAILED": 106,
        "CONTRADICTED": 20,
        "UNKNOWN": 120,
    }
    assert a44d["scope"]["calibration_case_rows_authorized"] == 0
    assert a44d["scope"]["candidate_rows_authorized"] == 0
    assert a44d["scope"]["confirmatory_query_rows_authorized"] == 0

    assert closure["status"] == "CLOSED_CALIBRATION_TEMPERATURE_FROZEN"
    assert closure["scientific_disposition"]["global_temperature_is_frozen_at_3_67"] is True
    assert closure["scientific_disposition"]["validation_remains_unopened"] is True
    assert closure["scientific_disposition"]["confirmatory_partition_remains_unopened"] is True
    assert a44d["frozen_calibration"]["selected_temperature"] == 3.67
    assert a44d["frozen_calibration"]["post_validation_refit_permitted"] is False

    verifier = a44b["semantic_verifier"]
    assert verifier["model_id"] == "FacebookAI/roberta-large-mnli"
    assert verifier["revision"] == "2a8f12d27941090092df78e4ba6f0928eb5eac98"
    assert verifier["weights_sha256"] == (
        "f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5"
    )
    assert verifier["class_decision"]["rule"] == "argmax_raw_logits"
    assert verifier["fine_tuning_permitted"] is False

    assert a44d["validation_requirements"] == a44a["future_validation_requirements"]
    assert a44d["validation_requirements"]["all_requirements_must_pass"] is True
    assert a44d["forbidden"]["temperature_refit"] is True
    assert a44d["forbidden"]["threshold_search"] is True
    assert a44d["forbidden"]["model_substitution"] is True
    assert a44d["forbidden"]["candidate_inference"] is True
    assert a44d["forbidden"]["confirmatory_inference"] is True
    assert a44d["forbidden"]["post_result_rescue"] is True

    print(
        json.dumps(
            {
                "status": "PASSED_PRE_EXECUTION_VALIDATION_ONLY_NO_RESULTS",
                "validation_cases_authorized": 144,
                "validation_intents_authorized": 20,
                "validation_semantic_pairs_registered": 246,
                "frozen_temperature": 3.67,
                "calibration_cases_authorized": 0,
                "candidate_rows_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "validation_cases_materialized_by_preflight": 0,
                "semantic_inference_performed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

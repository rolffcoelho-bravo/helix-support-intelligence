"""Fail-closed preflight for A4.4e methodology selection with no new inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44e_v1.json"
A44D_FORENSIC_PATH = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a44d_validation_postresult_v1"
    / "forensic_audit.json"
)
DECISION_DOC_PATH = ROOT / "docs" / "assistance-a44e-methodology-decision.md"
LITERATURE_PATH = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a44e_methodology_decision_v1"
    / "literature_matrix.md"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def main() -> None:
    config = _load(CONFIG_PATH)
    a44d = _load(A44D_FORENSIC_PATH)

    if config["checkpoint"] != "A4.4e":
        raise RuntimeError("A4.4e checkpoint identifier drifted.")
    if config["source_main_sha"] != "291718add7a37681761ce8365d6db8dfbe504151":
        raise RuntimeError("A4.4e predecessor main SHA drifted.")

    if a44d["status"] != "CLOSED_FAILED_REGISTERED_VALIDATION_NO_RESCUE":
        raise RuntimeError("A4.4d must remain closed as the registered failed validation.")
    disposition = a44d["scientific_disposition"]
    if not bool(disposition["confirmatory_partition_remains_unopened"]):
        raise RuntimeError("A4.4d no longer records the confirmatory partition as sealed.")
    if not bool(disposition["failed_result_is_preserved"]):
        raise RuntimeError("A4.4d failed result is not preserved.")

    scope = config["scope"]
    zero_fields = (
        "semantic_inference_authorized",
        "candidate_model_bindings_authorized",
        "candidate_rows_scored",
        "calibration_rows_rescored",
        "a44a_validation_rows_rescored",
        "confirmatory_query_rows_authorized",
        "confirmatory_query_records_inspected",
        "threshold_searches_authorized",
        "temperature_refits_authorized",
    )
    for field in zero_fields:
        if int(scope[field]) != 0:
            raise RuntimeError(f"A4.4e must keep {field}=0.")

    decision = config["methodology_decision"]
    if decision["selected_architecture"] != "Atom-Evidence Relation Factorization":
        raise RuntimeError("A4.4e selected architecture drifted.")
    if decision["short_name"] != "AERF":
        raise RuntimeError("A4.4e architecture short name drifted.")
    if decision["status"] != "ARCHITECTURE_SELECTED_MODEL_UNBOUND":
        raise RuntimeError("A4.4e must leave learned components unbound.")

    rejected = config["rejected_paths"]
    if not all(bool(value) for value in rejected.values()):
        raise RuntimeError("Every A4.4e rejected rescue path must remain rejected.")

    next_checkpoint = config["next_checkpoint"]
    if next_checkpoint["checkpoint"] != "A4.5a":
        raise RuntimeError("A4.4e next checkpoint drifted.")
    if bool(next_checkpoint["authorized_by_a44e"]):
        raise RuntimeError("A4.4e must not authorize A4.5a execution.")
    if not bool(next_checkpoint["requires_separate_approval"]):
        raise RuntimeError("A4.5a must require separate approval.")

    if not DECISION_DOC_PATH.exists() or not LITERATURE_PATH.exists():
        raise RuntimeError("A4.4e decision evidence is incomplete.")

    print(
        json.dumps(
            {
                "status": "PASSED_A44E_METHODOLOGY_DECISION_NO_INFERENCE",
                "selected_architecture": "AERF",
                "semantic_inference_authorized": 0,
                "model_bindings_authorized": 0,
                "a44a_validation_rows_rescored": 0,
                "confirmatory_queries_authorized": 0,
                "next_checkpoint_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

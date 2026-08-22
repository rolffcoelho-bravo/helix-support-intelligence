"""Fail-closed audit for the permanent A4.5b negative calibration closure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmarks" / "assistance" / "results" / "a45b_calibration_postresult_v1"
SUMMARY = RESULT_DIR / "result_summary.json"
FORENSIC = RESULT_DIR / "forensic_audit.json"
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def main() -> None:
    summary = _json(SUMMARY)
    forensic = _json(FORENSIC)
    config = _json(CONFIG)

    if summary["checkpoint"] != "A4.5b":
        raise RuntimeError("A4.5b closure checkpoint drifted")
    if summary["status"] != "CLOSED_FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED":
        raise RuntimeError("A4.5b closure status drifted")
    if summary["scientific_pass"] is not False:
        raise RuntimeError("A4.5b negative result cannot be promoted to pass")
    if summary["threshold_selection"]["joint_candidates_evaluated"] != 12050:
        raise RuntimeError("A4.5b threshold-grid cardinality drifted")
    if summary["threshold_selection"]["feasible_candidate_count"] != 0:
        raise RuntimeError("A4.5b closure must preserve zero feasible candidates")
    failed = summary["failed_readiness_requirements"]
    if set(failed) != {
        "relevance_macro_f1_min",
        "relevant_recall_min",
        "irrelevant_recall_min",
    }:
        raise RuntimeError("A4.5b failed readiness requirement set drifted")

    confusion = forensic["relevance_confusion"]
    expected_confusion = {
        "gold_relevant_pred_relevant": 239,
        "gold_relevant_pred_irrelevant": 41,
        "gold_irrelevant_pred_irrelevant": 59,
        "gold_irrelevant_pred_relevant": 21,
        "gold_relevant_total": 280,
        "gold_irrelevant_total": 80,
    }
    if confusion != expected_confusion:
        raise RuntimeError("A4.5b frozen relevance confusion matrix drifted")

    if forensic["deterministic_recovery"]["second_model_inference_performed"] is not False:
        raise RuntimeError("A4.5b deterministic recovery cannot perform second inference")
    if forensic["deterministic_recovery"]["artifact_id"] != 9478056833:
        raise RuntimeError("A4.5b recovery artifact drifted")
    if forensic["source_inference"]["artifact_id"] != 9477913279:
        raise RuntimeError("A4.5b source-inference artifact drifted")

    for name, value in summary["sealed_partitions"].items():
        if int(value) != 0:
            raise RuntimeError(f"A4.5b closure sealed counter is nonzero: {name}")

    governance = summary["governance"]
    if governance["post_result_threshold_rescue_authorized"] is not False:
        raise RuntimeError("A4.5b closure forbids threshold rescue")
    if governance["model_substitution_authorized"] is not False:
        raise RuntimeError("A4.5b closure forbids model substitution")
    if governance["a45c_eligible"] is not False or governance["a45c_authorized"] is not False:
        raise RuntimeError("A4.5c must remain ineligible and unauthorized")

    next_checkpoint = config["next_checkpoint"]
    if next_checkpoint["checkpoint"] != "A4.5c":
        raise RuntimeError("Registered A4.5c identity drifted")
    if next_checkpoint["authorized_by_a45b"] is not False:
        raise RuntimeError("A4.5c cannot be authorized by failed A4.5b")

    print(
        json.dumps(
            {
                "status": "PASSED_A45B_POSTRESULT_CLOSURE",
                "scientific_pass": False,
                "failed_readiness_requirements": 3,
                "feasible_threshold_candidates": 0,
                "validation_rows_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "a45c_eligible": False,
                "a45c_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

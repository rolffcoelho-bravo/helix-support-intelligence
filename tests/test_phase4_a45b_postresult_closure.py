"""Static tests for the permanent A4.5b negative calibration closure."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmarks" / "assistance" / "results" / "a45b_calibration_postresult_v1"


def _json(name: str) -> dict[str, object]:
    value = json.loads((RESULT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a45b_closure_freezes_negative_status() -> None:
    summary = _json("result_summary.json")
    assert summary["checkpoint"] == "A4.5b"
    assert summary["scientific_pass"] is False
    assert summary["scientific_status"] == "FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED"
    thresholds = summary["threshold_selection"]
    assert isinstance(thresholds, dict)
    assert thresholds["joint_candidates_evaluated"] == 12050
    assert thresholds["feasible_candidate_count"] == 0


def test_a45b_closure_preserves_failure_geometry() -> None:
    forensic = _json("forensic_audit.json")
    confusion = forensic["relevance_confusion"]
    assert isinstance(confusion, dict)
    assert confusion == {
        "gold_relevant_pred_relevant": 239,
        "gold_relevant_pred_irrelevant": 41,
        "gold_irrelevant_pred_irrelevant": 59,
        "gold_irrelevant_pred_relevant": 21,
        "gold_relevant_total": 280,
        "gold_irrelevant_total": 80,
    }
    geometry = forensic["failure_geometry"]
    assert isinstance(geometry, dict)
    assert geometry["relevant_but_insufficient"] == {"total": 40, "predicted_irrelevant": 40}
    assert geometry["cross_document_irrelevance"] == {"total": 40, "predicted_relevant": 21}


def test_a45b_closure_keeps_future_partitions_sealed() -> None:
    summary = _json("result_summary.json")
    sealed = summary["sealed_partitions"]
    assert isinstance(sealed, dict)
    assert all(int(value) == 0 for value in sealed.values())
    governance = summary["governance"]
    assert isinstance(governance, dict)
    assert governance["a45c_eligible"] is False
    assert governance["a45c_authorized"] is False
    assert governance["post_result_threshold_rescue_authorized"] is False
    assert governance["model_substitution_authorized"] is False


def test_a45b_recovery_provenance_is_frozen() -> None:
    forensic = _json("forensic_audit.json")
    source = forensic["source_inference"]
    recovery = forensic["deterministic_recovery"]
    assert isinstance(source, dict)
    assert isinstance(recovery, dict)
    assert source["workflow_run_id"] == 32581433921
    assert source["artifact_id"] == 9477913279
    assert recovery["workflow_run_id"] == 32581996227
    assert recovery["artifact_id"] == 9478056833
    assert recovery["second_model_inference_performed"] is False

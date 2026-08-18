from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "benchmarks" / "routing" / "results" / "calibration_validation_v1.json"
SCRIPT_PATH = ROOT / "benchmarks" / "routing" / "evaluate_calibration.py"
LOCK_PATH = ROOT / "benchmarks" / "routing" / "evaluate_calibration.py.lock"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "phase2-routing-calibration.yml"


def _checkpoint() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")),
    )


def test_calibration_checkpoint_selects_temperature_for_a1_and_a2() -> None:
    checkpoint = _checkpoint()
    a1 = checkpoint["A1"]
    a2 = checkpoint["A2"]
    assert isinstance(a1, dict)
    assert isinstance(a2, dict)
    assert a1["selected_method"] == "temperature_scaling"
    assert a2["selected_method"] == "temperature_scaling"
    assert checkpoint["test_set_opened"] is False


def test_calibration_balanced_cross_fit_is_frozen() -> None:
    checkpoint = _checkpoint()
    cross_fitting = checkpoint["cross_fitting"]
    assert isinstance(cross_fitting, dict)
    counts = cross_fitting["fold_counts"]
    assert isinstance(counts, dict)
    values = [int(counts[str(index)]) for index in range(5)]
    assert values == [390, 392, 393, 402, 399]
    assert max(values) - min(values) <= 12
    assert sum(values) == 1976
    assert cross_fitting["each_row_scored_once"] is True
    assert cross_fitting["score_row_used_to_fit_its_calibrator"] is False

    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'offset_digest = hashlib.sha256(f"{salt}\\t{label}".encode()).digest()' in script
    assert "assignment[row_index] = (position + offset) % folds" in script
    assert "assignment[row_index] = position % folds" not in script


def test_a2_temperature_improves_calibration_without_classification_drop() -> None:
    checkpoint = _checkpoint()
    a2 = checkpoint["A2"]
    assert isinstance(a2, dict)
    raw = a2["raw"]
    candidates = a2["candidates"]
    assert isinstance(raw, dict)
    assert isinstance(candidates, dict)
    temperature = candidates["temperature_scaling"]
    assert isinstance(temperature, dict)

    assert temperature["passes_guardrails"] is True
    assert float(temperature["macro_f1"]) == float(raw["macro_f1"])
    assert float(temperature["top3_recall"]) == float(raw["top3_recall"])
    assert float(temperature["expected_calibration_error_15bin"]) < 0.02
    assert float(temperature["multiclass_brier_score"]) < float(raw["multiclass_brier_score"])
    assert float(temperature["negative_log_likelihood"]) < float(raw["negative_log_likelihood"])


def test_a2_alternative_calibrators_fail_frozen_guardrails() -> None:
    checkpoint = _checkpoint()
    a2 = checkpoint["A2"]
    assert isinstance(a2, dict)
    candidates = a2["candidates"]
    assert isinstance(candidates, dict)
    isotonic = candidates["isotonic_regression"]
    platt = candidates["platt_scaling"]
    assert isinstance(isotonic, dict)
    assert isinstance(platt, dict)
    assert isotonic["passes_guardrails"] is False
    assert platt["passes_guardrails"] is False


def test_calibration_reproducibility_claim_is_bounded_not_bitwise() -> None:
    checkpoint = _checkpoint()
    reproducibility = checkpoint["reproducibility"]
    assert isinstance(reproducibility, dict)
    assert reproducibility["sample_order_identical"] is True
    assert reproducibility["fold_assignment_identical"] is True
    assert reproducibility["raw_predicted_intents_identical"] is True
    assert reproducibility["calibrated_predicted_intents_identical"] is True
    assert reproducibility["discrete_classification_metrics_identical"] is True
    assert reproducibility["selective_risk_curves_identical"] is True
    assert reproducibility["bitwise_probability_identity_required"] is False
    assert float(reproducibility["A2_max_calibrated_confidence_delta"]) < 2e-6


def test_calibration_environment_and_workflow_are_read_only() -> None:
    lock = LOCK_PATH.read_text(encoding="utf-8").lower()
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in lock
    assert 'name = "nvidia' not in lock
    assert 'name = "cuda' not in lock
    assert 'name = "triton"' not in lock
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "git push" not in workflow


def test_calibration_checkpoint_does_not_overclaim_h3_or_threshold() -> None:
    checkpoint = _checkpoint()
    interpretation = checkpoint["interpretation"]
    assert isinstance(interpretation, dict)
    assert interpretation["H3_status"] == "not_yet_evaluated"
    assert interpretation["threshold_status"] == "not_selected"
    assert interpretation["release_claim_status"] == "pending_confirmatory_evaluation"

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "models" / "routing_calibration.json"
SCRIPT_PATH = ROOT / "benchmarks" / "routing" / "evaluate_calibration.py"


def _load() -> dict[str, object]:
    return cast(dict[str, object], json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def test_calibration_eligible_models_and_methods_are_frozen() -> None:
    config = _load()
    assert config["eligible_models"] == ["A1", "A2"]
    excluded = config["excluded_models"]
    methods = config["methods"]
    assert isinstance(excluded, dict)
    assert isinstance(methods, list)
    assert "A3" in excluded
    assert [method["id"] for method in methods if isinstance(method, dict)] == [
        "temperature_scaling",
        "isotonic_regression",
        "platt_scaling",
    ]


def test_calibration_uses_validation_cross_fitting() -> None:
    config = _load()
    cross_fitting = config["cross_fitting"]
    assert isinstance(cross_fitting, dict)
    assert cross_fitting["enabled"] is True
    assert cross_fitting["folds"] == 5
    assert cross_fitting["stratification"] == "intent"
    assert cross_fitting["fit_folds_per_iteration"] == 4
    assert cross_fitting["score_fold_per_iteration"] == 1
    assert cross_fitting["full_validation_refit_only_after_method_selection"] is True


def test_calibration_selection_rule_and_guardrails_are_frozen() -> None:
    config = _load()
    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    guardrails = evaluation["classification_guardrails"]
    assert isinstance(guardrails, dict)
    assert evaluation["primary_selection_metric"] == "multiclass_brier_score"
    assert evaluation["secondary_selection_metrics"] == [
        "negative_log_likelihood",
        "expected_calibration_error_15bin",
    ]
    assert guardrails["max_macro_f1_drop_vs_raw"] == 0.005
    assert guardrails["max_top3_recall_drop_vs_raw"] == 0.002
    assert evaluation["operating_threshold_selected_here"] is False
    assert evaluation["routing_cost_used_for_method_selection_here"] is False


def test_calibration_keeps_test_sealed_and_a3_excluded() -> None:
    config = _load()
    data = config["data"]
    anti_shopping = config["anti_shopping"]
    assert isinstance(data, dict)
    assert isinstance(anti_shopping, dict)
    assert data["confirmatory_partition"] == "test"
    assert data["test_set_may_select_calibrator"] is False
    assert data["test_set_may_select_threshold"] is False
    assert anti_shopping["A3_reentry_allowed"] is False

    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "spec.test_url" not in script
    assert "Confirmatory test opened: false" in script


def test_calibration_declares_cpu_only_locked_environment() -> None:
    config = _load()
    dependency = config["dependency_policy"]
    assert isinstance(dependency, dict)
    assert dependency["torch_variant"] == "cpu_only"
    assert dependency["torch_index"] == "https://download.pytorch.org/whl/cpu"
    assert dependency["script_lock_required"] is True

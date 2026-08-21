from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a44c_v1.json"
EXECUTION = ROOT / "benchmarks" / "assistance" / "calibrate_grounding_a44c.py"
VERIFY = ROOT / "benchmarks" / "assistance" / "verify_calibration_a44c.py"
WORKFLOW = ROOT / ".github" / "workflows" / "phase4-assistance-a44c.yml"


def _config() -> dict[str, object]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a44c_scope_is_calibration_only() -> None:
    config = _config()
    scope = config["scope"]
    assert isinstance(scope, dict)
    assert scope["calibration_case_rows"] == 288
    assert scope["calibration_semantic_pair_rows"] == 491
    assert scope["validation_case_rows_authorized"] == 0
    assert scope["validation_semantic_pair_rows_authorized"] == 0
    assert scope["validation_metrics_authorized"] == 0
    assert scope["candidate_rows_authorized"] == 0
    assert scope["confirmatory_query_rows_authorized"] == 0


def test_a44c_temperature_grid_is_frozen() -> None:
    config = _config()
    calibration = config["calibration"]
    assert isinstance(calibration, dict)
    assert calibration["parameter"] == "single_global_temperature"
    assert calibration["objective"] == "three_class_negative_log_likelihood"
    assert calibration["grid_start"] == 0.25
    assert calibration["grid_stop"] == 4.0
    assert calibration["grid_step"] == 0.01
    assert calibration["grid_points"] == 376
    assert calibration["tie_break"] == "smallest_temperature"
    assert calibration["may_change_raw_argmax_class"] is False
    assert calibration["may_change_final_grounding_verdict"] is False


def test_a44c_gold_pair_arithmetic_is_registered() -> None:
    config = _config()
    scope = config["scope"]
    assert isinstance(scope, dict)
    counts = scope["calibration_gold_relation_counts"]
    assert counts == {"ENTAILED": 211, "CONTRADICTED": 40, "UNKNOWN": 240}
    assert sum(counts.values()) == 491


def test_a44c_execution_and_verifier_parse_without_running_models() -> None:
    ast.parse(EXECUTION.read_text(encoding="utf-8"))
    ast.parse(VERIFY.read_text(encoding="utf-8"))


def test_a44c_workflow_is_push_to_main_one_shot_path_scoped() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "branches: [main]" in workflow
    assert '".github/workflows/phase4-assistance-a44c.yml"' in workflow
    assert "workflow_dispatch" not in workflow
    assert "calibrate_grounding_a44c.py" in workflow
    assert "verify_calibration_a44c.py" in workflow

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "routing" / "evaluate_confirmatory.py"
CONFIG = ROOT / "configs" / "models" / "routing_confirmatory_v1.json"
SELECTED = ROOT / "configs" / "models" / "routing_selected_v1.json"
WORKFLOW = ROOT / ".github" / "workflows" / "phase2-routing-confirmatory.yml"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("test_phase2_confirmatory_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load confirmatory evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def test_confirmatory_protocol_matches_frozen_selected_router() -> None:
    config = _json(CONFIG)
    selected = _json(SELECTED)
    model = config["model"]
    calibration = config["calibration"]
    policies = config["frozen_policies"]
    governance = config["governance"]

    assert isinstance(model, dict)
    assert isinstance(calibration, dict)
    assert isinstance(policies, dict)
    assert isinstance(governance, dict)
    assert model["id"] == selected["model"]["id"]  # type: ignore[index]
    assert model["encoder_revision"] == selected["model"]["encoder_revision"]  # type: ignore[index]
    assert calibration["temperature"] == selected["calibration"]["temperature"]  # type: ignore[index]
    assert policies["A2_temperature_threshold"] == selected["operating_policy"]["threshold"]  # type: ignore[index]
    assert policies["A2_raw_threshold"] == 0.367217
    assert governance["one_registered_scientific_result"] is True
    assert governance["test_result_may_change_model"] is False
    assert governance["test_result_may_change_calibration"] is False
    assert governance["test_result_may_change_threshold"] is False
    assert governance["phase3_may_start_before_phase2_close"] is False


def test_preflight_does_not_download_or_open_test(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()

    def forbidden_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight attempted network download")

    monkeypatch.setattr(module, "_download", forbidden_download)
    result = module.preflight()
    assert result["status"] == "preflight_passed"
    assert result["test_set_opened"] is False


def test_invalid_authorization_stops_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()

    def forbidden_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid authorization reached network download")

    monkeypatch.setattr(module, "_download", forbidden_download)
    with pytest.raises(PermissionError, match="authorization token"):
        module.run(tmp_path, "WRONG_TOKEN")


def test_authorization_guard_precedes_test_url_access_in_source() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    guard = source.index("if authorization != AUTHORIZATION_TOKEN")
    test_access = source.index("_download(spec.test_url, test_csv)")
    assert guard < test_access


def test_hypothesis_verdict_rule_is_frozen_and_directional() -> None:
    module = _module()
    assert module._verdict([-0.20, -0.01]) == "supported"
    assert module._verdict([0.00, 0.15]) == "unsupported"
    assert module._verdict([-0.01, 0.03]) == "inconclusive"


def test_high_risk_error_overrides_queue_similarity() -> None:
    module = _module()
    operations = {
        "true_high": {"high_risk": True, "queue": "same"},
        "wrong_same": {"high_risk": False, "queue": "same"},
        "wrong_other": {"high_risk": False, "queue": "other"},
    }
    assert module._event("true_high", "wrong_same", operations) == "unsafe_high_risk_auto_route"
    assert module._event("wrong_same", "true_high", operations) == "wrong_intent_same_queue"
    assert module._event("wrong_same", "wrong_other", operations) == "wrong_queue"


def test_fixed_coverage_accepts_exact_registered_fraction() -> None:
    module = _module()
    errors = module.np.asarray([False, True, False, True, False, True, False, True], dtype=bool)
    confidence = module.np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    sample_ids = [f"id-{index}" for index in range(8)]
    result = module._risk_at_coverage(errors, confidence, sample_ids, 0.75)
    assert result["accepted"] == 6
    assert result["coverage"] == 0.75
    assert result["selective_risk"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_on_synthetic_rows() -> None:
    module = _module()
    raw_costs = module.np.asarray([0.0, 6.0, 1.0, 2.5, 0.0, 1.0])
    calibrated_costs = module.np.asarray([0.0, 1.0, 1.0, 2.5, 0.0, 1.0])
    errors = module.np.asarray([False, True, False, True, False, False], dtype=bool)
    confidence = module.np.asarray([0.95, 0.90, 0.85, 0.70, 0.60, 0.40])
    sample_ids = [f"sample-{index}" for index in range(6)]
    first = module._paired_bootstrap(
        raw_costs,
        calibrated_costs,
        errors,
        confidence,
        sample_ids,
        0.75,
        100,
        20260819,
        0.95,
    )
    second = module._paired_bootstrap(
        raw_costs,
        calibrated_costs,
        errors,
        confidence,
        sample_ids,
        0.75,
        100,
        20260819,
        0.95,
    )
    assert first == second


def test_confirmatory_workflow_is_manual_and_read_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "OPEN_FROZEN_TEST_ONCE" in workflow
    assert "--authorize-test-open" in workflow

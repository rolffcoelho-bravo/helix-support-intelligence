from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "routing" / "evaluate_confirmatory.py"
CONFIG = ROOT / "configs" / "models" / "routing_confirmatory_v1.json"
SELECTED = ROOT / "configs" / "models" / "routing_selected_v1.json"
RESULT = ROOT / "benchmarks" / "routing" / "results" / "confirmatory_test_v1.json"
AUDIT = ROOT / "benchmarks" / "routing" / "results" / "confirmatory_post_audit_v1.json"
CONSUMED_WORKFLOWS = (
    ROOT / ".github" / "workflows" / "phase2-routing-confirmatory.yml",
    ROOT / ".github" / "workflows" / "phase2-routing-confirmatory-preflight.yml",
    ROOT / ".github" / "workflows" / "phase2-routing-confirmatory-approval-bridge.yml",
)


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
    h3 = config["H3_confirmatory"]

    assert isinstance(model, dict)
    assert isinstance(calibration, dict)
    assert isinstance(policies, dict)
    assert isinstance(governance, dict)
    assert isinstance(h3, dict)
    assert model["id"] == selected["model"]["id"]  # type: ignore[index]
    assert model["encoder_revision"] == selected["model"]["encoder_revision"]  # type: ignore[index]
    assert calibration["temperature"] == selected["calibration"]["temperature"]  # type: ignore[index]
    assert policies["A2_temperature_threshold"] == selected["operating_policy"]["threshold"]  # type: ignore[index]
    assert policies["A2_raw_threshold"] == 0.367217
    assert governance["one_registered_scientific_result"] is True
    assert governance["test_result_may_change_model"] is False
    assert governance["test_result_may_change_calibration"] is False
    assert governance["test_result_may_change_threshold"] is False
    assert governance["H3_full_mixed_endpoint_has_independent_confirmatory_sample"] is False
    assert h3["scope"] == "independent_in_domain_component_only"
    assert h3["full_original_mixed_endpoint_confirmed_by_this_test"] is False


def test_authorization_guard_precedes_test_url_access_in_source() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    guard = source.index("if authorization != AUTHORIZATION_TOKEN")
    test_access = source.index("_download(spec.test_url, test_csv)")
    assert guard < test_access
    assert 'AUTHORIZATION_TOKEN = "OPEN_FROZEN_TEST_ONCE"' in source
    assert '"test_set_opened": True' in source
    assert '"H3_full_mixed_endpoint_claimed_confirmed": False' in source


def test_default_cli_path_remains_preflight_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "if args.preflight_only or args.authorize_test_open is None:" in source
    assert 'print("Confirmatory test opened: false")' in source
    assert "result = run(args.output_dir.resolve(), args.authorize_test_open)" in source


def test_permanent_confirmatory_result_is_frozen_and_audited() -> None:
    result = _json(RESULT)
    audit = _json(AUDIT)
    execution = result["execution"]
    h3 = result["H3_independent_in_domain_component"]
    h4 = result["H4_confirmatory"]
    governance = result["governance"]

    assert isinstance(execution, dict)
    assert isinstance(h3, dict)
    assert isinstance(h4, dict)
    assert isinstance(governance, dict)
    assert result["status"] == "registered_confirmatory_result"
    assert execution["workflow_run_id"] == 32243835846
    assert execution["one_scientific_run"] is True
    assert execution["later_observer_comments_triggered_scientific_rerun"] is False
    assert h3["verdict"] == "inconclusive"
    assert h3["full_original_mixed_endpoint_confirmed_by_this_test"] is False
    assert h4["verdict"] == "supported"
    assert governance["model_changed_after_test"] is False
    assert governance["calibration_changed_after_test"] is False
    assert governance["threshold_changed_after_test"] is False
    assert audit["status"] == "passed"


def test_consumed_confirmatory_execution_workflows_are_not_mergeable_runtime_surface() -> None:
    for workflow in CONSUMED_WORKFLOWS:
        assert not workflow.exists(), f"consumed one-shot workflow still active: {workflow}"

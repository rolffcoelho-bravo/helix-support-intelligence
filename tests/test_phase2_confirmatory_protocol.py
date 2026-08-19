from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "routing" / "evaluate_confirmatory.py"
CONFIG = ROOT / "configs" / "models" / "routing_confirmatory_v1.json"
SELECTED = ROOT / "configs" / "models" / "routing_selected_v1.json"
WORKFLOW = ROOT / ".github" / "workflows" / "phase2-routing-confirmatory.yml"
BRIDGE = ROOT / ".github" / "workflows" / "phase2-routing-confirmatory-approval-bridge.yml"


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
    assert governance["unconditional_automatic_test_trigger_allowed"] is False
    assert governance["approval_bridge_requires_exact_comment"] == "OPEN_FROZEN_TEST_ONCE"
    assert governance["approval_bridge_scientific_logic_changed"] is False
    assert governance["test_result_may_change_model"] is False
    assert governance["test_result_may_change_calibration"] is False
    assert governance["test_result_may_change_threshold"] is False
    assert governance["H3_full_mixed_endpoint_has_independent_confirmatory_sample"] is False
    assert governance["phase3_may_start_before_phase2_close"] is False
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


def test_default_cli_path_is_preflight_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "if args.preflight_only or args.authorize_test_open is None:" in source
    assert 'print("Confirmatory test opened: false")' in source
    assert "result = run(args.output_dir.resolve(), args.authorize_test_open)" in source


def test_confirmatory_workflow_is_manual_and_read_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "OPEN_FROZEN_TEST_ONCE" in workflow
    assert "--authorize-test-open" in workflow
    freeze = workflow.index("Verify pre-confirmatory artifact freeze")
    preflight = workflow.index("Run no-test preflight")
    test_open = workflow.index("Open frozen test and run registered confirmatory evaluation")
    assert freeze < preflight < test_open


def test_explicit_approval_bridge_preserves_frozen_scientific_path() -> None:
    bridge = BRIDGE.read_text(encoding="utf-8")
    assert "issue_comment:" in bridge
    assert "types: [created]" in bridge
    assert "github.event.issue.number == 6" in bridge
    assert "github.event.comment.body == 'OPEN_FROZEN_TEST_ONCE'" in bridge
    assert "ref: phase-2-routing-baseline-selective-policy" in bridge
    assert "contents: read" in bridge
    assert "contents: write" not in bridge
    assert "push:" not in bridge
    assert "pull_request:" not in bridge
    assert "python benchmarks/routing/verify_preconfirmatory_freeze.py" in bridge
    assert "--preflight-only" in bridge
    assert "--authorize-test-open \"${{ github.event.comment.body }}\"" in bridge
    freeze = bridge.index("Verify pre-confirmatory artifact freeze")
    preflight = bridge.index("Run no-test preflight")
    test_open = bridge.index("Open frozen test and run registered confirmatory evaluation")
    assert freeze < preflight < test_open

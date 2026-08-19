from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "configs" / "models" / "routing_selected_v1.json"
COST_RESULT = ROOT / "benchmarks" / "routing" / "results" / "cost_policy_validation_v1.json"
COST_WORKFLOW = ROOT / ".github" / "workflows" / "phase2-routing-cost-policy.yml"


def _json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def test_selected_router_is_frozen_a2_temperature_policy() -> None:
    selected = _json(SELECTED)
    model = selected["model"]
    calibration = selected["calibration"]
    policy = selected["operating_policy"]

    assert isinstance(model, dict)
    assert model["id"] == "A2"
    assert model["encoder_revision"] == "c315f904dfc467d8b9c40ab4ed50b3a8d0866c15"

    assert isinstance(calibration, dict)
    assert calibration["method"] == "temperature_scaling"
    assert calibration["temperature"] == 0.457974

    assert isinstance(policy, dict)
    assert policy["auto_route_when"] == "confidence >= threshold"
    assert policy["otherwise"] == "ESCALATE_LOW_CONFIDENCE"
    assert policy["threshold"] == 0.892704

    plateau = policy["audited_full_refit_plateau"]
    assert isinstance(plateau, dict)
    assert float(plateau["max_rejected_confidence"]) < float(policy["threshold"])
    assert float(policy["threshold"]) <= float(plateau["min_accepted_confidence"])


def test_selected_router_cannot_be_moved_by_confirmatory_test() -> None:
    selected = _json(SELECTED)
    governance = selected["governance"]
    hypotheses = selected["hypotheses"]

    assert isinstance(governance, dict)
    assert governance["confirmatory_test_opened"] is False
    assert governance["confirmatory_test_may_change_model"] is False
    assert governance["confirmatory_test_may_change_calibration"] is False
    assert governance["confirmatory_test_may_change_threshold"] is False

    assert isinstance(hypotheses, dict)
    assert hypotheses["H3_development"] == "unsupported_for_A2_primary_expected_cost"
    assert hypotheses["H4_development"] == "supported"
    assert hypotheses["confirmatory_status"] == "pending"


def test_cost_checkpoint_preserves_negative_h3_and_positive_h4() -> None:
    result = _json(COST_RESULT)
    h3 = result["H3_development"]
    h4 = result["H4_development"]
    selection = result["selection"]
    transfer = result["full_refit_transfer_audit"]

    assert result["test_set_opened"] is False

    assert isinstance(h3, dict)
    assert h3["status"] == "unsupported_for_A2_primary_expected_cost"
    assert float(h3["temperature_minus_raw"]) > 0.0

    assert isinstance(h4, dict)
    assert h4["status"] == "supported"
    assert float(h4["expected_cost_reduction_vs_full_automation"]) > 0.0
    assert float(h4["selective_risk_reduction_vs_full_automation"]) > 0.0

    assert isinstance(selection, dict)
    assert selection["frozen_calibrated_development_candidate"] == "A2_temperature"
    assert selection["stable_across_registered_calibrated_sensitivity"] is True
    assert selection["A2_temperature_wins_registered_sensitivity_cells"] == 9

    assert isinstance(transfer, dict)
    assert transfer["independent_rerun_byte_identical"] is True
    assert transfer["full_refit_accepted_rows"] == 1482
    assert transfer["changed_acceptance_decisions"] == 6
    assert transfer["frozen_deployment_threshold"] == 0.892704


def test_cost_workflow_is_read_only_after_checkpoint() -> None:
    workflow = COST_WORKFLOW.read_text(encoding="utf-8")
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "git push" not in workflow
    assert "Confirmatory test" not in workflow

"""Phase 2 routing-protocol invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
ROUTING_LADDER = ROOT / "configs" / "models" / "routing_ladder.json"


def _load() -> dict[str, Any]:
    payload = json.loads(ROUTING_LADDER.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_phase2_model_ladder_is_bounded() -> None:
    payload = _load()
    models = cast(list[dict[str, Any]], payload["models"])
    by_id = {str(model["id"]): model for model in models}

    assert list(by_id) == ["A0", "A1", "A2", "A3", "A4"]
    assert all(bool(by_id[model_id]["required"]) for model_id in ("A0", "A1", "A2", "A3"))
    assert by_id["A4"]["required"] is False
    assert by_id["A4"]["enabled"] is False


def test_phase2_never_selects_on_confirmatory_test() -> None:
    payload = _load()
    decision_policy = cast(dict[str, Any], payload["decision_policy"])

    assert payload["selection_partition"] == "validation"
    assert payload["confirmatory_partition"] == "test"
    assert decision_policy["threshold_source"] == "validation_only"
    assert decision_policy["test_set_may_select_threshold"] is False


def test_phase2_required_measurements_are_frozen() -> None:
    payload = _load()
    metrics = set(cast(list[str], payload["required_metrics"]))

    assert metrics == {
        "macro_f1",
        "balanced_accuracy",
        "top3_recall",
        "expected_calibration_error",
        "brier_score",
        "oos_auroc",
        "oos_fpr_at_declared_recall",
        "expected_routing_cost",
        "risk_coverage",
    }


def test_phase2_cost_events_include_safe_abstention() -> None:
    payload = _load()
    cost_events = set(cast(list[str], payload["cost_events"]))

    assert "unsafe_high_risk_auto_route" in cost_events
    assert "human_escalation" in cost_events
    assert payload["decision_policy"]["abstention_decision"] == "ESCALATE_LOW_CONFIDENCE"

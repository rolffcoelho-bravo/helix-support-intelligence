from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"


def test_a44b_is_single_nomination_without_empirical_optimality_claim() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selection = config["selection_policy"]

    assert (
        selection["binding_semantics"]
        == "single_pre_registered_nomination_not_empirical_optimum"
    )
    assert selection["empirical_optimality_claimed"] is False
    assert selection["failure_replacement_allowed_under_same_binding"] is False
    assert selection["replacement_requires_new_versioned_binding"] is True
    assert selection["empirical_candidate_bakeoff_performed"] is False
    assert selection["a44a_calibration_cases_opened"] is False
    assert selection["a44a_validation_cases_opened"] is False

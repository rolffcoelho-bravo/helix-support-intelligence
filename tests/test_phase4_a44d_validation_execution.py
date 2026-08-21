"""Registration tests for A4.4d sealed validation execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a44d_registration_is_validation_only_and_frozen() -> None:
    a44a = _load(ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json")
    a44d = _load(ROOT / "configs" / "models" / "assistance_grounding_a44d_v1.json")
    assert a44d["checkpoint"] == "A4.4d"
    assert a44d["scope"]["validation_case_rows"] == 144
    assert a44d["scope"]["validation_intents"] == 20
    assert a44d["scope"]["validation_semantic_pair_rows"] == 246
    assert a44d["scope"]["calibration_case_rows_authorized"] == 0
    assert a44d["scope"]["candidate_rows_authorized"] == 0
    assert a44d["scope"]["confirmatory_query_rows_authorized"] == 0
    assert a44d["frozen_calibration"]["selected_temperature"] == 3.67
    assert a44d["validation_requirements"] == a44a["future_validation_requirements"]


def test_a44d_forbids_rescue_and_post_validation_tuning() -> None:
    a44d = _load(ROOT / "configs" / "models" / "assistance_grounding_a44d_v1.json")
    forbidden = a44d["forbidden"]
    assert all(
        forbidden[name] is True
        for name in (
            "temperature_refit",
            "threshold_search",
            "class_specific_thresholds",
            "abstention_search",
            "model_substitution",
            "model_fine_tuning",
            "prompt_tuning",
            "candidate_inference",
            "candidate_comparison",
            "confirmatory_inference",
            "post_result_rescue",
        )
    )


def test_a44d_preflight_does_not_materialize_validation_cases() -> None:
    source = (ROOT / "scripts" / "preflight_phase4_a44d.py").read_text(encoding="utf-8")
    assert "generate_validation_cases" not in source
    assert "validation_cases_a44d" not in source
    assert "AutoModel" not in source
    assert "snapshot_download" not in source


def test_a44d_workflow_is_main_push_one_shot_surface() -> None:
    source = (ROOT / ".github" / "workflows" / "phase4-assistance-a44d.yml").read_text(
        encoding="utf-8"
    )
    assert "push:" in source
    assert "branches: [main]" in source
    assert '".github/workflows/phase4-assistance-a44d.yml"' in source
    assert "pull_request:" not in source
    assert "workflow_dispatch:" not in source

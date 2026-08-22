"""Static tests for deterministic A4.5b recovery from the immutable partial artifact."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "benchmarks" / "assistance" / "finalize_calibration_a45b_recovery.py"
VERIFIER = ROOT / "benchmarks" / "assistance" / "verify_calibration_a45b_recovery.py"
PROVENANCE = ROOT / "benchmarks" / "assistance" / "results" / "a45b_partial_attempt1_v1.json"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_requirement_name_dispatch_is_mechanical() -> None:
    module = _load(FINALIZER, "finalize_calibration_a45b_recovery_test")
    assert module.registered_metric_name("relevance_macro_f1_min") == (
        "relevance_macro_f1",
        "min",
    )
    assert module.registered_metric_name("sufficiency_macro_f1_min_on_relevant_pairs") == (
        "sufficiency_macro_f1_on_relevant_pairs",
        "min",
    )
    assert module.registered_metric_name("polarity_macro_f1_min_on_relevant_sufficient_pairs") == (
        "polarity_macro_f1_on_relevant_sufficient_pairs",
        "min",
    )
    assert module.registered_metric_name(
        "cross_document_irrelevance_false_contradiction_rate_max"
    ) == ("cross_document_irrelevance_false_contradiction_rate", "max")


def test_partial_artifact_boundary_is_frozen() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["workflow_run_id"] == 32581433921
    assert provenance["artifact"]["id"] == 9477913279
    assert provenance["score_rows"] == 360
    assert provenance["unique_pair_ids"] == 360
    assert provenance["score_splits"] == ["calibration"]
    assert provenance["threshold_selected"] is False
    assert provenance["scientific_pass_fail_computed"] is False
    assert provenance["validation_rows_scored"] == 0
    assert provenance["confirmatory_queries_scored"] == 0


def test_recovery_has_no_model_loading_surface() -> None:
    for path in (FINALIZER, VERIFIER):
        source = path.read_text(encoding="utf-8")
        assert "from transformers" not in source
        assert "import transformers" not in source
        assert "hf_hub_download" not in source
        assert "calibrate_aerf_a45b.py" not in source

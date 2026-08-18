from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "benchmarks" / "routing" / "verify_preconfirmatory_freeze.py"
MANIFEST = ROOT / "configs" / "models" / "routing_preconfirmatory_manifest_v1.json"
WORKFLOW = ROOT / ".github" / "workflows" / "phase2-routing-confirmatory.yml"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("test_phase2_preconfirmatory_verifier", VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pre-confirmatory verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict[str, object]:
    payload: object = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("pre-confirmatory manifest must be an object")
    return cast(dict[str, object], payload)


def test_preconfirmatory_manifest_verifies_current_repository_bytes() -> None:
    module = _module()
    result = module.verify()
    assert result["status"] == "preconfirmatory_freeze_verified"
    assert result["test_set_opened"] is False
    assert int(result["artifacts_checked"]) >= 30


def test_manifest_freezes_selected_scientific_and_execution_surface() -> None:
    manifest = _manifest()
    artifacts = manifest["artifacts"]
    governance = manifest["governance"]
    assert isinstance(artifacts, dict)
    assert isinstance(governance, dict)

    required = {
        "configs/data/banking77.json",
        "configs/models/routing_a2.json",
        "configs/models/routing_calibration.json",
        "configs/models/routing_oos.json",
        "configs/models/routing_cost_matrix.json",
        "configs/models/routing_selected_v1.json",
        "configs/models/routing_confirmatory_v1.json",
        "data/oos/routing_oos_v1.json",
        "benchmarks/routing/evaluate_calibration.py",
        "benchmarks/routing/evaluate_confirmatory.py",
        "benchmarks/routing/verify_preconfirmatory_freeze.py",
        "benchmarks/routing/results/cost_policy_validation_v1.json",
        ".github/workflows/phase2-routing-confirmatory.yml",
    }
    assert required <= set(artifacts)
    assert manifest["test_set_opened"] is False
    assert governance["artifact_hash_mismatch_blocks_test_open"] is True
    assert governance["confirmatory_test_may_change_frozen_artifacts"] is False
    assert governance["phase3_may_start_before_phase2_close"] is False
    assert governance["oos_benchmark_is_independent_confirmatory_evidence"] is False


def test_confirmatory_workflow_checks_freeze_before_preflight_and_test_access() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    freeze = workflow.index("Verify pre-confirmatory artifact freeze")
    preflight = workflow.index("Run no-test preflight")
    test_open = workflow.index("Open frozen test and run registered confirmatory evaluation")
    assert freeze < preflight < test_open
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow

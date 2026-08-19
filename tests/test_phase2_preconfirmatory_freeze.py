from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "models" / "routing_preconfirmatory_manifest_v1.json"
RESULT = ROOT / "benchmarks" / "routing" / "results" / "confirmatory_test_v1.json"
AUDIT = ROOT / "benchmarks" / "routing" / "results" / "confirmatory_post_audit_v1.json"


def _json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def test_preconfirmatory_manifest_is_preserved_as_historical_freeze_record() -> None:
    manifest = _json(MANIFEST)
    artifacts = manifest["artifacts"]
    governance = manifest["governance"]
    assert isinstance(artifacts, dict)
    assert isinstance(governance, dict)

    assert manifest["status"] == "frozen_before_confirmatory_test_open"
    assert manifest["test_set_opened"] is False
    assert len(artifacts) == 36
    assert governance["artifact_hash_mismatch_blocks_test_open"] is True
    assert governance["confirmatory_test_may_change_frozen_artifacts"] is False
    assert governance["oos_benchmark_is_independent_confirmatory_evidence"] is False
    assert governance["approval_bridge_is_authorization_transport_only"] is True


def test_manifest_freezes_selected_scientific_and_historical_execution_surface() -> None:
    manifest = _json(MANIFEST)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)

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
        ".github/workflows/phase2-routing-confirmatory-preflight.yml",
        ".github/workflows/phase2-routing-confirmatory-approval-bridge.yml",
    }
    assert required <= set(artifacts)
    assert artifacts[".github/workflows/phase2-routing-confirmatory-approval-bridge.yml"] == (
        "2ccb726ded479fb51c68647f2b344a95393cd878"
    )


def test_consumed_freeze_is_linked_to_permanent_confirmatory_result_and_audit() -> None:
    result = _json(RESULT)
    audit = _json(AUDIT)
    execution = result["execution"]
    data = result["data"]
    governance = result["governance"]

    assert isinstance(execution, dict)
    assert isinstance(data, dict)
    assert isinstance(governance, dict)
    assert result["status"] == "registered_confirmatory_result"
    assert execution["workflow_run_id"] == 32243835846
    assert execution["frozen_branch_head_at_execution"] == (
        "9f69bfc8d8e7f5520bee49cb6e9c8770fa20595a"
    )
    assert execution["one_scientific_run"] is True
    assert data["test_set_opened"] is True
    assert governance["model_changed_after_test"] is False
    assert governance["calibration_changed_after_test"] is False
    assert governance["threshold_changed_after_test"] is False
    assert audit["status"] == "passed"

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "models" / "routing_a2.json"
LOCK_PATH = ROOT / "benchmarks" / "routing" / "evaluate_a2.py.lock"
CHECKPOINT_PATH = ROOT / "benchmarks" / "routing" / "results" / "a2_validation_v2.json"
SUPERSEDED_JSON = ROOT / "benchmarks" / "routing" / "results" / "a2_validation_v1.json"
SUPERSEDED_MD = ROOT / "benchmarks" / "routing" / "results" / "a2_validation_v1.md"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a2_declared_cpu_environment_matches_lock() -> None:
    config = _load_json(CONFIG_PATH)
    representation = config["representation"]
    dependency_policy = config["dependency_policy"]
    assert isinstance(representation, dict)
    assert isinstance(dependency_policy, dict)

    assert representation["device"] == "cpu"
    assert dependency_policy["torch_variant"] == "cpu_only"
    assert dependency_policy["torch_index"] == "https://download.pytorch.org/whl/cpu"

    lock = LOCK_PATH.read_text(encoding="utf-8").lower()
    assert "https://download.pytorch.org/whl/cpu" in lock
    assert 'name = "torch"' in lock
    assert 'name = "nvidia' not in lock
    assert 'name = "cuda' not in lock
    assert 'name = "triton"' not in lock


def test_a2_checkpoint_uses_audited_reproducibility_claims() -> None:
    checkpoint = _load_json(CHECKPOINT_PATH)
    environment = checkpoint["environment"]
    reproducibility = checkpoint["reproducibility"]
    assert isinstance(environment, dict)
    assert isinstance(reproducibility, dict)

    assert checkpoint["checkpoint_id"] == "phase2-a2-validation-v2"
    assert checkpoint["test_set_opened"] is False
    assert environment["torch"] == "2.13.0+cpu"
    assert environment["torch_cuda_available"] is False
    assert environment["cuda_or_nvidia_packages_in_lock"] is False

    assert reproducibility["bitwise_identity_required"] is False
    assert reproducibility["bitwise_identity_observed"] is False
    assert reproducibility["sample_order_identical"] is True
    assert reproducibility["predicted_intents_identical"] is True
    assert reproducibility["top3_intents_identical"] is True
    assert reproducibility["selective_risk_curve_identical"] is True
    assert reproducibility["discrete_metrics_identical"] is True
    assert float(reproducibility["max_abs_confidence_delta"]) < 1e-5


def test_a2_risk_coverage_dominates_a1_on_registered_grid() -> None:
    checkpoint = _load_json(CHECKPOINT_PATH)
    rows = checkpoint["risk_coverage"]
    assert isinstance(rows, list)
    assert len(rows) == 10

    for row in rows:
        assert isinstance(row, dict)
        assert float(row["a2_selective_risk"]) <= float(row["a1_selective_risk"])


def test_a2_confusion_audit_uses_full_counts_for_a1_pairs() -> None:
    checkpoint = _load_json(CHECKPOINT_PATH)
    rows = checkpoint["a1_top_confusion_pair_changes"]
    assert isinstance(rows, list)
    counts = {
        (str(row["true_intent"]), str(row["predicted_intent"])): (
            int(row["a1_count"]),
            int(row["a2_count"]),
        )
        for row in rows
        if isinstance(row, dict)
    }

    assert counts[("compromised_card", "lost_or_stolen_card")] == (4, 1)
    assert counts[("pending_cash_withdrawal", "declined_cash_withdrawal")] == (3, 1)
    assert counts[("top_up_failed", "top_up_reverted")] == (4, 0)
    assert counts[("declined_transfer", "failed_transfer")] == (4, 6)


def test_superseded_a2_v1_evidence_is_removed() -> None:
    assert not SUPERSEDED_JSON.exists()
    assert not SUPERSEDED_MD.exists()

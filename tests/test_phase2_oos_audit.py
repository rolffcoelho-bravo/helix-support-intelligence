from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "benchmarks" / "routing" / "results" / "oos_validation_v1.json"
BENCHMARK_PATH = ROOT / "data" / "oos" / "routing_oos_v1.json"
LOCK_PATH = ROOT / "benchmarks" / "routing" / "evaluate_oos.py.lock"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "phase2-routing-oos.yml"
SCRIPT_PATH = ROOT / "benchmarks" / "routing" / "evaluate_oos.py"


def _checkpoint() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")),
    )


def test_oos_checkpoint_prefers_a2_but_does_not_claim_oos_solved() -> None:
    checkpoint = _checkpoint()
    interpretation = checkpoint["interpretation"]
    assert isinstance(interpretation, dict)
    assert checkpoint["preferred_model"] == "A2"
    assert interpretation["oos_status"] == "not_solved"
    assert interpretation["threshold_status"] == "not_selected"
    assert checkpoint["test_set_opened"] is False


def test_a2_oos_primary_metrics_beat_a1_but_high_recall_fpr_stays_large() -> None:
    checkpoint = _checkpoint()
    models = checkpoint["models"]
    assert isinstance(models, dict)
    a1 = models["A1"]
    a2 = models["A2"]
    assert isinstance(a1, dict)
    assert isinstance(a2, dict)
    a1_primary = a1["primary_cross_fitted"]
    a2_primary = a2["primary_cross_fitted"]
    assert isinstance(a1_primary, dict)
    assert isinstance(a2_primary, dict)

    assert float(a2_primary["weighted_oos_auroc"]) > float(a1_primary["weighted_oos_auroc"])
    assert float(a2_primary["weighted_in_domain_fpr_at_declared_oos_recall"]) < float(
        a1_primary["weighted_in_domain_fpr_at_declared_oos_recall"]
    )
    assert float(a2_primary["weighted_in_domain_fpr_at_declared_oos_recall"]) > 0.40


def test_near_boundary_oos_is_reported_as_harder() -> None:
    checkpoint = _checkpoint()
    models = checkpoint["models"]
    assert isinstance(models, dict)
    a2 = models["A2"]
    assert isinstance(a2, dict)
    tiers = a2["tier_diagnostics"]
    categories = a2["category_diagnostics"]
    assert isinstance(tiers, dict)
    assert isinstance(categories, dict)

    near = tiers["near"]
    medium = tiers["medium"]
    assert isinstance(near, dict)
    assert isinstance(medium, dict)
    assert float(near["weighted_oos_auroc"]) < float(medium["weighted_oos_auroc"])
    direct_debit = categories["direct_debit_management"]
    assert isinstance(direct_debit, dict)
    assert float(direct_debit["weighted_oos_auroc"]) < 0.50


def test_raw_a2_oos_diagnostic_is_better_but_cannot_replace_primary_post_hoc() -> None:
    checkpoint = _checkpoint()
    models = checkpoint["models"]
    interpretation = checkpoint["interpretation"]
    assert isinstance(models, dict)
    assert isinstance(interpretation, dict)
    a2 = models["A2"]
    assert isinstance(a2, dict)
    raw = a2["raw_diagnostic"]
    primary = a2["primary_cross_fitted"]
    assert isinstance(raw, dict)
    assert isinstance(primary, dict)

    assert float(raw["oos_auroc"]) > float(primary["weighted_oos_auroc"])
    assert float(raw["in_domain_fpr_at_declared_oos_recall"]) < float(
        primary["weighted_in_domain_fpr_at_declared_oos_recall"]
    )
    assert "cannot replace" in str(interpretation["raw_vs_calibrated"])


def test_oos_benchmark_is_frozen_synthetic_and_has_no_exact_overlap() -> None:
    checkpoint = _checkpoint()
    construction = checkpoint["benchmark_construction"]
    benchmark = checkpoint["benchmark"]
    assert isinstance(construction, dict)
    assert isinstance(benchmark, dict)
    assert construction["frozen_before_model_scoring"] is True
    assert construction["kind"] == "hand_authored_support_like_synthetic_oos"
    assert construction["query_count"] == 160
    assert construction["category_count"] == 20
    assert benchmark["exact_normalized_overlap_with_banking77_source_train"] == 0
    assert BENCHMARK_PATH.exists()


def test_oos_reproducibility_claim_is_exact_for_primary_metrics_not_bitwise_floats() -> None:
    checkpoint = _checkpoint()
    reproducibility = checkpoint["reproducibility"]
    assert isinstance(reproducibility, dict)
    assert reproducibility["A1_primary_metrics_identical"] is True
    assert reproducibility["A2_primary_metrics_identical"] is True
    assert reproducibility["A1_tier_diagnostics_identical"] is True
    assert reproducibility["A2_tier_diagnostics_identical"] is True
    assert reproducibility["preferred_model_identical"] is True
    assert reproducibility["bitwise_probability_identity_required"] is False
    assert float(reproducibility["max_observed_numeric_delta_across_all_float_fields"]) < 1e-6


def test_oos_environment_workflow_and_test_boundary_are_locked() -> None:
    lock = LOCK_PATH.read_text(encoding="utf-8").lower()
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in lock
    assert 'name = "nvidia' not in lock
    assert 'name = "cuda' not in lock
    assert 'name = "triton"' not in lock
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "git push" not in workflow
    assert "spec.test_url" not in script
    assert "Confirmatory test opened: false" in script

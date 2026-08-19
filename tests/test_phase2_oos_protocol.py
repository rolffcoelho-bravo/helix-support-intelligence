from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "oos" / "routing_oos_v1.json"
CONFIG_PATH = ROOT / "configs" / "models" / "routing_oos.json"


def _load(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def test_oos_benchmark_shape_is_frozen() -> None:
    benchmark = _load(BENCHMARK_PATH)
    categories = benchmark["categories"]
    assert benchmark["version"] == "routing-oos-v1"
    assert benchmark["status"] == "frozen_before_model_scoring"
    assert benchmark["query_count"] == 160
    assert benchmark["category_count"] == 20
    assert isinstance(categories, list)
    assert len(categories) == 20

    texts: list[str] = []
    tier_counts: dict[str, int] = {}
    category_ids: list[str] = []
    for category in categories:
        assert isinstance(category, dict)
        category_id = str(category["id"])
        tier = str(category["tier"])
        queries = category["queries"]
        assert isinstance(queries, list)
        assert len(queries) == 8
        category_ids.append(category_id)
        tier_counts[tier] = tier_counts.get(tier, 0) + len(queries)
        texts.extend(str(query) for query in queries)

    assert len(set(category_ids)) == 20
    assert len(texts) == 160
    assert len(set(text.casefold().strip() for text in texts)) == 160
    assert tier_counts == {"near": 80, "medium": 64, "far_support": 16}


def test_oos_primary_evaluation_is_calibration_cross_fitted() -> None:
    config = _load(CONFIG_PATH)
    calibration = config["calibration_evaluation"]
    metrics = config["metrics"]
    assert isinstance(calibration, dict)
    assert isinstance(metrics, dict)
    assert calibration["primary_protocol"] == "five_fold_cross_fitted_temperature"
    assert calibration["folds"] == 5
    assert calibration["salt"] == "helix-phase2-calibration-v1-2026-08-18"
    assert "held-out in-domain fold" in str(calibration["per_fold_rule"])
    assert metrics["primary"] == "cross_fitted_oos_auroc"
    assert metrics["declared_oos_recall"] == 0.95


def test_oos_evaluation_keeps_selection_boundary_frozen() -> None:
    config = _load(CONFIG_PATH)
    selection = config["selection"]
    diagnostics = config["diagnostics"]
    assert isinstance(selection, dict)
    assert isinstance(diagnostics, dict)
    assert selection["final_operating_threshold_selected_here"] is False
    assert selection["routing_cost_used_here"] is False
    assert selection["test_set_may_be_opened"] is False
    assert selection["A3_reentry_allowed"] is False
    assert selection["additional_OOS_detector_family_allowed"] is False
    assert diagnostics["raw_variants_may_win_selection"] is False
    assert diagnostics["full_validation_temperature_variant_may_win_primary_selection"] is False


def test_oos_eligible_models_use_frozen_temperatures() -> None:
    config = _load(CONFIG_PATH)
    models = config["eligible_models"]
    assert isinstance(models, list)
    by_id = {str(model["id"]): model for model in models if isinstance(model, dict)}
    assert set(by_id) == {"A1_temperature", "A2_temperature"}
    assert by_id["A1_temperature"]["full_validation_temperature"] == 0.3464179550044782
    assert by_id["A2_temperature"]["full_validation_temperature"] == 0.45797404927014607

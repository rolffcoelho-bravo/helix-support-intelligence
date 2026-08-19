# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5",
#   "scikit-learn==1.8.0",
#   "scipy==1.17.0",
#   "sentence-transformers==5.5.1",
#   "torch==2.13.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Evaluate the frozen Phase 2 out-of-scope routing benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "banking77.json"
A2_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_a2.json"
CALIBRATION_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_calibration.json"
OOS_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_oos.json"
OOS_BENCHMARK_PATH = REPO_ROOT / "data" / "oos" / "routing_oos_v1.json"
CALIBRATION_SCRIPT_PATH = REPO_ROOT / "benchmarks" / "routing" / "evaluate_calibration.py"
A1_CHECKPOINT_PATH = REPO_ROOT / "benchmarks" / "routing" / "results" / "a0_a1_validation_v1.json"
A2_CHECKPOINT_PATH = REPO_ROOT / "benchmarks" / "routing" / "results" / "a2_validation_v2.json"
RUN_ID = "phase2-routing-oos-v1"


def _load_calibration_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "helix_phase2_calibration", CALIBRATION_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen calibration benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _data_module() -> Any:
    source_root = str(REPO_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    import importlib

    return importlib.import_module("helix_support_intelligence.data.banking77")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-support-intelligence-phase2/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def _load_oos_records() -> list[dict[str, str]]:
    payload = json.loads(OOS_BENCHMARK_PATH.read_text(encoding="utf-8"))
    categories = payload["categories"]
    if payload["version"] != "routing-oos-v1":
        raise ValueError("OOS benchmark version drifted")
    if len(categories) != int(payload["category_count"]):
        raise ValueError("OOS category count drifted")

    records: list[dict[str, str]] = []
    for category in categories:
        category_id = str(category["id"])
        tier = str(category["tier"])
        for index, text in enumerate(category["queries"]):
            records.append(
                {
                    "oos_id": f"{category_id}:{index:02d}",
                    "category": category_id,
                    "tier": tier,
                    "text": str(text),
                }
            )
    if len(records) != int(payload["query_count"]):
        raise ValueError("OOS query count drifted")
    if len({record["text"].casefold().strip() for record in records}) != len(records):
        raise ValueError("OOS benchmark contains duplicate query text")
    return records


def _oos_score(probabilities: np.ndarray) -> np.ndarray:
    return 1.0 - np.max(probabilities, axis=1)


def _binary_oos_metrics(
    in_domain_scores: np.ndarray,
    oos_scores: np.ndarray,
    declared_recall: float,
) -> dict[str, float]:
    if len(in_domain_scores) == 0 or len(oos_scores) == 0:
        raise ValueError("OOS metrics require non-empty in-domain and OOS samples")
    targets = np.concatenate(
        [np.zeros(len(in_domain_scores), dtype=np.int64), np.ones(len(oos_scores), dtype=np.int64)]
    )
    scores = np.concatenate([in_domain_scores, oos_scores])
    auroc = float(roc_auc_score(targets, scores))
    threshold = float(np.quantile(oos_scores, 1.0 - declared_recall, method="lower"))
    actual_recall = float(np.mean(oos_scores >= threshold))
    in_domain_fpr = float(np.mean(in_domain_scores >= threshold))
    return {
        "oos_auroc": auroc,
        "metric_threshold": threshold,
        "actual_oos_recall": actual_recall,
        "in_domain_fpr_at_declared_oos_recall": in_domain_fpr,
    }


def _weighted_mean(
    rows: list[dict[str, object]], key: str, weight_key: str = "heldout_rows"
) -> float:
    weights = np.asarray([float(row[weight_key]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights))


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _temperature_bounds(calibration_config: dict[str, object]) -> tuple[float, float]:
    methods = calibration_config["methods"]
    assert isinstance(methods, list)
    for method in methods:
        if isinstance(method, dict) and method.get("id") == "temperature_scaling":
            bounds = method["temperature_bounds"]
            assert isinstance(bounds, list) and len(bounds) == 2
            return float(bounds[0]), float(bounds[1])
    raise ValueError("temperature scaling configuration is missing")


def _model_full_temperature(oos_config: dict[str, object], model_id: str) -> float:
    models = oos_config["eligible_models"]
    assert isinstance(models, list)
    for model in models:
        if isinstance(model, dict) and model.get("id") == f"{model_id}_temperature":
            return float(model["full_validation_temperature"])
    raise ValueError(f"OOS temperature is missing for {model_id}")


def _subgroup_indices(records: list[dict[str, str]], key: str) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record[key], []).append(index)
    return {group: np.asarray(indices, dtype=np.int64) for group, indices in sorted(groups.items())}


def _fold_subgroup_metrics(
    in_domain_scores: np.ndarray,
    oos_scores: np.ndarray,
    subgroups: dict[str, np.ndarray],
    declared_recall: float,
) -> dict[str, dict[str, float]]:
    return {
        group: _binary_oos_metrics(in_domain_scores, oos_scores[indices], declared_recall)
        for group, indices in subgroups.items()
    }


def _aggregate_subgroups(
    fold_rows: list[dict[str, object]],
    field: str,
) -> dict[str, dict[str, float]]:
    first = fold_rows[0][field]
    assert isinstance(first, dict)
    output: dict[str, dict[str, float]] = {}
    for group in sorted(first):
        group_rows: list[dict[str, object]] = []
        for fold_row in fold_rows:
            groups = fold_row[field]
            assert isinstance(groups, dict)
            metrics = groups[group]
            assert isinstance(metrics, dict)
            group_rows.append(
                {
                    "heldout_rows": fold_row["heldout_rows"],
                    **metrics,
                }
            )
        output[group] = {
            "weighted_oos_auroc": _weighted_mean(group_rows, "oos_auroc"),
            "weighted_in_domain_fpr_at_declared_oos_recall": _weighted_mean(
                group_rows,
                "in_domain_fpr_at_declared_oos_recall",
            ),
            "min_fold_oos_auroc": min(float(row["oos_auroc"]) for row in group_rows),
            "max_fold_oos_auroc": max(float(row["oos_auroc"]) for row in group_rows),
        }
    return output


def _evaluate_model(
    model_id: str,
    raw_validation: np.ndarray,
    raw_oos: np.ndarray,
    y_validation: np.ndarray,
    fold_assignment: np.ndarray,
    records: list[dict[str, str]],
    calibration_module: Any,
    temperature_bounds: tuple[float, float],
    full_temperature: float,
    declared_recall: float,
) -> dict[str, object]:
    tier_indices = _subgroup_indices(records, "tier")
    category_indices = _subgroup_indices(records, "category")
    fold_rows: list[dict[str, object]] = []

    for fold in sorted(np.unique(fold_assignment)):
        fit_mask = fold_assignment != fold
        score_mask = fold_assignment == fold
        temperature = calibration_module._temperature_fit(
            raw_validation[fit_mask],
            y_validation[fit_mask],
            temperature_bounds,
        )
        calibrated_id = calibration_module._temperature_apply(
            raw_validation[score_mask],
            temperature,
        )
        calibrated_oos = calibration_module._temperature_apply(raw_oos, temperature)
        id_scores = _oos_score(calibrated_id)
        oos_scores = _oos_score(calibrated_oos)
        overall = _binary_oos_metrics(id_scores, oos_scores, declared_recall)
        fold_rows.append(
            {
                "fold": int(fold),
                "heldout_rows": int(np.sum(score_mask)),
                "temperature": temperature,
                **overall,
                "tiers": _fold_subgroup_metrics(
                    id_scores,
                    oos_scores,
                    tier_indices,
                    declared_recall,
                ),
                "categories": _fold_subgroup_metrics(
                    id_scores,
                    oos_scores,
                    category_indices,
                    declared_recall,
                ),
            }
        )

    primary = {
        "weighted_oos_auroc": _weighted_mean(fold_rows, "oos_auroc"),
        "weighted_in_domain_fpr_at_declared_oos_recall": _weighted_mean(
            fold_rows,
            "in_domain_fpr_at_declared_oos_recall",
        ),
        "weighted_actual_oos_recall": _weighted_mean(fold_rows, "actual_oos_recall"),
        "min_fold_oos_auroc": min(float(row["oos_auroc"]) for row in fold_rows),
        "max_fold_oos_auroc": max(float(row["oos_auroc"]) for row in fold_rows),
        "min_fold_fpr": min(
            float(row["in_domain_fpr_at_declared_oos_recall"]) for row in fold_rows
        ),
        "max_fold_fpr": max(
            float(row["in_domain_fpr_at_declared_oos_recall"]) for row in fold_rows
        ),
    }

    raw_id_scores = _oos_score(raw_validation)
    raw_oos_scores = _oos_score(raw_oos)
    raw_diagnostic = _binary_oos_metrics(raw_id_scores, raw_oos_scores, declared_recall)

    full_id = calibration_module._temperature_apply(raw_validation, full_temperature)
    full_oos = calibration_module._temperature_apply(raw_oos, full_temperature)
    full_id_scores = _oos_score(full_id)
    full_oos_scores = _oos_score(full_oos)
    full_diagnostic = {
        **_binary_oos_metrics(full_id_scores, full_oos_scores, declared_recall),
        "temperature": full_temperature,
        "score_quantiles": {
            "in_domain": _quantiles(full_id_scores),
            "oos": _quantiles(full_oos_scores),
        },
    }

    return {
        "model_id": model_id,
        "primary_cross_fitted": primary,
        "folds": fold_rows,
        "tier_diagnostics": _aggregate_subgroups(fold_rows, "tiers"),
        "category_diagnostics": _aggregate_subgroups(fold_rows, "categories"),
        "raw_diagnostic": raw_diagnostic,
        "full_validation_temperature_diagnostic": full_diagnostic,
    }


def _markdown_report(result: dict[str, object]) -> str:
    models = result["models"]
    assert isinstance(models, dict)
    lines = [
        "# Phase 2 OOS Development Benchmark",
        "",
        (
            "> Frozen OOS benchmark and validation-only in-domain reference. "
            "The confirmatory BANKING77 test split was not downloaded or opened."
        ),
        "",
        "| Model | Cross-fitted OOS AUROC | ID FPR @ >=95% OOS recall | Fold AUROC range |",
        "|---|---:|---:|---:|",
    ]
    for model_id in ("A1", "A2"):
        model = models[model_id]
        assert isinstance(model, dict)
        primary = model["primary_cross_fitted"]
        assert isinstance(primary, dict)
        lines.append(
            "| {model} | {auroc:.4f} | {fpr:.4f} | {low:.4f}-{high:.4f} |".format(
                model=model_id,
                auroc=float(primary["weighted_oos_auroc"]),
                fpr=float(primary["weighted_in_domain_fpr_at_declared_oos_recall"]),
                low=float(primary["min_fold_oos_auroc"]),
                high=float(primary["max_fold_oos_auroc"]),
            )
        )
    lines.extend(
        [
            "",
            "Primary metrics use calibration-cross-fitted held-out in-domain folds.",
            "The metric-specific FPR threshold is not the final routing operating threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_dir: Path) -> dict[str, object]:
    calibration_module = _load_calibration_module()
    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)
    a2_config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    calibration_config = json.loads(CALIBRATION_CONFIG_PATH.read_text(encoding="utf-8"))
    oos_config = json.loads(OOS_CONFIG_PATH.read_text(encoding="utf-8"))
    records = _load_oos_records()

    with tempfile.TemporaryDirectory(prefix="helix-phase2-oos-") as temp:
        train_csv = Path(temp) / "train.csv"
        _download(spec.train_url, train_csv)
        if data_api.sha256_file(train_csv) != spec.train_sha256:
            raise ValueError("BANKING77 train checksum drifted")
        source_train = data_api.load_csv(train_csv, "train")

    train, validation, quarantined = data_api.split_training_examples(source_train, spec)
    if len(train) != spec.expected_counts["train"]:
        raise ValueError("derived train count drifted")
    if len(validation) != spec.expected_counts["validation"]:
        raise ValueError("derived validation count drifted")
    if len(quarantined) != spec.expected_counts["quarantine"]:
        raise ValueError("quarantine count drifted")

    validation_hash = data_api.sha256_bytes(
        data_api.canonical_jsonl_bytes(validation, spec.source_revision)
    )
    if validation_hash != oos_config["in_domain_reference"]["sha256"]:
        raise ValueError("OOS in-domain validation hash drifted")

    oos_texts = [record["text"] for record in records]
    source_texts = {data_api.canonical_text(example.text) for example in source_train}
    oos_normalized = {data_api.canonical_text(text) for text in oos_texts}
    overlap = source_texts & oos_normalized
    if overlap:
        raise ValueError(f"OOS benchmark has exact normalized BANKING77 overlap: {sorted(overlap)}")

    labels = sorted({example.intent for example in train})
    label_to_index = {label: index for index, label in enumerate(labels)}
    x_train = [example.text for example in train]
    x_validation = [example.text for example in validation]
    y_train = np.asarray([label_to_index[example.intent] for example in train], dtype=np.int64)
    y_validation = np.asarray(
        [label_to_index[example.intent] for example in validation],
        dtype=np.int64,
    )
    x_evaluation = x_validation + oos_texts

    torch.manual_seed(20260818)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    a1_all = calibration_module._fit_a1(x_train, y_train, x_evaluation)
    a2_all = calibration_module._fit_a2(x_train, y_train, x_evaluation, a2_config)
    split_at = len(validation)
    raw = {
        "A1": (a1_all[:split_at], a1_all[split_at:]),
        "A2": (a2_all[:split_at], a2_all[split_at:]),
    }

    a1_checkpoint = json.loads(A1_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    a2_checkpoint = json.loads(A2_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    expected_macro_f1 = {
        "A1": float(a1_checkpoint["models"]["A1"]["metrics"]["macro_f1"]),
        "A2": float(a2_checkpoint["metrics"]["macro_f1"]),
    }
    for model_id, (validation_probabilities, _) in raw.items():
        observed = calibration_module._metrics(y_validation, validation_probabilities)
        if abs(float(observed["macro_f1"]) - expected_macro_f1[model_id]) > 1e-12:
            raise ValueError(f"{model_id} raw classifier no longer matches frozen checkpoint")

    calibration_cross_fitting = calibration_config["cross_fitting"]
    fold_assignment = calibration_module._fold_assignment(
        validation,
        labels,
        spec.source_revision,
        str(calibration_cross_fitting["salt"]),
        int(calibration_cross_fitting["folds"]),
        data_api,
    )
    fold_counts = Counter(int(value) for value in fold_assignment)
    expected_fold_counts = {0: 390, 1: 392, 2: 393, 3: 402, 4: 399}
    if dict(sorted(fold_counts.items())) != expected_fold_counts:
        raise ValueError(f"audited calibration fold assignment drifted: {fold_counts}")

    declared_recall = float(oos_config["metrics"]["declared_oos_recall"])
    bounds = _temperature_bounds(calibration_config)
    model_results: dict[str, object] = {}
    for model_id in ("A1", "A2"):
        validation_probabilities, oos_probabilities = raw[model_id]
        model_results[model_id] = _evaluate_model(
            model_id,
            validation_probabilities,
            oos_probabilities,
            y_validation,
            fold_assignment,
            records,
            calibration_module,
            bounds,
            _model_full_temperature(oos_config, model_id),
            declared_recall,
        )

    a1_primary = model_results["A1"]["primary_cross_fitted"]
    a2_primary = model_results["A2"]["primary_cross_fitted"]
    assert isinstance(a1_primary, dict)
    assert isinstance(a2_primary, dict)
    if float(a2_primary["weighted_oos_auroc"]) > float(a1_primary["weighted_oos_auroc"]):
        preferred = "A2"
    elif float(a2_primary["weighted_oos_auroc"]) < float(a1_primary["weighted_oos_auroc"]):
        preferred = "A1"
    else:
        preferred = (
            "A2"
            if float(a2_primary["weighted_in_domain_fpr_at_declared_oos_recall"])
            <= float(a1_primary["weighted_in_domain_fpr_at_declared_oos_recall"])
            else "A1"
        )

    result: dict[str, object] = {
        "run_id": RUN_ID,
        "status": "development_validation_plus_frozen_oos",
        "test_set_opened": False,
        "benchmark": {
            "version": "routing-oos-v1",
            "queries": len(records),
            "categories": len({record["category"] for record in records}),
            "tier_counts": dict(sorted(Counter(record["tier"] for record in records).items())),
            "exact_normalized_overlap_with_banking77_source_train": 0,
        },
        "in_domain": {
            "partition": "validation",
            "rows": len(validation),
            "sha256": validation_hash,
        },
        "cross_fitting": {
            "fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
            "salt": calibration_cross_fitting["salt"],
        },
        "metrics_contract": {
            "declared_oos_recall": declared_recall,
            "primary": oos_config["metrics"]["primary"],
            "final_operating_threshold_selected": False,
        },
        "models": model_results,
        "oos_preferred_model_by_frozen_primary_rule": preferred,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    print(_markdown_report(result))
    print("Confirmatory test opened: false")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "phase2-routing" / "oos",
    )
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()

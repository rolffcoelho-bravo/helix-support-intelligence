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
"""Evaluate frozen Phase 2 calibration candidates with validation cross-fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
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
from scipy.optimize import minimize_scalar
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

ECE_BINS = 15
RISK_COVERAGE_POINTS = tuple(step / 10 for step in range(1, 11))
RUN_ID = "phase2-routing-calibration-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "banking77.json"
A2_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_a2.json"
CALIBRATION_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_calibration.json"
A1_CHECKPOINT_PATH = REPO_ROOT / "benchmarks" / "routing" / "results" / "a0_a1_validation_v1.json"
A2_CHECKPOINT_PATH = REPO_ROOT / "benchmarks" / "routing" / "results" / "a2_validation_v2.json"


def _data_module() -> Any:
    source_root = str(REPO_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return importlib.import_module("helix_support_intelligence.data.banking77")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-support-intelligence-phase2/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def _top_k_recall(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    top_k = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k == y_true[:, None], axis=1)))


def _multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _negative_log_likelihood(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return float(-np.mean(np.log(clipped)))


def _expected_calibration_error(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(np.sum(mask))
        if count == 0:
            continue
        accuracy = float(np.mean(correct[mask]))
        mean_confidence = float(np.mean(confidence[mask]))
        ece += (count / total) * abs(accuracy - mean_confidence)
    return float(ece)


def _risk_coverage(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    confidence = np.max(probabilities, axis=1)
    order = np.argsort(-confidence, kind="stable")
    rows: list[dict[str, float | int]] = []
    for coverage in RISK_COVERAGE_POINTS:
        accepted = max(1, min(len(order), round(coverage * len(order))))
        selected = order[:accepted]
        accuracy = float(np.mean(predicted[selected] == y_true[selected]))
        rows.append(
            {
                "coverage": coverage,
                "accepted": accepted,
                "confidence_threshold": float(confidence[selected[-1]]),
                "selective_risk": 1.0 - accuracy,
            }
        )
    return rows


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predicted = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "top3_recall": _top_k_recall(y_true, probabilities, 3),
        "expected_calibration_error_15bin": _expected_calibration_error(
            y_true,
            predicted,
            probabilities,
        ),
        "multiclass_brier_score": _multiclass_brier(y_true, probabilities),
        "negative_log_likelihood": _negative_log_likelihood(y_true, probabilities),
        "mean_max_probability": float(np.mean(np.max(probabilities, axis=1))),
        "risk_coverage": _risk_coverage(y_true, predicted, probabilities),
    }


def _fold_assignment(
    validation: list[Any],
    labels: list[str],
    source_revision: str,
    salt: str,
    folds: int,
    data_api: Any,
) -> np.ndarray:
    label_to_rows: dict[str, list[tuple[str, int]]] = {label: [] for label in labels}
    for index, example in enumerate(validation):
        sample_id = data_api.sample_id(example, source_revision)
        digest = hashlib.sha256(f"{salt}\t{sample_id}".encode()).hexdigest()
        label_to_rows[example.intent].append((digest, index))

    assignment = np.full(len(validation), -1, dtype=np.int64)
    for label in labels:
        ordered = sorted(label_to_rows[label])
        for position, (_, row_index) in enumerate(ordered):
            assignment[row_index] = position % folds
    if np.any(assignment < 0):
        raise RuntimeError("calibration fold assignment is incomplete")
    return assignment


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _temperature_fit(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    logits = np.log(np.clip(probabilities, 1e-15, 1.0))

    def objective(temperature: float) -> float:
        calibrated = _softmax(logits / temperature)
        return _negative_log_likelihood(y_true, calibrated)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(result.x)


def _temperature_apply(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-15, 1.0))
    return _softmax(logits / temperature)


def _normalize_rows(transformed: np.ndarray, raw: np.ndarray) -> np.ndarray:
    totals = np.sum(transformed, axis=1, keepdims=True)
    output = np.divide(
        transformed,
        totals,
        out=np.zeros_like(transformed),
        where=totals > 0.0,
    )
    zero_rows = np.squeeze(totals <= 0.0, axis=1)
    output[zero_rows] = raw[zero_rows]
    return output


def _isotonic_fit(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> list[IsotonicRegression]:
    calibrators: list[IsotonicRegression] = []
    for class_index in range(probabilities.shape[1]):
        target = (y_true == class_index).astype(np.int64)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probabilities[:, class_index], target)
        calibrators.append(calibrator)
    return calibrators


def _isotonic_apply(
    probabilities: np.ndarray,
    calibrators: list[IsotonicRegression],
) -> np.ndarray:
    transformed = np.column_stack(
        [
            calibrator.predict(probabilities[:, class_index])
            for class_index, calibrator in enumerate(calibrators)
        ]
    )
    return _normalize_rows(transformed, probabilities)


def _log_odds(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _platt_fit(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
) -> list[LogisticRegression]:
    scores = _log_odds(probabilities)
    calibrators: list[LogisticRegression] = []
    for class_index in range(probabilities.shape[1]):
        target = (y_true == class_index).astype(np.int64)
        calibrator = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=max_iter,
        )
        calibrator.fit(scores[:, class_index].reshape(-1, 1), target)
        calibrators.append(calibrator)
    return calibrators


def _platt_apply(
    probabilities: np.ndarray,
    calibrators: list[LogisticRegression],
) -> np.ndarray:
    scores = _log_odds(probabilities)
    transformed = np.column_stack(
        [
            calibrator.predict_proba(scores[:, class_index].reshape(-1, 1))[:, 1]
            for class_index, calibrator in enumerate(calibrators)
        ]
    )
    return _normalize_rows(transformed, probabilities)


def _cross_fit(
    method_id: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    fold_assignment: np.ndarray,
    method_config: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    calibrated = np.zeros_like(probabilities)
    fold_records: list[dict[str, object]] = []
    for fold in sorted(np.unique(fold_assignment)):
        fit_mask = fold_assignment != fold
        score_mask = fold_assignment == fold
        fit_probabilities = probabilities[fit_mask]
        fit_y = y_true[fit_mask]
        score_probabilities = probabilities[score_mask]

        if method_id == "temperature_scaling":
            bounds_raw = method_config["temperature_bounds"]
            if not isinstance(bounds_raw, list) or len(bounds_raw) != 2:
                raise ValueError("invalid temperature bounds")
            bounds = (float(bounds_raw[0]), float(bounds_raw[1]))
            fitted = _temperature_fit(fit_probabilities, fit_y, bounds)
            fold_output = _temperature_apply(score_probabilities, fitted)
            parameters: dict[str, object] = {"temperature": fitted}
        elif method_id == "isotonic_regression":
            fitted_isotonic = _isotonic_fit(fit_probabilities, fit_y)
            fold_output = _isotonic_apply(score_probabilities, fitted_isotonic)
            parameters = {
                "class_calibrators": len(fitted_isotonic),
                "total_threshold_points": sum(
                    len(calibrator.X_thresholds_) for calibrator in fitted_isotonic
                ),
            }
        elif method_id == "platt_scaling":
            fitted_platt = _platt_fit(
                fit_probabilities,
                fit_y,
                c_value=float(method_config["C"]),
                max_iter=int(method_config["max_iter"]),
            )
            fold_output = _platt_apply(score_probabilities, fitted_platt)
            parameters = {
                "class_calibrators": len(fitted_platt),
                "mean_coefficient": float(
                    np.mean([calibrator.coef_[0, 0] for calibrator in fitted_platt])
                ),
                "mean_intercept": float(
                    np.mean([calibrator.intercept_[0] for calibrator in fitted_platt])
                ),
            }
        else:
            raise ValueError(f"unknown calibration method: {method_id}")

        calibrated[score_mask] = fold_output
        fold_records.append(
            {
                "fold": int(fold),
                "fit_rows": int(np.sum(fit_mask)),
                "score_rows": int(np.sum(score_mask)),
                "parameters": parameters,
            }
        )

    row_sums = np.sum(calibrated, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise RuntimeError("cross-fitted calibrated probabilities do not sum to one")
    return calibrated, fold_records


def _fit_a1(
    x_train: list[str],
    y_train: np.ndarray,
    x_validation: list[str],
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(x_train)
    validation_matrix = vectorizer.transform(x_validation)
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-4,
    )
    classifier.fit(train_matrix, y_train)
    return classifier.predict_proba(validation_matrix)


def _fit_a2(
    x_train: list[str],
    y_train: np.ndarray,
    x_validation: list[str],
    a2_config: dict[str, object],
) -> np.ndarray:
    representation = a2_config["representation"]
    classifier_config = a2_config["classifier"]
    if not isinstance(representation, dict) or not isinstance(classifier_config, dict):
        raise TypeError("invalid A2 configuration")

    model = SentenceTransformer(
        str(representation["model_id"]),
        revision=str(representation["revision"]),
        device="cpu",
        trust_remote_code=False,
    )
    batch_size = int(representation["batch_size"])
    train_embeddings = model.encode(
        x_train,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(representation["normalize_embeddings"]),
    )
    validation_embeddings = model.encode(
        x_validation,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(representation["normalize_embeddings"]),
    )

    classifier = LogisticRegression(
        C=float(classifier_config["C"]),
        solver=str(classifier_config["solver"]),
        max_iter=int(classifier_config["max_iter"]),
        tol=float(classifier_config["tol"]),
    )
    classifier.fit(train_embeddings, y_train)
    return classifier.predict_proba(validation_embeddings)


def _method_configs(config: dict[str, object]) -> dict[str, dict[str, object]]:
    methods = config["methods"]
    if not isinstance(methods, list):
        raise TypeError("calibration methods must be a list")
    output: dict[str, dict[str, object]] = {}
    for method in methods:
        if not isinstance(method, dict):
            raise TypeError("calibration method must be an object")
        output[str(method["id"])] = method
    return output


def _passes_guardrails(
    raw_metrics: dict[str, object],
    calibrated_metrics: dict[str, object],
    guardrails: dict[str, object],
) -> bool:
    macro_drop = float(raw_metrics["macro_f1"]) - float(calibrated_metrics["macro_f1"])
    top3_drop = float(raw_metrics["top3_recall"]) - float(calibrated_metrics["top3_recall"])
    return macro_drop <= float(guardrails["max_macro_f1_drop_vs_raw"]) and top3_drop <= float(
        guardrails["max_top3_recall_drop_vs_raw"]
    )


def _select_method(
    raw_metrics: dict[str, object],
    candidates: dict[str, dict[str, object]],
    guardrails: dict[str, object],
) -> tuple[str, str]:
    eligible = {
        method_id: result
        for method_id, result in candidates.items()
        if _passes_guardrails(raw_metrics, result["metrics"], guardrails)
    }
    if not eligible:
        return "raw", "no calibration candidate passed classification guardrails"

    ranked = sorted(
        eligible.items(),
        key=lambda item: (
            float(item[1]["metrics"]["multiclass_brier_score"]),
            float(item[1]["metrics"]["negative_log_likelihood"]),
            float(item[1]["metrics"]["expected_calibration_error_15bin"]),
            item[0],
        ),
    )
    winner_id, winner = ranked[0]
    if float(winner["metrics"]["multiclass_brier_score"]) >= float(
        raw_metrics["multiclass_brier_score"]
    ):
        return "raw", "best eligible calibrator did not improve primary Brier score"
    return winner_id, "lowest cross-fitted Brier score among guardrail-valid candidates"


def _full_refit_summary(
    method_id: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    method_config: dict[str, object] | None,
) -> dict[str, object]:
    if method_id == "raw":
        return {"method_id": "raw", "parameters": None}
    if method_config is None:
        raise ValueError("selected calibration method config is missing")
    if method_id == "temperature_scaling":
        bounds_raw = method_config["temperature_bounds"]
        assert isinstance(bounds_raw, list)
        temperature = _temperature_fit(
            probabilities,
            y_true,
            (float(bounds_raw[0]), float(bounds_raw[1])),
        )
        return {"method_id": method_id, "parameters": {"temperature": temperature}}
    if method_id == "isotonic_regression":
        calibrators = _isotonic_fit(probabilities, y_true)
        digest = hashlib.sha256()
        total_points = 0
        for calibrator in calibrators:
            total_points += len(calibrator.X_thresholds_)
            digest.update(np.asarray(calibrator.X_thresholds_, dtype=np.float64).tobytes())
            digest.update(np.asarray(calibrator.y_thresholds_, dtype=np.float64).tobytes())
        return {
            "method_id": method_id,
            "parameters": {
                "class_calibrators": len(calibrators),
                "total_threshold_points": total_points,
                "threshold_arrays_sha256": digest.hexdigest(),
            },
        }
    if method_id == "platt_scaling":
        calibrators = _platt_fit(
            probabilities,
            y_true,
            c_value=float(method_config["C"]),
            max_iter=int(method_config["max_iter"]),
        )
        coefficients = [float(calibrator.coef_[0, 0]) for calibrator in calibrators]
        intercepts = [float(calibrator.intercept_[0]) for calibrator in calibrators]
        digest = hashlib.sha256(
            np.asarray(coefficients + intercepts, dtype=np.float64).tobytes()
        ).hexdigest()
        return {
            "method_id": method_id,
            "parameters": {
                "class_calibrators": len(calibrators),
                "mean_coefficient": float(np.mean(coefficients)),
                "mean_intercept": float(np.mean(intercepts)),
                "coefficient_intercept_sha256": digest,
            },
        }
    raise ValueError(f"unknown selected calibration method: {method_id}")


def _write_predictions(
    path: Path,
    validation: list[Any],
    labels: list[str],
    source_revision: str,
    data_api: Any,
    raw_probabilities: np.ndarray,
    selected_probabilities: np.ndarray,
    fold_assignment: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "true_intent",
                "fold",
                "raw_predicted_intent",
                "raw_confidence",
                "calibrated_predicted_intent",
                "calibrated_confidence",
            ],
        )
        writer.writeheader()
        for example, raw_row, calibrated_row, fold in zip(
            validation,
            raw_probabilities,
            selected_probabilities,
            fold_assignment,
            strict=True,
        ):
            raw_index = int(np.argmax(raw_row))
            calibrated_index = int(np.argmax(calibrated_row))
            writer.writerow(
                {
                    "sample_id": data_api.sample_id(example, source_revision),
                    "true_intent": example.intent,
                    "fold": int(fold),
                    "raw_predicted_intent": labels[raw_index],
                    "raw_confidence": f"{float(raw_row[raw_index]):.12f}",
                    "calibrated_predicted_intent": labels[calibrated_index],
                    "calibrated_confidence": f"{float(calibrated_row[calibrated_index]):.12f}",
                }
            )


def _markdown_report(result: dict[str, object]) -> str:
    lines = [
        "# Phase 2 Cross-Fitted Calibration Benchmark",
        "",
        (
            "> Validation-only development evidence. The confirmatory test split was not "
            "downloaded or opened."
        ),
        "",
    ]
    models = result["models"]
    assert isinstance(models, dict)
    for model_id in ("A1", "A2"):
        model = models[model_id]
        assert isinstance(model, dict)
        raw = model["raw_metrics"]
        candidates = model["candidates"]
        assert isinstance(raw, dict)
        assert isinstance(candidates, dict)
        lines.extend(
            [
                f"## {model_id}",
                "",
                "| Method | Macro-F1 | Top-3 | ECE | Brier | NLL |",
                "|---|---:|---:|---:|---:|---:|",
                ("| raw | {f1:.4f} | {top3:.4f} | {ece:.4f} | {brier:.4f} | {nll:.4f} |").format(
                    f1=float(raw["macro_f1"]),
                    top3=float(raw["top3_recall"]),
                    ece=float(raw["expected_calibration_error_15bin"]),
                    brier=float(raw["multiclass_brier_score"]),
                    nll=float(raw["negative_log_likelihood"]),
                ),
            ]
        )
        for method_id in ("temperature_scaling", "isotonic_regression", "platt_scaling"):
            candidate = candidates[method_id]
            assert isinstance(candidate, dict)
            metrics = candidate["metrics"]
            assert isinstance(metrics, dict)
            lines.append(
                (
                    "| {method} | {f1:.4f} | {top3:.4f} | {ece:.4f} | {brier:.4f} | {nll:.4f} |"
                ).format(
                    method=method_id,
                    f1=float(metrics["macro_f1"]),
                    top3=float(metrics["top3_recall"]),
                    ece=float(metrics["expected_calibration_error_15bin"]),
                    brier=float(metrics["multiclass_brier_score"]),
                    nll=float(metrics["negative_log_likelihood"]),
                )
            )
        lines.extend(
            [
                "",
                f"Selected method: **{model['selected_method']}**.",
                f"Selection reason: {model['selection_reason']}.",
                "",
            ]
        )
    return "\n".join(lines)


def run(output_dir: Path) -> dict[str, object]:
    calibration_config = json.loads(CALIBRATION_CONFIG_PATH.read_text(encoding="utf-8"))
    a2_config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    cross_fitting = calibration_config["cross_fitting"]
    evaluation = calibration_config["evaluation"]
    assert isinstance(cross_fitting, dict)
    assert isinstance(evaluation, dict)
    guardrails = evaluation["classification_guardrails"]
    assert isinstance(guardrails, dict)
    methods = _method_configs(calibration_config)

    torch.manual_seed(20260818)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)
    with tempfile.TemporaryDirectory(prefix="helix-phase2-calibration-") as temp:
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

    train_hash = data_api.sha256_bytes(data_api.canonical_jsonl_bytes(train, spec.source_revision))
    validation_hash = data_api.sha256_bytes(
        data_api.canonical_jsonl_bytes(validation, spec.source_revision)
    )
    if train_hash != spec.expected_hashes["train"]:
        raise ValueError("derived train hash drifted")
    if validation_hash != spec.expected_hashes["validation"]:
        raise ValueError("derived validation hash drifted")

    labels = sorted({example.intent for example in train})
    label_to_index = {label: index for index, label in enumerate(labels)}
    x_train = [example.text for example in train]
    x_validation = [example.text for example in validation]
    y_train = np.asarray([label_to_index[example.intent] for example in train], dtype=np.int64)
    y_validation = np.asarray(
        [label_to_index[example.intent] for example in validation],
        dtype=np.int64,
    )

    fold_assignment = _fold_assignment(
        validation,
        labels,
        spec.source_revision,
        str(cross_fitting["salt"]),
        int(cross_fitting["folds"]),
        data_api,
    )
    fold_counts = Counter(int(item) for item in fold_assignment)
    if len(fold_counts) != int(cross_fitting["folds"]):
        raise RuntimeError("calibration fold cardinality drifted")

    raw_probabilities = {
        "A1": _fit_a1(x_train, y_train, x_validation),
        "A2": _fit_a2(x_train, y_train, x_validation, a2_config),
    }

    a1_checkpoint = json.loads(A1_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    a2_checkpoint = json.loads(A2_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    expected = {
        "A1": a1_checkpoint["models"]["A1"]["metrics"],
        "A2": a2_checkpoint["metrics"],
    }
    for model_id, probabilities in raw_probabilities.items():
        observed = _metrics(y_validation, probabilities)
        expected_metrics = expected[model_id]
        if abs(float(observed["macro_f1"]) - float(expected_metrics["macro_f1"])) > 1e-12:
            raise ValueError(f"{model_id} raw macro-F1 no longer matches frozen checkpoint")
        if abs(float(observed["top3_recall"]) - float(expected_metrics["top3_recall"])) > 1e-12:
            raise ValueError(f"{model_id} raw top-3 recall no longer matches frozen checkpoint")

    model_results: dict[str, object] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id in ("A1", "A2"):
        raw = raw_probabilities[model_id]
        raw_metrics = _metrics(y_validation, raw)
        candidate_results: dict[str, dict[str, object]] = {}
        candidate_probabilities: dict[str, np.ndarray] = {}
        for method_id, method_config in methods.items():
            calibrated, fold_records = _cross_fit(
                method_id,
                raw,
                y_validation,
                fold_assignment,
                method_config,
            )
            metrics = _metrics(y_validation, calibrated)
            candidate_probabilities[method_id] = calibrated
            candidate_results[method_id] = {
                "metrics": metrics,
                "passes_classification_guardrails": _passes_guardrails(
                    raw_metrics,
                    metrics,
                    guardrails,
                ),
                "fold_records": fold_records,
            }

        selected_method, selection_reason = _select_method(
            raw_metrics,
            candidate_results,
            guardrails,
        )
        selected_probabilities = (
            raw if selected_method == "raw" else candidate_probabilities[selected_method]
        )
        full_refit = _full_refit_summary(
            selected_method,
            raw,
            y_validation,
            methods.get(selected_method),
        )
        _write_predictions(
            output_dir / f"{model_id.lower()}_cross_fitted_calibration_predictions.csv",
            validation,
            labels,
            spec.source_revision,
            data_api,
            raw,
            selected_probabilities,
            fold_assignment,
        )
        model_results[model_id] = {
            "raw_metrics": raw_metrics,
            "candidates": candidate_results,
            "selected_method": selected_method,
            "selection_reason": selection_reason,
            "selected_cross_fitted_metrics": _metrics(y_validation, selected_probabilities),
            "full_validation_refit": full_refit,
        }

    result: dict[str, object] = {
        "run_id": RUN_ID,
        "status": "development_validation_only",
        "test_set_opened": False,
        "data": {
            "dataset_contract": spec.version,
            "source_revision": spec.source_revision,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        },
        "cross_fitting": {
            "folds": int(cross_fitting["folds"]),
            "assignment": cross_fitting["assignment"],
            "salt": cross_fitting["salt"],
            "fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        },
        "selection": {
            "primary_metric": evaluation["primary_selection_metric"],
            "secondary_metrics": evaluation["secondary_selection_metrics"],
            "classification_guardrails": guardrails,
            "operating_threshold_selected": False,
            "routing_cost_used": False,
        },
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
        "models": model_results,
        "excluded_models": calibration_config["excluded_models"],
    }

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
        default=REPO_ROOT / "artifacts" / "phase2-routing" / "calibration",
    )
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5",
#   "scikit-learn==1.8.0",
#   "scipy==1.17.0",
#   "sentence-transformers==5.5.1",
# ]
# ///
"""Run the frozen Phase 2 A2 routing benchmark on validation only."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sentence_transformers
import sklearn
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

ECE_BINS = 15
RISK_COVERAGE_POINTS = tuple(step / 10 for step in range(1, 11))
RUN_ID = "phase2-development-a2-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "banking77.json"
A2_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_a2.json"
A1_CHECKPOINT_PATH = (
    REPO_ROOT / "benchmarks" / "routing" / "results" / "a0_a1_validation_v1.json"
)


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


def _confusion_pairs(
    y_true: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
    limit: int = 20,
) -> list[dict[str, str | int]]:
    errors = Counter(
        (labels[int(true_index)], labels[int(pred_index)])
        for true_index, pred_index in zip(y_true, predicted, strict=True)
        if true_index != pred_index
    )
    return [
        {"true_intent": pair[0], "predicted_intent": pair[1], "count": count}
        for pair, count in errors.most_common(limit)
    ]


def _metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
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
        "mean_max_probability": float(np.mean(np.max(probabilities, axis=1))),
        "risk_coverage": _risk_coverage(y_true, predicted, probabilities),
        "top_confusion_pairs": _confusion_pairs(y_true, predicted, labels),
    }


def _risk_coverage_delta(
    a1_metrics: dict[str, object],
    a2_metrics: dict[str, object],
) -> list[dict[str, float]]:
    a1_rows = a1_metrics["risk_coverage"]
    a2_rows = a2_metrics["risk_coverage"]
    assert isinstance(a1_rows, list)
    assert isinstance(a2_rows, list)
    rows: list[dict[str, float]] = []
    for a1_row, a2_row in zip(a1_rows, a2_rows, strict=True):
        assert isinstance(a1_row, dict)
        assert isinstance(a2_row, dict)
        coverage = float(a1_row["coverage"])
        if coverage != float(a2_row["coverage"]):
            raise ValueError("risk-coverage grid drifted")
        a1_risk = float(a1_row["selective_risk"])
        a2_risk = float(a2_row["selective_risk"])
        rows.append(
            {
                "coverage": coverage,
                "a1_selective_risk": a1_risk,
                "a2_selective_risk": a2_risk,
                "a2_minus_a1_selective_risk": a2_risk - a1_risk,
            }
        )
    return rows


def _confusion_change(
    a1_metrics: dict[str, object],
    a2_metrics: dict[str, object],
) -> list[dict[str, str | int]]:
    a1_rows = a1_metrics["top_confusion_pairs"]
    a2_rows = a2_metrics["top_confusion_pairs"]
    assert isinstance(a1_rows, list)
    assert isinstance(a2_rows, list)

    def as_map(rows: list[object]) -> dict[tuple[str, str], int]:
        mapped: dict[tuple[str, str], int] = {}
        for row in rows:
            assert isinstance(row, dict)
            key = (str(row["true_intent"]), str(row["predicted_intent"]))
            mapped[key] = int(row["count"])
        return mapped

    a1_map = as_map(a1_rows)
    a2_map = as_map(a2_rows)
    keys = sorted(set(a1_map) | set(a2_map))
    return [
        {
            "true_intent": key[0],
            "predicted_intent": key[1],
            "a1_count": a1_map.get(key, 0),
            "a2_count": a2_map.get(key, 0),
            "a2_minus_a1": a2_map.get(key, 0) - a1_map.get(key, 0),
        }
        for key in keys
    ]


def _write_predictions(
    path: Path,
    validation: list[Any],
    predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    source_revision: str,
    data_api: Any,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "true_intent",
                "predicted_intent",
                "confidence",
                "top3_intents",
            ],
        )
        writer.writeheader()
        for example, pred_index, row in zip(validation, predicted, probabilities, strict=True):
            order = np.argsort(-row, kind="stable")[:3]
            writer.writerow(
                {
                    "sample_id": data_api.sample_id(example, source_revision),
                    "true_intent": example.intent,
                    "predicted_intent": labels[int(pred_index)],
                    "confidence": f"{float(row[int(pred_index)]):.12f}",
                    "top3_intents": "|".join(labels[int(index)] for index in order),
                }
            )


def _markdown_report(result: dict[str, object]) -> str:
    metrics = result["metrics"]
    comparison = result["comparison_to_a1"]
    assert isinstance(metrics, dict)
    assert isinstance(comparison, dict)
    return "\n".join(
        [
            "# Phase 2 A2 Development Benchmark",
            "",
            (
                "> Validation evidence only. The confirmatory BANKING77 test split was not "
                "downloaded or opened."
            ),
            "",
            "| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |",
            "|---|---:|---:|---:|---:|---:|",
            (
                "| A2 | {f1:.4f} | {bal:.4f} | {top3:.4f} | {ece:.4f} | "
                "{brier:.4f} |"
            ).format(
                f1=float(metrics["macro_f1"]),
                bal=float(metrics["balanced_accuracy"]),
                top3=float(metrics["top3_recall"]),
                ece=float(metrics["expected_calibration_error_15bin"]),
                brier=float(metrics["multiclass_brier_score"]),
            ),
            "",
            "## Delta versus frozen A1",
            "",
            f"- Macro-F1: {float(comparison['macro_f1_delta']):+.4f}",
            f"- Balanced accuracy: {float(comparison['balanced_accuracy_delta']):+.4f}",
            f"- Top-3 recall: {float(comparison['top3_recall_delta']):+.4f}",
            f"- ECE: {float(comparison['ece_delta']):+.4f}",
            f"- Brier: {float(comparison['brier_delta']):+.4f}",
            "",
            "Wall-clock timing is stored separately because it is not deterministic evidence.",
            "",
        ]
    )


def run(output_dir: Path) -> dict[str, object]:
    a2_config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    seed = int(a2_config["evaluation"]["seed"])
    representation = a2_config["representation"]
    classifier_config = a2_config["classifier"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)

    with tempfile.TemporaryDirectory(prefix="helix-phase2-a2-") as temp:
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
    y_train = np.asarray([label_to_index[item.intent] for item in train], dtype=np.int64)
    y_validation = np.asarray(
        [label_to_index[item.intent] for item in validation],
        dtype=np.int64,
    )

    load_started = time.perf_counter()
    encoder = SentenceTransformer(
        str(representation["model_id"]),
        revision=str(representation["revision"]),
        device=str(representation["device"]),
        trust_remote_code=False,
    )
    model_load_seconds = time.perf_counter() - load_started

    train_started = time.perf_counter()
    train_embeddings = encoder.encode(
        x_train,
        batch_size=int(representation["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(representation["normalize_embeddings"]),
    )
    train_encoding_seconds = time.perf_counter() - train_started

    validation_started = time.perf_counter()
    validation_embeddings = encoder.encode(
        x_validation,
        batch_size=int(representation["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(representation["normalize_embeddings"]),
    )
    validation_encoding_seconds = time.perf_counter() - validation_started

    expected_dimension = int(representation["embedding_dimension"])
    if train_embeddings.shape[1] != expected_dimension:
        raise ValueError("A2 embedding dimension drifted")

    fit_started = time.perf_counter()
    classifier = LogisticRegression(
        C=float(classifier_config["C"]),
        solver=str(classifier_config["solver"]),
        max_iter=int(classifier_config["max_iter"]),
        tol=float(classifier_config["tol"]),
        class_weight=classifier_config["class_weight"],
    )
    classifier.fit(train_embeddings, y_train)
    classifier_fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    probabilities = classifier.predict_proba(validation_embeddings)
    predicted = classifier.predict(validation_embeddings)
    prediction_seconds = time.perf_counter() - predict_started

    metrics = _metrics(y_validation, predicted, probabilities, labels)
    a1_checkpoint = json.loads(A1_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    a1_metrics = a1_checkpoint["models"]["A1"]["metrics"]

    comparison = {
        "macro_f1_delta": float(metrics["macro_f1"]) - float(a1_metrics["macro_f1"]),
        "balanced_accuracy_delta": (
            float(metrics["balanced_accuracy"]) - float(a1_metrics["balanced_accuracy"])
        ),
        "top3_recall_delta": float(metrics["top3_recall"]) - float(a1_metrics["top3_recall"]),
        "ece_delta": (
            float(metrics["expected_calibration_error_15bin"])
            - float(a1_metrics["expected_calibration_error_15bin"])
        ),
        "brier_delta": (
            float(metrics["multiclass_brier_score"])
            - float(a1_metrics["multiclass_brier_score"])
        ),
        "risk_coverage_delta": _risk_coverage_delta(a1_metrics, metrics),
        "top_confusion_pair_change": _confusion_change(a1_metrics, metrics),
    }

    result: dict[str, object] = {
        "run_id": RUN_ID,
        "phase": 2,
        "status": "development_validation_only",
        "test_set_opened": False,
        "selection_partition": "validation",
        "seed": seed,
        "data": {
            "dataset_contract": spec.version,
            "source_revision": spec.source_revision,
            "source_train_sha256": spec.train_sha256,
            "derived_train_rows": len(train),
            "derived_validation_rows": len(validation),
            "quarantine_rows": len(quarantined),
            "derived_train_sha256": train_hash,
            "derived_validation_sha256": validation_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "specification": {
            "config_version": a2_config["version"],
            "representation": representation,
            "classifier": classifier_config,
            "classifier_iterations": [int(item) for item in classifier.n_iter_],
        },
        "metrics": metrics,
        "comparison_to_a1": comparison,
        "pending_phase2_evidence": [
            "A3",
            "calibration_comparison",
            "frozen_oos_benchmark",
            "expected_routing_cost",
            "final_risk_coverage_operating_point",
        ],
    }

    timing = {
        "run_id": RUN_ID,
        "non_deterministic_wall_clock": True,
        "model_load_seconds": model_load_seconds,
        "train_encoding_seconds": train_encoding_seconds,
        "validation_encoding_seconds": validation_encoding_seconds,
        "validation_encoding_ms_per_example": (
            1000.0 * validation_encoding_seconds / len(validation)
        ),
        "classifier_fit_seconds": classifier_fit_seconds,
        "classifier_validation_prediction_seconds": prediction_seconds,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    _write_predictions(
        output_dir / "a2_validation_predictions.csv",
        validation,
        predicted,
        probabilities,
        labels,
        spec.source_revision,
        data_api,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "phase2-routing" / "a2",
    )
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    metrics = result["metrics"]
    comparison = result["comparison_to_a1"]
    assert isinstance(metrics, dict)
    assert isinstance(comparison, dict)
    print(
        "A2: macro-F1={f1:.4f}, balanced-accuracy={bal:.4f}, "
        "top-3={top3:.4f}, ECE={ece:.4f}".format(
            f1=float(metrics["macro_f1"]),
            bal=float(metrics["balanced_accuracy"]),
            top3=float(metrics["top3_recall"]),
            ece=float(metrics["expected_calibration_error_15bin"]),
        )
    )
    print(f"A2-A1 macro-F1 delta: {float(comparison['macro_f1_delta']):+.4f}")
    print("Confirmatory test opened: false")


if __name__ == "__main__":
    main()

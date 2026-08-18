# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "joblib==1.5.3",
#   "numpy==2.3.5",
#   "scikit-learn==1.8.0",
#   "scipy==1.17.0",
#   "threadpoolctl==3.6.0",
# ]
# ///
"""Run the frozen Phase 2 A0/A1 routing development benchmark on validation only."""

from __future__ import annotations

import argparse
import csv
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
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

SEED = 20260818
ECE_BINS = 15
RISK_COVERAGE_POINTS = tuple(step / 10 for step in range(1, 11))
RUN_ID = "phase2-development-a0-a1-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "data" / "banking77.json"


def _data_module() -> Any:
    """Import repository data primitives without installing the project into the script env."""

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
        bin_accuracy = float(np.mean(correct[mask]))
        bin_confidence = float(np.mean(confidence[mask]))
        ece += (count / total) * abs(bin_accuracy - bin_confidence)
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
        accepted = max(1, min(len(order), int(round(coverage * len(order)))))
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
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "top3_recall": _top_k_recall(y_true, probabilities, 3),
        "expected_calibration_error_15bin": _expected_calibration_error(
            y_true,
            predicted,
            probabilities,
        ),
        "multiclass_brier_score": _multiclass_brier(y_true, probabilities),
        "risk_coverage": _risk_coverage(y_true, predicted, probabilities),
        "top_confusion_pairs": _confusion_pairs(y_true, predicted, labels),
    }


def _evaluate_dummy(
    strategy: str,
    x_train: list[str],
    y_train: np.ndarray,
    x_validation: list[str],
    y_validation: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    classifier = DummyClassifier(strategy=strategy, random_state=SEED)
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_validation)
    predicted = classifier.predict(x_validation)
    return {
        "model_id": f"A0-{strategy}",
        "family": "dummy_reference",
        "specification": {"strategy": strategy, "random_state": SEED},
        "metrics": _metrics(y_validation, predicted, probabilities, labels),
    }


def _evaluate_a1(
    x_train: list[str],
    y_train: np.ndarray,
    x_validation: list[str],
    y_validation: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
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
    probabilities = classifier.predict_proba(validation_matrix)
    predicted = classifier.predict(validation_matrix)

    result = {
        "model_id": "A1",
        "family": "tfidf_logistic_regression",
        "specification": {
            "tfidf": {
                "lowercase": True,
                "ngram_range": [1, 2],
                "min_df": 2,
                "sublinear_tf": True,
                "norm": "l2",
                "feature_count": len(vectorizer.vocabulary_),
            },
            "logistic_regression": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "tol": 1e-4,
                "iterations": [int(item) for item in classifier.n_iter_],
            },
        },
        "metrics": _metrics(y_validation, predicted, probabilities, labels),
    }
    return result, predicted, probabilities


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
    models = result["models"]
    assert isinstance(models, list)
    lines = [
        "# Phase 2 A0/A1 Development Benchmark",
        "",
        "> Validation evidence only. The confirmatory BANKING77 test split was not downloaded or opened.",
        "",
        "| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        assert isinstance(model, dict)
        metrics = model["metrics"]
        assert isinstance(metrics, dict)
        lines.append(
            "| {model} | {f1:.4f} | {bal:.4f} | {top3:.4f} | {ece:.4f} | {brier:.4f} |".format(
                model=model["model_id"],
                f1=float(metrics["macro_f1"]),
                bal=float(metrics["balanced_accuracy"]),
                top3=float(metrics["top3_recall"]),
                ece=float(metrics["expected_calibration_error_15bin"]),
                brier=float(metrics["multiclass_brier_score"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These are Phase 2 development results on the frozen validation partition. They may guide the bounded model ladder, calibration, and selective-routing work, but they are not release or test-set claims.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_dir: Path) -> dict[str, object]:
    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(CONFIG_PATH)

    with tempfile.TemporaryDirectory(prefix="helix-phase2-a0-a1-") as temp:
        train_csv = Path(temp) / "train.csv"
        _download(spec.train_url, train_csv)
        observed_hash = data_api.sha256_file(train_csv)
        if observed_hash != spec.train_sha256:
            raise ValueError("BANKING77 train checksum does not match the frozen Phase 1 contract")
        source_train = data_api.load_csv(train_csv, "train")

    if len(source_train) != spec.train_examples:
        raise ValueError("BANKING77 train row count drifted")

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
    if len(labels) != spec.intent_count:
        raise ValueError("training labels no longer cover all frozen intents")
    label_to_index = {label: index for index, label in enumerate(labels)}

    x_train = [example.text for example in train]
    x_validation = [example.text for example in validation]
    y_train = np.asarray([label_to_index[example.intent] for example in train], dtype=np.int64)
    y_validation = np.asarray(
        [label_to_index[example.intent] for example in validation],
        dtype=np.int64,
    )

    models = [
        _evaluate_dummy("most_frequent", x_train, y_train, x_validation, y_validation, labels),
        _evaluate_dummy("stratified", x_train, y_train, x_validation, y_validation, labels),
    ]
    a1_result, a1_predicted, a1_probabilities = _evaluate_a1(
        x_train,
        y_train,
        x_validation,
        y_validation,
        labels,
    )
    models.append(a1_result)

    result: dict[str, object] = {
        "run_id": RUN_ID,
        "phase": 2,
        "status": "development_validation_only",
        "test_set_opened": False,
        "selection_partition": "validation",
        "seed": SEED,
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
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "labels": labels,
        "models": models,
        "pending_phase2_evidence": [
            "calibration_comparison",
            "frozen_oos_benchmark",
            "expected_routing_cost",
            "final_risk_coverage_operating_point",
            "A2",
            "A3",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    _write_predictions(
        output_dir / "a1_validation_predictions.csv",
        validation,
        a1_predicted,
        a1_probabilities,
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
        default=REPO_ROOT / "artifacts" / "phase2-routing" / "a0-a1",
    )
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    for model in result["models"]:
        assert isinstance(model, dict)
        metrics = model["metrics"]
        assert isinstance(metrics, dict)
        print(
            f"{model['model_id']}: macro-F1={float(metrics['macro_f1']):.4f}, "
            f"balanced-accuracy={float(metrics['balanced_accuracy']):.4f}, "
            f"top-3={float(metrics['top3_recall']):.4f}"
        )
    print("Confirmatory test opened: false")


if __name__ == "__main__":
    main()

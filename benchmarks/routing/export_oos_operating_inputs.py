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
"""Export public operating inputs for the private routing-cost analysis."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "banking77.json"
A2_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_a2.json"
CALIBRATION_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_calibration.json"
OPERATIONS_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_operations.json"
OOS_BENCHMARK_PATH = REPO_ROOT / "data" / "oos" / "routing_oos_v1.json"
CALIBRATION_SCRIPT_PATH = REPO_ROOT / "benchmarks" / "routing" / "evaluate_calibration.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
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


def _oos_records() -> list[dict[str, str]]:
    payload = json.loads(OOS_BENCHMARK_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    for category in payload["categories"]:
        for index, text in enumerate(category["queries"]):
            records.append(
                {
                    "oos_id": f"{category['id']}:{index:02d}",
                    "category": str(category["id"]),
                    "tier": str(category["tier"]),
                    "text": str(text),
                }
            )
    if len(records) != 160:
        raise ValueError("frozen OOS record count drifted")
    return records


def _temperature_bounds(config: dict[str, object]) -> tuple[float, float]:
    methods = config["methods"]
    assert isinstance(methods, list)
    for method in methods:
        if isinstance(method, dict) and method.get("id") == "temperature_scaling":
            bounds = method["temperature_bounds"]
            assert isinstance(bounds, list) and len(bounds) == 2
            return float(bounds[0]), float(bounds[1])
    raise ValueError("temperature scaling bounds missing")


def _operations() -> tuple[dict[str, dict[str, object]], set[str]]:
    payload = json.loads(OPERATIONS_CONFIG_PATH.read_text(encoding="utf-8"))
    intents = payload["intents"]
    queues = payload["queues"]
    if not isinstance(intents, dict) or not isinstance(queues, list):
        raise TypeError("invalid routing operations configuration")
    return intents, {str(queue) for queue in queues}


def _event_class(
    true_intent: str,
    predicted_intent: str,
    operations: dict[str, dict[str, object]],
) -> str:
    if true_intent == predicted_intent:
        return "correct_route"
    true_record = operations[true_intent]
    predicted_record = operations[predicted_intent]
    if bool(true_record["high_risk"]):
        return "unsafe_high_risk_auto_route"
    if str(true_record["queue"]) == str(predicted_record["queue"]):
        return "wrong_intent_same_queue"
    return "wrong_queue"


def _fold_temperatures(
    calibration: Any,
    validation_probabilities: np.ndarray,
    y_validation: np.ndarray,
    folds: np.ndarray,
    bounds: tuple[float, float],
) -> list[float]:
    return [
        calibration._temperature_fit(
            validation_probabilities[folds != fold],
            y_validation[folds != fold],
            bounds,
        )
        for fold in range(5)
    ]


def _cross_fitted_validation_probabilities(
    calibration: Any,
    raw_probabilities: np.ndarray,
    folds: np.ndarray,
    temperatures: list[float],
) -> np.ndarray:
    output = np.zeros_like(raw_probabilities)
    for fold, temperature in enumerate(temperatures):
        score_mask = folds == fold
        output[score_mask] = calibration._temperature_apply(
            raw_probabilities[score_mask],
            temperature,
        )
    return output


def _fold_specific_oos_confidences(
    calibration: Any,
    raw_oos_probabilities: np.ndarray,
    temperatures: list[float],
) -> np.ndarray:
    return np.stack(
        [
            np.max(
                calibration._temperature_apply(raw_oos_probabilities, temperature),
                axis=1,
            )
            for temperature in temperatures
        ],
        axis=0,
    )


def _write_validation_inputs(
    path: Path,
    validation: list[Any],
    labels: list[str],
    source_revision: str,
    data_api: Any,
    folds: np.ndarray,
    model_exports: dict[str, dict[str, np.ndarray]],
    operations: dict[str, dict[str, object]],
) -> None:
    fieldnames = [
        "sample_id",
        "true_intent",
        "true_queue",
        "true_high_risk",
        "fold",
    ]
    for model_id in ("A1", "A2"):
        fieldnames.extend(
            [
                f"{model_id}_predicted_intent",
                f"{model_id}_predicted_queue",
                f"{model_id}_event",
                f"{model_id}_raw_confidence",
                f"{model_id}_temperature_confidence",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, example in enumerate(validation):
            true_intent = str(example.intent)
            true_record = operations[true_intent]
            row: dict[str, object] = {
                "sample_id": data_api.sample_id(example, source_revision),
                "true_intent": true_intent,
                "true_queue": str(true_record["queue"]),
                "true_high_risk": str(bool(true_record["high_risk"])).lower(),
                "fold": int(folds[index]),
            }
            for model_id in ("A1", "A2"):
                predicted_index = int(model_exports[model_id]["predicted_index"][index])
                predicted_intent = labels[predicted_index]
                predicted_record = operations[predicted_intent]
                row.update(
                    {
                        f"{model_id}_predicted_intent": predicted_intent,
                        f"{model_id}_predicted_queue": str(predicted_record["queue"]),
                        f"{model_id}_event": _event_class(
                            true_intent,
                            predicted_intent,
                            operations,
                        ),
                        f"{model_id}_raw_confidence": (
                            f"{float(model_exports[model_id]['raw_confidence'][index]):.12f}"
                        ),
                        f"{model_id}_temperature_confidence": (
                            f"{float(model_exports[model_id]['temperature_confidence'][index]):.12f}"
                        ),
                    }
                )
            writer.writerow(row)


def _write_oos_inputs(
    path: Path,
    records: list[dict[str, str]],
    labels: list[str],
    model_exports: dict[str, dict[str, np.ndarray]],
    operations: dict[str, dict[str, object]],
) -> None:
    fold_fields = [
        f"{model_id}_temperature_fold_{fold}_confidence"
        for model_id in ("A1", "A2")
        for fold in range(5)
    ]
    fieldnames = ["oos_id", "category", "tier"]
    for model_id in ("A1", "A2"):
        fieldnames.extend(
            [
                f"{model_id}_raw_predicted_intent",
                f"{model_id}_raw_predicted_queue",
                f"{model_id}_raw_confidence",
            ]
        )
    fieldnames.extend(fold_fields)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            row: dict[str, object] = {
                "oos_id": record["oos_id"],
                "category": record["category"],
                "tier": record["tier"],
            }
            for model_id in ("A1", "A2"):
                predicted_index = int(model_exports[model_id]["oos_predicted_index"][index])
                predicted_intent = labels[predicted_index]
                predicted_queue = str(operations[predicted_intent]["queue"])
                row.update(
                    {
                        f"{model_id}_raw_predicted_intent": predicted_intent,
                        f"{model_id}_raw_predicted_queue": predicted_queue,
                        f"{model_id}_raw_confidence": (
                            f"{float(model_exports[model_id]['oos_raw_confidence'][index]):.12f}"
                        ),
                    }
                )
                fold_values = model_exports[model_id]["oos_fold_temperature_confidence"]
                for fold in range(5):
                    row[f"{model_id}_temperature_fold_{fold}_confidence"] = (
                        f"{float(fold_values[fold, index]):.12f}"
                    )
            writer.writerow(row)


def run(output_dir: Path) -> None:
    calibration = _load_module(CALIBRATION_SCRIPT_PATH, "helix_calibration_export")
    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)
    a2_config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    calibration_config = json.loads(CALIBRATION_CONFIG_PATH.read_text(encoding="utf-8"))
    operations, queues = _operations()
    records = _oos_records()

    with tempfile.TemporaryDirectory(prefix="helix-operating-export-") as temp:
        train_csv = Path(temp) / "train.csv"
        _download(spec.train_url, train_csv)
        if data_api.sha256_file(train_csv) != spec.train_sha256:
            raise ValueError("BANKING77 train checksum drifted")
        source_train = data_api.load_csv(train_csv, "train")

    train, validation, quarantined = data_api.split_training_examples(source_train, spec)
    if len(train) != 7904 or len(validation) != 1976 or len(quarantined) != 123:
        raise ValueError("frozen BANKING77 derived counts drifted")

    labels = sorted({example.intent for example in train})
    if set(labels) != set(operations):
        missing = sorted(set(labels) - set(operations))
        extra = sorted(set(operations) - set(labels))
        raise ValueError(f"routing operations coverage drifted: missing={missing}, extra={extra}")
    for intent, record in operations.items():
        if str(record["queue"]) not in queues:
            raise ValueError(f"unknown queue for {intent}: {record['queue']}")

    label_to_index = {label: index for index, label in enumerate(labels)}
    x_train = [example.text for example in train]
    x_validation = [example.text for example in validation]
    y_train = np.asarray(
        [label_to_index[example.intent] for example in train],
        dtype=np.int64,
    )
    y_validation = np.asarray(
        [label_to_index[example.intent] for example in validation],
        dtype=np.int64,
    )
    oos_texts = [record["text"] for record in records]
    x_evaluation = x_validation + oos_texts

    torch.manual_seed(20260818)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    raw_all = {
        "A1": calibration._fit_a1(x_train, y_train, x_evaluation),
        "A2": calibration._fit_a2(x_train, y_train, x_evaluation, a2_config),
    }
    split_at = len(validation)

    cross_fitting = calibration_config["cross_fitting"]
    assert isinstance(cross_fitting, dict)
    folds = calibration._fold_assignment(
        validation,
        labels,
        spec.source_revision,
        str(cross_fitting["salt"]),
        int(cross_fitting["folds"]),
        data_api,
    )
    fold_counts = [int(np.sum(folds == index)) for index in range(5)]
    if fold_counts != [390, 392, 393, 402, 399]:
        raise ValueError("audited calibration fold assignment drifted")

    bounds = _temperature_bounds(calibration_config)
    model_exports: dict[str, dict[str, np.ndarray]] = {}
    for model_id in ("A1", "A2"):
        validation_probabilities = raw_all[model_id][:split_at]
        oos_probabilities = raw_all[model_id][split_at:]
        temperatures = _fold_temperatures(
            calibration,
            validation_probabilities,
            y_validation,
            folds,
            bounds,
        )
        calibrated_validation = _cross_fitted_validation_probabilities(
            calibration,
            validation_probabilities,
            folds,
            temperatures,
        )
        model_exports[model_id] = {
            "predicted_index": np.argmax(validation_probabilities, axis=1),
            "raw_confidence": np.max(validation_probabilities, axis=1),
            "temperature_confidence": np.max(calibrated_validation, axis=1),
            "oos_predicted_index": np.argmax(oos_probabilities, axis=1),
            "oos_raw_confidence": np.max(oos_probabilities, axis=1),
            "oos_fold_temperature_confidence": _fold_specific_oos_confidences(
                calibration,
                oos_probabilities,
                temperatures,
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_validation_inputs(
        output_dir / "validation_inputs.csv",
        validation,
        labels,
        spec.source_revision,
        data_api,
        folds,
        model_exports,
        operations,
    )
    _write_oos_inputs(
        output_dir / "oos_inputs.csv",
        records,
        labels,
        model_exports,
        operations,
    )

    event_counts: dict[str, dict[str, int]] = {}
    for model_id in ("A1", "A2"):
        counts = {
            "correct_route": 0,
            "wrong_intent_same_queue": 0,
            "wrong_queue": 0,
            "unsafe_high_risk_auto_route": 0,
        }
        predicted = model_exports[model_id]["predicted_index"]
        for example, predicted_index in zip(validation, predicted, strict=True):
            event = _event_class(example.intent, labels[int(predicted_index)], operations)
            counts[event] += 1
        event_counts[model_id] = counts

    manifest = {
        "version": "routing-operating-inputs-v1",
        "test_set_opened": False,
        "validation_rows": len(validation),
        "oos_rows": len(records),
        "fold_counts": fold_counts,
        "intent_count": len(labels),
        "queue_count": len(queues),
        "high_risk_intent_count": sum(
            1 for record in operations.values() if bool(record["high_risk"])
        ),
        "full_automation_validation_event_counts": event_counts,
        "private_cost_weights_included": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {len(validation)} validation operating-input rows")
    print(f"Exported {len(records)} frozen OOS operating-input rows")
    print("Preserved five fold-specific temperature confidences per model for OOS")
    print("Private cost weights included: false")
    print("Confirmatory test opened: false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "phase2-routing" / "operating"),
    )
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()

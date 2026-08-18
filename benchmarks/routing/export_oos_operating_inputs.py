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
"""Export frozen OOS confidences for private cost-selection analysis."""

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


def _fold_specific_oos_confidences(
    calibration: Any,
    validation_probabilities: np.ndarray,
    oos_probabilities: np.ndarray,
    y_validation: np.ndarray,
    folds: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    fold_confidences: list[np.ndarray] = []
    for fold in range(5):
        fit_mask = folds != fold
        temperature = calibration._temperature_fit(
            validation_probabilities[fit_mask],
            y_validation[fit_mask],
            bounds,
        )
        calibrated_oos = calibration._temperature_apply(oos_probabilities, temperature)
        fold_confidences.append(np.max(calibrated_oos, axis=1))
    return np.stack(fold_confidences, axis=0)


def run(output_path: Path) -> None:
    calibration = _load_module(CALIBRATION_SCRIPT_PATH, "helix_calibration_export")
    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)
    a2_config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    calibration_config = json.loads(CALIBRATION_CONFIG_PATH.read_text(encoding="utf-8"))
    records = _oos_records()

    with tempfile.TemporaryDirectory(prefix="helix-oos-operating-export-") as temp:
        train_csv = Path(temp) / "train.csv"
        _download(spec.train_url, train_csv)
        if data_api.sha256_file(train_csv) != spec.train_sha256:
            raise ValueError("BANKING77 train checksum drifted")
        source_train = data_api.load_csv(train_csv, "train")

    train, validation, quarantined = data_api.split_training_examples(source_train, spec)
    if len(train) != 7904 or len(validation) != 1976 or len(quarantined) != 123:
        raise ValueError("frozen BANKING77 derived counts drifted")

    labels = sorted({example.intent for example in train})
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

    a1_all = calibration._fit_a1(x_train, y_train, x_evaluation)
    a2_all = calibration._fit_a2(x_train, y_train, x_evaluation, a2_config)
    split_at = len(validation)
    probabilities = {
        "A1": (a1_all[:split_at], a1_all[split_at:]),
        "A2": (a2_all[:split_at], a2_all[split_at:]),
    }

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
    exports: dict[str, dict[str, np.ndarray]] = {}
    for model_id in ("A1", "A2"):
        validation_probabilities, oos_probabilities = probabilities[model_id]
        exports[model_id] = {
            "raw_confidence": np.max(oos_probabilities, axis=1),
            "raw_predicted_index": np.argmax(oos_probabilities, axis=1),
            "fold_temperature_confidence": _fold_specific_oos_confidences(
                calibration,
                validation_probabilities,
                oos_probabilities,
                y_validation,
                folds,
                bounds,
            ),
        }

    fold_fields = [
        f"{model_id}_temperature_fold_{fold}_confidence"
        for model_id in ("A1", "A2")
        for fold in range(5)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "oos_id",
                "category",
                "tier",
                "A1_raw_predicted_intent",
                "A1_raw_confidence",
                "A2_raw_predicted_intent",
                "A2_raw_confidence",
                *fold_fields,
            ],
        )
        writer.writeheader()
        for index, record in enumerate(records):
            row: dict[str, object] = {
                "oos_id": record["oos_id"],
                "category": record["category"],
                "tier": record["tier"],
                "A1_raw_predicted_intent": labels[
                    int(exports["A1"]["raw_predicted_index"][index])
                ],
                "A1_raw_confidence": f"{float(exports['A1']['raw_confidence'][index]):.12f}",
                "A2_raw_predicted_intent": labels[
                    int(exports["A2"]["raw_predicted_index"][index])
                ],
                "A2_raw_confidence": f"{float(exports['A2']['raw_confidence'][index]):.12f}",
            }
            for model_id in ("A1", "A2"):
                fold_values = exports[model_id]["fold_temperature_confidence"]
                for fold in range(5):
                    row[f"{model_id}_temperature_fold_{fold}_confidence"] = (
                        f"{float(fold_values[fold, index]):.12f}"
                    )
            writer.writerow(row)

    print(f"Exported {len(records)} frozen OOS operating-input rows")
    print("Preserved five fold-specific temperature confidences per model")
    print("Confirmatory test opened: false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "phase2-routing" / "operating" / "oos_inputs.csv"),
    )
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()

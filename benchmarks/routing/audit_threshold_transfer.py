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
"""Audit transfer of the selected cross-fitted routing policy to the final calibrator."""

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _oos_texts() -> list[str]:
    payload = json.loads(OOS_BENCHMARK_PATH.read_text(encoding="utf-8"))
    texts = [str(text) for category in payload["categories"] for text in category["queries"]]
    if len(texts) != 160:
        raise ValueError("frozen OOS query count drifted")
    return texts


def _temperature_bounds(config: dict[str, object]) -> tuple[float, float]:
    methods = config["methods"]
    if not isinstance(methods, list):
        raise TypeError("calibration method contract drifted")
    for method in methods:
        if isinstance(method, dict) and method.get("id") == "temperature_scaling":
            bounds = method["temperature_bounds"]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise TypeError("temperature bounds drifted")
            return float(bounds[0]), float(bounds[1])
    raise ValueError("temperature scaling bounds missing")


def _coverage_transfer_threshold(confidence: np.ndarray, accepted: int) -> dict[str, float]:
    if accepted <= 0 or accepted >= len(confidence):
        raise ValueError("transfer audit requires a non-degenerate accepted set")
    ordered = np.sort(confidence)
    rejected_count = len(confidence) - accepted
    max_rejected = float(ordered[rejected_count - 1])
    min_accepted = float(ordered[rejected_count])
    if not max_rejected < min_accepted:
        raise ValueError("full-refit confidence boundary has no open decision plateau")
    return {
        "max_rejected_confidence": max_rejected,
        "min_accepted_confidence": min_accepted,
        "midpoint_threshold": (max_rejected + min_accepted) / 2.0,
    }


def run(
    validation_inputs_path: Path,
    oos_inputs_path: Path,
    cost_results_path: Path,
    output_path: Path,
) -> dict[str, object]:
    validation_inputs = _read_csv(validation_inputs_path)
    oos_inputs = _read_csv(oos_inputs_path)
    cost_results = json.loads(cost_results_path.read_text(encoding="utf-8"))
    if len(validation_inputs) != 1976 or len(oos_inputs) != 160:
        raise ValueError("operating-input counts drifted")
    if cost_results["test_set_opened"] is not False:
        raise ValueError("cost result test-set status drifted")
    final = cost_results["final_development_selection"]
    if final["candidate"] != "A2_temperature":
        raise ValueError("transfer audit is pinned to the selected calibrated A2 candidate")

    calibration = _load_module(CALIBRATION_SCRIPT_PATH, "helix_threshold_transfer_calibration")
    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)
    a2_config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    calibration_config = json.loads(CALIBRATION_CONFIG_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="helix-threshold-transfer-") as temp:
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
    y_train = np.asarray([label_to_index[example.intent] for example in train], dtype=np.int64)
    y_validation = np.asarray(
        [label_to_index[example.intent] for example in validation],
        dtype=np.int64,
    )
    oos_texts = _oos_texts()

    torch.manual_seed(20260818)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw = calibration._fit_a2(x_train, y_train, x_validation + oos_texts, a2_config)
    split_at = len(validation)
    raw_validation = raw[:split_at]
    raw_oos = raw[split_at:]
    bounds = _temperature_bounds(calibration_config)
    full_temperature = calibration._temperature_fit(raw_validation, y_validation, bounds)
    full_validation = calibration._temperature_apply(raw_validation, full_temperature)
    full_oos = calibration._temperature_apply(raw_oos, full_temperature)
    validation_confidence = np.max(full_validation, axis=1)
    oos_confidence = np.max(full_oos, axis=1)

    source_threshold = float(final["threshold"])
    source_accepted_ids = {
        row["sample_id"]
        for row in validation_inputs
        if float(row["A2_temperature_confidence"]) >= source_threshold
    }
    accepted_count = len(source_accepted_ids)
    expected_accepted = round(
        float(final["id_automation_coverage"]) * len(validation_inputs)
    )
    if accepted_count != expected_accepted:
        raise ValueError("selected development coverage drifted")

    plateau = _coverage_transfer_threshold(validation_confidence, accepted_count)
    transfer_threshold = plateau["midpoint_threshold"]
    transfer_accepted_ids = {
        data_api.sample_id(example, spec.source_revision)
        for example, confidence in zip(validation, validation_confidence, strict=True)
        if float(confidence) >= transfer_threshold
    }
    intersection = len(source_accepted_ids & transfer_accepted_ids)
    union = len(source_accepted_ids | transfer_accepted_ids)
    source_events = {row["sample_id"]: row["A2_event"] for row in validation_inputs}

    transfer_event_counts = {
        "correct_route": 0,
        "wrong_intent_same_queue": 0,
        "wrong_queue": 0,
        "unsafe_high_risk_auto_route": 0,
    }
    for sample_id in transfer_accepted_ids:
        transfer_event_counts[source_events[sample_id]] += 1
    transfer_risk = 1.0 - transfer_event_counts["correct_route"] / accepted_count
    transfer_oos_escalation = float(np.mean(oos_confidence < transfer_threshold))

    result: dict[str, object] = {
        "version": "phase2-threshold-transfer-audit-v1",
        "status": "audit_only_no_reselection",
        "test_set_opened": False,
        "selected_development_candidate": "A2_temperature",
        "selected_cross_fitted_threshold": source_threshold,
        "selected_cross_fitted_accepted_rows": accepted_count,
        "selected_cross_fitted_id_risk": float(final["id_selective_risk"]),
        "selected_cross_fitted_oos_escalation": float(final["oos_escalation_recall"]),
        "full_validation_refit_temperature": full_temperature,
        "full_refit_coverage_transfer": {
            **plateau,
            "accepted_rows": len(transfer_accepted_ids),
            "accepted_set_intersection": intersection,
            "accepted_set_union": union,
            "accepted_set_jaccard": intersection / union,
            "changed_acceptance_decisions": len(source_accepted_ids ^ transfer_accepted_ids),
            "id_event_counts": transfer_event_counts,
            "id_selective_risk": transfer_risk,
            "oos_escalation_recall": transfer_oos_escalation,
        },
        "interpretation_rule": (
            "The audit maps the already-selected in-domain coverage to the full-validation refit "
            "probability scale. It does not re-optimize expected cost, model family, calibration "
            "method, or coverage and does not inspect the confirmatory test split."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Confirmatory test opened: false")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-inputs", type=Path, required=True)
    parser.add_argument("--oos-inputs", type=Path, required=True)
    parser.add_argument("--cost-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.validation_inputs, args.oos_inputs, args.cost_results, args.output)


if __name__ == "__main__":
    main()

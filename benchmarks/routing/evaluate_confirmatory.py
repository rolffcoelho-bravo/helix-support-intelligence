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
"""Run the one-shot frozen Phase 2 BANKING77 confirmatory evaluation."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "banking77.json"
A2_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_a2.json"
SELECTED_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_selected_v1.json"
CONFIRMATORY_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_confirmatory_v1.json"
COST_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_cost_matrix.json"
OPERATIONS_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_operations.json"
CALIBRATION_SCRIPT_PATH = REPO_ROOT / "benchmarks" / "routing" / "evaluate_calibration.py"
AUTHORIZATION_TOKEN = "OPEN_FROZEN_TEST_ONCE"


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
    return importlib.import_module("helix_support_intelligence.data.banking77")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-support-intelligence-phase2-confirmatory/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def _assert_close(
    observed: float,
    expected: float,
    name: str,
    tolerance: float = 1e-12,
) -> None:
    if abs(observed - expected) > tolerance:
        raise ValueError(f"{name} drifted: {observed} != {expected}")


def preflight() -> dict[str, object]:
    """Validate frozen confirmatory contracts without touching the test source."""
    confirmatory = _read_json(CONFIRMATORY_CONFIG_PATH)
    selected = _read_json(SELECTED_CONFIG_PATH)
    a2 = _read_json(A2_CONFIG_PATH)
    cost = _read_json(COST_CONFIG_PATH)
    operations = _read_json(OPERATIONS_CONFIG_PATH)
    data = _read_json(DATA_CONFIG_PATH)

    if confirmatory["status"] != "frozen_before_confirmatory_test_open":
        raise ValueError("confirmatory protocol status drifted")
    if confirmatory["governance"]["manual_authorization_token"] != AUTHORIZATION_TOKEN:
        raise ValueError("confirmatory authorization token drifted")
    if selected["version"] != "routing-selected-v1" or selected["model"]["id"] != "A2":
        raise ValueError("selected model drifted")
    if selected["model"]["encoder_revision"] != a2["representation"]["revision"]:
        raise ValueError("selected/A2 encoder revision mismatch")
    if selected["calibration"]["method"] != "temperature_scaling":
        raise ValueError("selected calibration method drifted")

    _assert_close(
        float(selected["calibration"]["temperature"]),
        float(confirmatory["calibration"]["temperature"]),
        "confirmatory temperature",
    )
    _assert_close(
        float(selected["operating_policy"]["threshold"]),
        float(confirmatory["frozen_policies"]["A2_temperature_threshold"]),
        "confirmatory selected threshold",
    )
    _assert_close(
        float(confirmatory["frozen_policies"]["A2_raw_threshold"]),
        0.367217,
        "raw comparator threshold",
    )

    if cost["primary_matrix"]["id"] != confirmatory["cost"]["matrix"]:
        raise ValueError("confirmatory cost matrix drifted")
    if cost["primary_matrix"]["costs"] != confirmatory["cost"]["event_costs"]:
        raise ValueError("confirmatory cost weights drifted")
    if len(operations["intents"]) != 77:
        raise ValueError("routing operation intent count drifted")

    source = data["source"]
    expected = data["split"]["expected"]
    expected_test_rows = int(confirmatory["data"]["confirmatory_rows"])
    if int(source["examples"]["test"]) != expected_test_rows:
        raise ValueError("confirmatory row count contract drifted")
    if source["sha256"]["test"] != confirmatory["data"]["confirmatory_source_sha256"]:
        raise ValueError("confirmatory raw test hash contract drifted")
    expected_derived = confirmatory["data"]["confirmatory_derived_sha256"]
    if expected["jsonl_sha256"]["test"] != expected_derived:
        raise ValueError("confirmatory derived test hash contract drifted")

    return {
        "status": "preflight_passed",
        "test_set_opened": False,
        "protocol": confirmatory["version"],
        "selected_router": selected["version"],
        "model": selected["model"]["id"],
        "temperature": selected["calibration"]["temperature"],
        "selected_threshold": selected["operating_policy"]["threshold"],
        "raw_comparator_threshold": confirmatory["frozen_policies"]["A2_raw_threshold"],
        "bootstrap_replicates": confirmatory["uncertainty"]["replicates"],
    }


def _event(
    true_intent: str,
    predicted_intent: str,
    operation_rows: dict[str, dict[str, object]],
) -> str:
    if predicted_intent == true_intent:
        return "correct_route"
    true_row = operation_rows[true_intent]
    predicted_row = operation_rows[predicted_intent]
    if bool(true_row["high_risk"]):
        return "unsafe_high_risk_auto_route"
    if true_row["queue"] == predicted_row["queue"]:
        return "wrong_intent_same_queue"
    return "wrong_queue"


def _policy_row_costs(
    true_intents: list[str],
    predicted_intents: list[str],
    confidence: np.ndarray,
    threshold: float,
    operation_rows: dict[str, dict[str, object]],
    costs: dict[str, float],
) -> tuple[np.ndarray, Counter[str]]:
    row_costs = np.zeros(len(true_intents), dtype=np.float64)
    counts: Counter[str] = Counter()
    pairs = zip(true_intents, predicted_intents, strict=True)
    for index, (true_intent, predicted_intent) in enumerate(pairs):
        if float(confidence[index]) < threshold:
            event = "human_escalation"
        else:
            event = _event(true_intent, predicted_intent, operation_rows)
        row_costs[index] = float(costs[event])
        counts[event] += 1
    return row_costs, counts


def _risk_at_coverage(
    errors: np.ndarray,
    confidence: np.ndarray,
    sample_ids: list[str],
    coverage: float,
) -> dict[str, float | int]:
    accepted = max(1, min(len(errors), round(coverage * len(errors))))
    order = sorted(
        range(len(errors)),
        key=lambda index: (-float(confidence[index]), sample_ids[index]),
    )
    selected = np.asarray(order[:accepted], dtype=np.int64)
    return {
        "coverage": coverage,
        "accepted": accepted,
        "selective_risk": float(np.mean(errors[selected])),
        "confidence_boundary": float(confidence[selected[-1]]),
    }


def _percentile_interval(values: np.ndarray, confidence_level: float) -> list[float]:
    alpha = (1.0 - confidence_level) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def _verdict(interval: list[float]) -> str:
    lower, upper = interval
    if upper < 0.0:
        return "supported"
    if lower >= 0.0:
        return "unsupported"
    return "inconclusive"


def _paired_bootstrap(
    raw_costs: np.ndarray,
    calibrated_costs: np.ndarray,
    errors: np.ndarray,
    confidence: np.ndarray,
    sample_ids: list[str],
    coverage: float,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_rows = len(errors)
    accepted = round(coverage * n_rows)
    tie_rank = np.empty(n_rows, dtype=np.int64)
    sample_order = sorted(range(n_rows), key=lambda index: sample_ids[index])
    for rank, index in enumerate(sample_order):
        tie_rank[index] = rank

    h3 = np.empty(replicates, dtype=np.float64)
    h4 = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, n_rows, size=n_rows)
        difference = calibrated_costs[sampled] - raw_costs[sampled]
        h3[replicate] = float(np.mean(difference))

        sampled_errors = errors[sampled]
        sampled_confidence = confidence[sampled]
        sampled_tie = tie_rank[sampled]
        order = np.lexsort((sampled_tie, -sampled_confidence))
        selective = float(np.mean(sampled_errors[order[:accepted]]))
        full = float(np.mean(sampled_errors))
        h4[replicate] = selective - full

    return {
        "H3_temperature_minus_raw_cost_CI": _percentile_interval(
            h3,
            confidence_level,
        ),
        "H4_selective_minus_full_risk_CI": _percentile_interval(
            h4,
            confidence_level,
        ),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
    }


def _write_predictions(
    path: Path,
    examples: list[Any],
    labels: list[str],
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    source_revision: str,
    data_api: Any,
) -> None:
    raw_pred = np.argmax(raw_probabilities, axis=1)
    calibrated_pred = np.argmax(calibrated_probabilities, axis=1)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "true_intent",
                "predicted_intent",
                "raw_confidence",
                "calibrated_confidence",
            ],
        )
        writer.writeheader()
        rows = zip(
            examples,
            raw_pred,
            calibrated_pred,
            raw_probabilities,
            calibrated_probabilities,
            strict=True,
        )
        for example, raw_index, calibrated_index, raw_row, calibrated_row in rows:
            if int(raw_index) != int(calibrated_index):
                raise RuntimeError("temperature scaling changed the predicted class")
            writer.writerow(
                {
                    "sample_id": data_api.sample_id(example, source_revision),
                    "true_intent": example.intent,
                    "predicted_intent": labels[int(raw_index)],
                    "raw_confidence": f"{float(np.max(raw_row)):.12f}",
                    "calibrated_confidence": f"{float(np.max(calibrated_row)):.12f}",
                }
            )


def _report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    h3 = result["H3_confirmatory"]
    h4 = result["H4_confirmatory"]
    return "\n".join(
        [
            "# Phase 2 Registered Confirmatory Evaluation",
            "",
            "> One-shot BANKING77 test evaluation of the frozen routing-selected-v1 policy.",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Macro-F1 | {float(metrics['macro_f1']):.4f} |",
            f"| Balanced accuracy | {float(metrics['balanced_accuracy']):.4f} |",
            f"| Top-3 recall | {float(metrics['top3_recall']):.4f} |",
            f"| ECE | {float(metrics['expected_calibration_error_15bin']):.4f} |",
            f"| Brier | {float(metrics['multiclass_brier_score']):.4f} |",
            "",
            f"H3 confirmatory verdict: **{h3['verdict']}**.",
            f"H4 confirmatory verdict: **{h4['verdict']}**.",
            "",
            (
                "The frozen synthetic OOS benchmark was already used during development and is "
                "not counted as independent confirmatory evidence."
            ),
            "",
        ]
    )


def run(output_dir: Path, authorization: str) -> dict[str, object]:
    preflight_result = preflight()
    if authorization != AUTHORIZATION_TOKEN:
        message = "confirmatory test access requires the exact frozen authorization token"
        raise PermissionError(message)

    confirmatory = _read_json(CONFIRMATORY_CONFIG_PATH)
    selected = _read_json(SELECTED_CONFIG_PATH)
    a2_config = _read_json(A2_CONFIG_PATH)
    cost_config = _read_json(COST_CONFIG_PATH)
    operations_config = _read_json(OPERATIONS_CONFIG_PATH)

    seed = int(a2_config["evaluation"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)
    with tempfile.TemporaryDirectory(prefix="helix-phase2-confirmatory-") as temp:
        train_csv = Path(temp) / "train.csv"
        test_csv = Path(temp) / "test.csv"
        _download(spec.train_url, train_csv)
        if data_api.sha256_file(train_csv) != spec.train_sha256:
            raise ValueError("BANKING77 train checksum drifted")
        _download(spec.test_url, test_csv)
        if data_api.sha256_file(test_csv) != spec.test_sha256:
            raise ValueError("BANKING77 confirmatory test checksum drifted")
        source_train = data_api.load_csv(train_csv, "train")
        source_test = data_api.load_csv(test_csv, "test")

    train, validation, quarantined = data_api.split_training_examples(source_train, spec)
    data_api.verify_derived_contract(
        train,
        validation,
        source_test,
        quarantined,
        spec,
    )
    test_bytes = data_api.canonical_jsonl_bytes(source_test, spec.source_revision)
    test_hash = data_api.sha256_bytes(test_bytes)
    if test_hash != confirmatory["data"]["confirmatory_derived_sha256"]:
        raise ValueError("confirmatory derived test hash drifted")

    labels = sorted({example.intent for example in train})
    label_to_index = {label: index for index, label in enumerate(labels)}
    operation_rows = operations_config["intents"]
    if set(operation_rows) != set(labels):
        raise ValueError("routing operations and trained label vocabulary drifted")

    x_train = [example.text for example in train]
    y_train = np.asarray(
        [label_to_index[example.intent] for example in train],
        dtype=np.int64,
    )
    x_test = [example.text for example in source_test]
    y_test = np.asarray(
        [label_to_index[example.intent] for example in source_test],
        dtype=np.int64,
    )
    true_intents = [example.intent for example in source_test]
    sample_ids = [data_api.sample_id(example, spec.source_revision) for example in source_test]

    calibration = _load_module(
        CALIBRATION_SCRIPT_PATH,
        "helix_confirmatory_calibration",
    )
    raw_probabilities = calibration._fit_a2(x_train, y_train, x_test, a2_config)
    temperature = float(selected["calibration"]["temperature"])
    calibrated_probabilities = calibration._temperature_apply(
        raw_probabilities,
        temperature,
    )
    raw_predicted = np.argmax(raw_probabilities, axis=1)
    calibrated_predicted = np.argmax(calibrated_probabilities, axis=1)
    if not np.array_equal(raw_predicted, calibrated_predicted):
        raise RuntimeError("temperature scaling changed discrete A2 predictions")

    metrics = calibration._metrics(y_test, calibrated_probabilities)
    raw_confidence = np.max(raw_probabilities, axis=1)
    calibrated_confidence = np.max(calibrated_probabilities, axis=1)
    predicted_intents = [labels[int(index)] for index in raw_predicted]
    cost_rows = cost_config["primary_matrix"]["costs"]
    primary_costs = {key: float(value) for key, value in cost_rows.items()}

    raw_threshold = float(confirmatory["frozen_policies"]["A2_raw_threshold"])
    calibrated_threshold = float(confirmatory["frozen_policies"]["A2_temperature_threshold"])
    raw_costs, raw_events = _policy_row_costs(
        true_intents,
        predicted_intents,
        raw_confidence,
        raw_threshold,
        operation_rows,
        primary_costs,
    )
    calibrated_costs, calibrated_events = _policy_row_costs(
        true_intents,
        predicted_intents,
        calibrated_confidence,
        calibrated_threshold,
        operation_rows,
        primary_costs,
    )

    errors = raw_predicted != y_test
    coverage = float(confirmatory["H4_confirmatory"]["primary_coverage"])
    fixed_coverage = _risk_at_coverage(
        errors,
        calibrated_confidence,
        sample_ids,
        coverage,
    )
    full_risk = float(np.mean(errors))
    threshold_mask = calibrated_confidence >= calibrated_threshold
    threshold_accepted = int(np.sum(threshold_mask))
    threshold_risk = None
    if threshold_accepted > 0:
        threshold_risk = float(np.mean(errors[threshold_mask]))

    uncertainty = confirmatory["uncertainty"]
    bootstrap = _paired_bootstrap(
        raw_costs,
        calibrated_costs,
        errors,
        calibrated_confidence,
        sample_ids,
        coverage,
        int(uncertainty["replicates"]),
        int(uncertainty["seed"]),
        float(uncertainty["confidence_level"]),
    )
    h3_difference = float(np.mean(calibrated_costs - raw_costs))
    h4_difference = float(fixed_coverage["selective_risk"]) - full_risk
    h3_interval = bootstrap["H3_temperature_minus_raw_cost_CI"]
    h4_interval = bootstrap["H4_selective_minus_full_risk_CI"]
    assert isinstance(h3_interval, list)
    assert isinstance(h4_interval, list)

    fixed_coverage_points = {
        str(value): _risk_at_coverage(
            errors,
            calibrated_confidence,
            sample_ids,
            value,
        )
        for value in (0.50, 0.70, 0.75, 0.90)
    }

    result: dict[str, object] = {
        "run_id": "phase2-routing-confirmatory-v1",
        "phase": 2,
        "status": "registered_confirmatory_result",
        "test_set_opened": True,
        "protocol": confirmatory["version"],
        "preflight": preflight_result,
        "data": {
            "source_revision": spec.source_revision,
            "test_rows": len(source_test),
            "source_test_sha256": spec.test_sha256,
            "derived_test_sha256": test_hash,
        },
        "frozen_configuration": {
            "model": "A2",
            "temperature": temperature,
            "raw_threshold": raw_threshold,
            "calibrated_threshold": calibrated_threshold,
        },
        "metrics": metrics,
        "fixed_coverage_risk": fixed_coverage_points,
        "frozen_threshold": {
            "accepted": threshold_accepted,
            "coverage": threshold_accepted / len(source_test),
            "selective_risk": threshold_risk,
            "mean_in_domain_cost": float(np.mean(calibrated_costs)),
            "event_counts": dict(calibrated_events),
        },
        "H3_confirmatory": {
            "primary_estimand": confirmatory["H3_confirmatory"]["primary_estimand"],
            "raw_mean_in_domain_cost": float(np.mean(raw_costs)),
            "calibrated_mean_in_domain_cost": float(np.mean(calibrated_costs)),
            "temperature_minus_raw": h3_difference,
            "confidence_interval_95": h3_interval,
            "verdict": _verdict(h3_interval),
            "raw_event_counts": dict(raw_events),
            "calibrated_event_counts": dict(calibrated_events),
            "oos_independence_note": confirmatory["H3_confirmatory"]["independence_reason"],
        },
        "H4_confirmatory": {
            "primary_estimand": confirmatory["H4_confirmatory"]["primary_estimand"],
            "full_automation_risk": full_risk,
            "selective_risk_at_75pct_coverage": fixed_coverage["selective_risk"],
            "selective_minus_full_risk": h4_difference,
            "confidence_interval_95": h4_interval,
            "verdict": _verdict(h4_interval),
        },
        "uncertainty": bootstrap,
        "governance": {
            "model_changed_after_test": False,
            "calibration_changed_after_test": False,
            "threshold_changed_after_test": False,
            "cost_matrix_changed_after_test": False,
            "oos_benchmark_counted_as_independent_confirmatory_evidence": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_report(result), encoding="utf-8")
    _write_predictions(
        output_dir / "test_predictions.csv",
        source_test,
        labels,
        raw_probabilities,
        calibrated_probabilities,
        spec.source_revision,
        data_api,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "phase2-routing" / "confirmatory",
    )
    parser.add_argument("--authorize-test-open", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.preflight_only or args.authorize_test_open is None:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        print("Confirmatory test opened: false")
        return

    result = run(args.output_dir.resolve(), args.authorize_test_open)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Confirmatory test opened: true")


if __name__ == "__main__":
    main()

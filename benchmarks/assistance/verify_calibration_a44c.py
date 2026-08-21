"""Independently reconstruct and audit A4.4c calibration arithmetic from raw logits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

RELATION_TO_LABEL = {"CONTRADICTED": 0, "UNKNOWN": 1, "ENTAILED": 2}
LABEL_TO_RELATION = {value: key for key, value in RELATION_TO_LABEL.items()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path} JSONL rows must be objects.")
            rows.append(value)
    return rows


def _nll(rows: list[dict[str, Any]], temperature: float) -> float:
    losses: list[float] = []
    for row in rows:
        logits = [float(value) / temperature for value in row["logits"]]
        maximum = max(logits)
        logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
        losses.append(logsumexp - logits[int(row["gold_label"])])
    return sum(losses) / len(losses)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _grid() -> list[float]:
    return [integer / 100.0 for integer in range(25, 401)]


def verify(output_dir: Path) -> dict[str, Any]:
    result = _read_json(output_dir / "results.json")
    rows = _read_jsonl(output_dir / "calibration_pair_logits.jsonl")
    grid_rows = _read_jsonl(output_dir / "temperature_grid.jsonl")
    if not rows:
        raise RuntimeError("A4.4c raw calibration logits are missing.")
    if any(row.get("split") != "calibration" for row in rows):
        raise RuntimeError("A4.4c raw artifact contains a non-calibration semantic pair.")
    if len({str(row["pair_id"]) for row in rows}) != len(rows):
        raise RuntimeError("A4.4c raw semantic pair ids are not unique.")
    if len(grid_rows) != 376:
        raise RuntimeError("A4.4c temperature grid must contain 376 points.")

    for row in rows:
        gold_relation = str(row["gold_relation"])
        gold_label = int(row["gold_label"])
        if RELATION_TO_LABEL[gold_relation] != gold_label:
            raise RuntimeError("Gold relation/label mapping drifted.")
        raw_argmax = _argmax([float(value) for value in row["logits"]])
        if raw_argmax != int(row["raw_argmax_label"]):
            raise RuntimeError("Stored raw argmax label does not reconstruct from logits.")
        if LABEL_TO_RELATION[raw_argmax] != str(row["raw_argmax_relation"]):
            raise RuntimeError("Stored raw argmax relation does not reconstruct from logits.")
        if bool(row["raw_correct"]) != (raw_argmax == gold_label):
            raise RuntimeError("Stored raw correctness does not reconstruct from logits.")

    reconstructed_grid = [
        {"temperature": temperature, "nll": _nll(rows, temperature)} for temperature in _grid()
    ]
    for stored, reconstructed in zip(grid_rows, reconstructed_grid, strict=True):
        if float(stored["temperature"]) != float(reconstructed["temperature"]):
            raise RuntimeError("Stored temperature grid point drifted.")
        if not math.isclose(
            float(stored["nll"]), float(reconstructed["nll"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("Stored temperature-grid NLL does not reconstruct.")

    selected = min(
        reconstructed_grid,
        key=lambda row: (float(row["nll"]), float(row["temperature"])),
    )
    selected_temperature = float(selected["temperature"])
    raw_nll = _nll(rows, 1.0)
    calibrated_nll = _nll(rows, selected_temperature)
    raw_correct = sum(bool(row["raw_correct"]) for row in rows)
    raw_accuracy = raw_correct / len(rows)

    calibration = result["calibration"]
    sealed = result["sealed_boundaries"]
    checks = {
        "semantic_pair_count": int(calibration["semantic_pairs"]) == len(rows),
        "grid_point_count": int(calibration["grid_points"]) == 376,
        "selected_temperature": float(calibration["selected_temperature"]) == selected_temperature,
        "raw_nll": math.isclose(
            float(calibration["raw_nll_at_temperature_1"]), raw_nll, rel_tol=0.0, abs_tol=1e-12
        ),
        "calibrated_nll": math.isclose(
            float(calibration["selected_temperature_nll"]),
            calibrated_nll,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "raw_accuracy": math.isclose(
            float(calibration["raw_argmax_accuracy"]), raw_accuracy, rel_tol=0.0, abs_tol=1e-15
        ),
        "argmax_preserved": bool(calibration["argmax_preserved_after_temperature"]),
        "validation_materialized_zero": int(sealed["validation_case_rows_materialized"]) == 0,
        "validation_cases_scored_zero": int(sealed["validation_case_rows_scored"]) == 0,
        "validation_metrics_zero": int(sealed["validation_metrics_computed"]) == 0,
        "confirmatory_scored_zero": int(sealed["confirmatory_queries_scored"]) == 0,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"A4.4c independent audit failed: {failed}")

    audit = {
        "audit_id": "phase4-assistance-a4.4c-calibration-reconstruction-v1",
        "status": "PASSED_CALIBRATION_ONLY_RECONSTRUCTION",
        "semantic_pairs": len(rows),
        "selected_temperature": selected_temperature,
        "raw_nll": raw_nll,
        "calibrated_nll": calibrated_nll,
        "raw_argmax_accuracy": raw_accuracy,
        "checks": checks,
        "validation_cases_materialized": 0,
        "validation_results_opened": False,
        "confirmatory_results_opened": False,
    }
    (output_dir / "post_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "post_audit.md").write_text(
        "\n".join(
            [
                "# A4.4c independent calibration reconstruction",
                "",
                "**Status: PASSED_CALIBRATION_ONLY_RECONSTRUCTION**",
                "",
                f"Semantic pairs independently reconstructed: **{len(rows)}**.",
                (
                    "Selected temperature independently reconstructed: "
                    f"**{selected_temperature:.2f}**."
                ),
                f"Raw NLL: **{raw_nll:.6f}**.",
                f"Calibrated NLL: **{calibrated_nll:.6f}**.",
                "",
                "No validation case was materialized or scored by A4.4c.",
                "No confirmatory result was opened by this reconstruction.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()

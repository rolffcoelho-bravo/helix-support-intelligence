"""Reproduce the frozen Phase 2 routing cost and operating-point selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
COST_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_cost_matrix.json"
FOLD_COUNTS = {0: 390, 1: 392, 2: 393, 3: 402, 4: 399}
EVENTS = (
    "correct_route",
    "wrong_intent_same_queue",
    "wrong_queue",
    "unsafe_high_risk_auto_route",
    "human_escalation",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(
    validation_rows: list[dict[str, str]],
    oos_rows: list[dict[str, str]],
) -> None:
    if len(validation_rows) != 1976:
        raise ValueError("validation operating-input count drifted")
    if len(oos_rows) != 160:
        raise ValueError("OOS operating-input count drifted")
    observed_folds = Counter(int(row["fold"]) for row in validation_rows)
    if dict(sorted(observed_folds.items())) != FOLD_COUNTS:
        raise ValueError(f"calibration fold counts drifted: {observed_folds}")
    for model_id in ("A1", "A2"):
        events = {row[f"{model_id}_event"] for row in validation_rows}
        if not events <= set(EVENTS[:-1]):
            raise ValueError(f"unknown {model_id} automatic-route event: {events}")
        for fold in range(5):
            column = f"{model_id}_temperature_fold_{fold}_confidence"
            if column not in oos_rows[0]:
                raise ValueError(f"missing OOS fold confidence: {column}")


def _build_items(
    validation_rows: list[dict[str, str]],
    oos_rows: list[dict[str, str]],
    model_id: str,
    variant: str,
    costs: dict[str, float],
    oos_prevalence: float,
) -> list[tuple[float, float, float, str, str]]:
    items: list[tuple[float, float, float, str, str]] = []
    validation_weight = (1.0 - oos_prevalence) / len(validation_rows)
    id_confidence = (
        f"{model_id}_raw_confidence" if variant == "raw" else f"{model_id}_temperature_confidence"
    )
    for row in validation_rows:
        event = row[f"{model_id}_event"]
        items.append(
            (
                float(row[id_confidence]),
                validation_weight,
                float(costs[event]),
                "id",
                event,
            )
        )

    if variant == "raw":
        oos_weight = oos_prevalence / len(oos_rows)
        for row in oos_rows:
            items.append(
                (
                    float(row[f"{model_id}_raw_confidence"]),
                    oos_weight,
                    float(costs["wrong_queue"]),
                    "oos",
                    "wrong_queue",
                )
            )
    elif variant == "temperature":
        total_validation = len(validation_rows)
        for fold, heldout_rows in FOLD_COUNTS.items():
            oos_weight = oos_prevalence * (heldout_rows / total_validation) / len(oos_rows)
            confidence_key = f"{model_id}_temperature_fold_{fold}_confidence"
            for row in oos_rows:
                items.append(
                    (
                        float(row[confidence_key]),
                        oos_weight,
                        float(costs["wrong_queue"]),
                        "oos",
                        "wrong_queue",
                    )
                )
    else:
        raise ValueError(f"unknown score variant: {variant}")
    return items


def _snapshot(
    threshold: float,
    expected_cost: float,
    validation_rows: list[dict[str, str]],
    oos_prevalence: float,
    id_event_counts: Counter[str],
    id_automatic: int,
    id_correct_automatic: int,
    oos_mixture_weight_automatic: float,
    costs: dict[str, float],
) -> dict[str, Any]:
    total_id = len(validation_rows)
    oos_automatic_rate = (
        0.0 if oos_prevalence == 0.0 else oos_mixture_weight_automatic / oos_prevalence
    )
    id_mean_cost = sum(id_event_counts[event] * costs[event] for event in EVENTS) / total_id
    oos_mean_cost = (
        oos_automatic_rate * costs["wrong_queue"]
        + (1.0 - oos_automatic_rate) * costs["human_escalation"]
    )
    return {
        "threshold": threshold,
        "expected_cost": expected_cost,
        "mean_id_cost": id_mean_cost,
        "mean_oos_cost": oos_mean_cost,
        "id_automation_coverage": id_automatic / total_id,
        "id_selective_risk": (
            None if id_automatic == 0 else 1.0 - id_correct_automatic / id_automatic
        ),
        "oos_escalation_recall": 1.0 - oos_automatic_rate,
        "unsafe_high_risk_rate": (
            (1.0 - oos_prevalence) * id_event_counts["unsafe_high_risk_auto_route"] / total_id
        ),
        "wrong_queue_rate": (
            (1.0 - oos_prevalence) * id_event_counts["wrong_queue"] / total_id
            + oos_mixture_weight_automatic
        ),
        "id_event_counts": dict(id_event_counts),
    }


def _selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        round(float(result["expected_cost"]), 12),
        round(float(result["unsafe_high_risk_rate"]), 12),
        round(float(result["wrong_queue_rate"]), 12),
        -float(result["id_automation_coverage"]),
    )


def _optimize(
    validation_rows: list[dict[str, str]],
    oos_rows: list[dict[str, str]],
    model_id: str,
    variant: str,
    costs: dict[str, float],
    oos_prevalence: float,
) -> dict[str, Any]:
    items = sorted(
        _build_items(
            validation_rows,
            oos_rows,
            model_id,
            variant,
            costs,
            oos_prevalence,
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    human_cost = float(costs["human_escalation"])
    expected_cost = human_cost
    id_event_counts: Counter[str] = Counter({"human_escalation": len(validation_rows)})
    id_automatic = 0
    id_correct_automatic = 0
    oos_mixture_weight_automatic = 0.0
    best: dict[str, Any] | None = None

    def consider(threshold: float) -> None:
        nonlocal best
        result = _snapshot(
            threshold,
            expected_cost,
            validation_rows,
            oos_prevalence,
            id_event_counts,
            id_automatic,
            id_correct_automatic,
            oos_mixture_weight_automatic,
            costs,
        )
        if best is None or _selection_key(result) < _selection_key(best):
            best = result

    index = 0
    while index < len(items) and items[index][0] >= 1.0:
        confidence = items[index][0]
        while index < len(items) and items[index][0] == confidence:
            _, weight, auto_cost, kind, event = items[index]
            expected_cost += weight * (auto_cost - human_cost)
            if kind == "id":
                id_event_counts["human_escalation"] -= 1
                id_event_counts[event] += 1
                id_automatic += 1
                if event == "correct_route":
                    id_correct_automatic += 1
            else:
                oos_mixture_weight_automatic += weight
            index += 1
    consider(1.0)

    while index < len(items):
        confidence = items[index][0]
        while index < len(items) and items[index][0] == confidence:
            _, weight, auto_cost, kind, event = items[index]
            expected_cost += weight * (auto_cost - human_cost)
            if kind == "id":
                id_event_counts["human_escalation"] -= 1
                id_event_counts[event] += 1
                id_automatic += 1
                if event == "correct_route":
                    id_correct_automatic += 1
            else:
                oos_mixture_weight_automatic += weight
            index += 1
        consider(confidence)
    consider(0.0)

    if best is None:
        raise RuntimeError("threshold optimization produced no candidate")
    return best


def _evaluate_threshold(
    validation_rows: list[dict[str, str]],
    oos_rows: list[dict[str, str]],
    model_id: str,
    variant: str,
    threshold: float,
    costs: dict[str, float],
    oos_prevalence: float,
) -> dict[str, Any]:
    id_event_counts: Counter[str] = Counter()
    id_automatic = 0
    id_correct = 0
    id_confidence = (
        f"{model_id}_raw_confidence" if variant == "raw" else f"{model_id}_temperature_confidence"
    )
    for row in validation_rows:
        if float(row[id_confidence]) >= threshold:
            event = row[f"{model_id}_event"]
            id_event_counts[event] += 1
            id_automatic += 1
            if event == "correct_route":
                id_correct += 1
        else:
            id_event_counts["human_escalation"] += 1

    if variant == "raw":
        oos_automatic_rate = sum(
            float(row[f"{model_id}_raw_confidence"]) >= threshold for row in oos_rows
        ) / len(oos_rows)
    else:
        oos_automatic_rate = sum(
            (FOLD_COUNTS[fold] / len(validation_rows))
            * sum(
                float(row[f"{model_id}_temperature_fold_{fold}_confidence"]) >= threshold
                for row in oos_rows
            )
            / len(oos_rows)
            for fold in range(5)
        )

    id_mean_cost = sum(id_event_counts[event] * costs[event] for event in EVENTS) / len(
        validation_rows
    )
    oos_mean_cost = (
        oos_automatic_rate * costs["wrong_queue"]
        + (1.0 - oos_automatic_rate) * costs["human_escalation"]
    )
    expected_cost = (1.0 - oos_prevalence) * id_mean_cost + oos_prevalence * oos_mean_cost
    return {
        "threshold": threshold,
        "expected_cost": expected_cost,
        "mean_id_cost": id_mean_cost,
        "mean_oos_cost": oos_mean_cost,
        "id_automation_coverage": id_automatic / len(validation_rows),
        "id_selective_risk": (None if id_automatic == 0 else 1.0 - id_correct / id_automatic),
        "oos_escalation_recall": 1.0 - oos_automatic_rate,
        "unsafe_high_risk_rate": (
            (1.0 - oos_prevalence)
            * id_event_counts["unsafe_high_risk_auto_route"]
            / len(validation_rows)
        ),
        "wrong_queue_rate": (
            (1.0 - oos_prevalence) * id_event_counts["wrong_queue"] / len(validation_rows)
            + oos_prevalence * oos_automatic_rate
        ),
        "id_event_counts": dict(id_event_counts),
    }


def _selective_risk_at_coverage(
    validation_rows: list[dict[str, str]],
    model_id: str,
    coverage: float,
) -> dict[str, float | int]:
    confidence_key = f"{model_id}_temperature_confidence"
    ordered = sorted(
        validation_rows,
        key=lambda row: float(row[confidence_key]),
        reverse=True,
    )
    accepted = max(1, min(len(ordered), round(coverage * len(ordered))))
    selected = ordered[:accepted]
    correct = sum(row[f"{model_id}_event"] == "correct_route" for row in selected)
    return {
        "coverage": coverage,
        "accepted": accepted,
        "selective_risk": 1.0 - correct / accepted,
        "confidence_boundary": float(selected[-1][confidence_key]),
    }


def _matrix_map(cost_config: dict[str, Any]) -> dict[str, dict[str, float]]:
    output = {cost_config["primary_matrix"]["id"]: cost_config["primary_matrix"]["costs"]}
    for matrix in cost_config["sensitivity_matrices"]:
        output[matrix["id"]] = matrix["costs"]
    return output


def _candidate_key(
    result: dict[str, Any],
    candidate: str,
) -> tuple[float, float, float, float, int]:
    return (
        round(float(result["expected_cost"]), 12),
        round(float(result["unsafe_high_risk_rate"]), 12),
        round(float(result["wrong_queue_rate"]), 12),
        -float(result["id_automation_coverage"]),
        0 if candidate.startswith("A1") else 1,
    )


def _decision_hashes(
    validation_rows: list[dict[str, str]],
    oos_rows: list[dict[str, str]],
    model_id: str,
    threshold: float,
) -> dict[str, Any]:
    id_lines = [
        f"{row['sample_id']}\t{int(float(row[f'{model_id}_temperature_confidence']) >= threshold)}"
        for row in validation_rows
    ]
    id_hash = hashlib.sha256(("\n".join(id_lines) + "\n").encode()).hexdigest()
    oos_hashes: dict[str, str] = {}
    for fold in range(5):
        column = f"{model_id}_temperature_fold_{fold}_confidence"
        lines = [f"{row['oos_id']}\t{int(float(row[column]) >= threshold)}" for row in oos_rows]
        oos_hashes[str(fold)] = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    return {
        "validation_auto_route_sha256": id_hash,
        "oos_auto_route_sha256_by_fold": oos_hashes,
    }


def _markdown_report(result: dict[str, Any]) -> str:
    primary = result["primary_results"]
    final = result["final_development_selection"]
    h3 = result["H3_development"]
    h4 = result["H4_development"]
    return "\n".join(
        [
            "# Phase 2 Routing Cost and Operating-Point Benchmark",
            "",
            (
                "> Development evidence using synthetic scenario costs. The weights are not "
                "real-bank economics, and the confirmatory BANKING77 test split remains "
                "unopened."
            ),
            "",
            (
                "| Candidate | Threshold | Expected cost | ID coverage | "
                "ID selective risk | OOS escalation |"
            ),
            "|---|---:|---:|---:|---:|---:|",
            *[
                (
                    "| {candidate} | {threshold:.6f} | {cost:.4f} | {coverage:.2%} | "
                    "{risk:.2%} | {oos:.2%} |"
                ).format(
                    candidate=candidate,
                    threshold=float(primary[candidate]["threshold"]),
                    cost=float(primary[candidate]["expected_cost"]),
                    coverage=float(primary[candidate]["id_automation_coverage"]),
                    risk=float(primary[candidate]["id_selective_risk"]),
                    oos=float(primary[candidate]["oos_escalation_recall"]),
                )
                for candidate in (
                    "A1_raw",
                    "A1_temperature",
                    "A2_raw",
                    "A2_temperature",
                )
            ],
            "",
            f"Frozen calibrated development candidate: **{final['candidate']}**.",
            "",
            (
                "H3 development status for A2: **unsupported**; temperature scaling "
                f"changes minimum expected cost by {float(h3['A2']['temperature_minus_raw']):+.4f}."
            ),
            (
                "H4 development status: **supported**; the cost-selected policy reduces "
                "expected cost versus full automation by {:.4f}.".format(
                    float(h4["expected_cost_reduction_vs_full_automation"])
                )
            ),
            "",
        ]
    )


def run(
    validation_path: Path,
    oos_path: Path,
    cost_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validation_rows = _read_csv(validation_path)
    oos_rows = _read_csv(oos_path)
    _validate_inputs(validation_rows, oos_rows)
    cost_config = json.loads(cost_config_path.read_text(encoding="utf-8"))
    if cost_config["status"] != "published_after_private_selection_audit":
        raise ValueError("public cost matrix publication status drifted")

    matrices = _matrix_map(cost_config)
    primary_matrix_id = cost_config["primary_matrix"]["id"]
    primary_prevalence = float(cost_config["population_mixture"]["primary_oos_prevalence"])
    prevalences = {
        primary_prevalence,
        *(
            float(value)
            for value in cost_config["population_mixture"]["sensitivity_oos_prevalence"]
        ),
    }

    scenarios: dict[str, Any] = {}
    for matrix_id, costs in matrices.items():
        scenarios[matrix_id] = {}
        for prevalence in sorted(prevalences):
            cell: dict[str, Any] = {}
            for model_id in ("A1", "A2"):
                for variant in ("raw", "temperature"):
                    cell[f"{model_id}_{variant}"] = _optimize(
                        validation_rows,
                        oos_rows,
                        model_id,
                        variant,
                        costs,
                        prevalence,
                    )
            scenarios[matrix_id][f"{prevalence:.2f}"] = cell

    primary = scenarios[primary_matrix_id][f"{primary_prevalence:.2f}"]
    calibrated_candidates = ("A1_temperature", "A2_temperature")
    final_candidate = min(
        calibrated_candidates,
        key=lambda candidate: _candidate_key(primary[candidate], candidate),
    )
    final_model = final_candidate.split("_")[0]
    final_result = primary[final_candidate]
    primary_costs = matrices[primary_matrix_id]
    full_automation = _evaluate_threshold(
        validation_rows,
        oos_rows,
        final_model,
        "temperature",
        0.0,
        primary_costs,
        primary_prevalence,
    )
    all_escalate = _evaluate_threshold(
        validation_rows,
        oos_rows,
        final_model,
        "temperature",
        1.0,
        primary_costs,
        primary_prevalence,
    )

    h3: dict[str, Any] = {}
    for model_id in ("A1", "A2"):
        raw = primary[f"{model_id}_raw"]
        calibrated = primary[f"{model_id}_temperature"]
        h3[model_id] = {
            "raw_min_expected_cost": raw["expected_cost"],
            "temperature_min_expected_cost": calibrated["expected_cost"],
            "temperature_minus_raw": (calibrated["expected_cost"] - raw["expected_cost"]),
            "supported_on_development_primary_cost": (
                calibrated["expected_cost"] < raw["expected_cost"]
            ),
        }

    h4 = {
        "candidate": final_candidate,
        "cost_selected_operating_point": final_result,
        "full_automation": full_automation,
        "all_escalate": all_escalate,
        "expected_cost_reduction_vs_full_automation": (
            full_automation["expected_cost"] - final_result["expected_cost"]
        ),
        "selective_risk_reduction_vs_full_automation": (
            float(full_automation["id_selective_risk"]) - float(final_result["id_selective_risk"])
        ),
        "fixed_coverage": [
            _selective_risk_at_coverage(validation_rows, final_model, coverage)
            for coverage in (0.50, 0.70, 0.90)
        ],
    }

    sensitivity_winners: dict[str, str] = {}
    for matrix_id, matrix_scenarios in scenarios.items():
        for prevalence, cell in matrix_scenarios.items():
            sensitivity_winners[f"{matrix_id}@oos={prevalence}"] = min(
                calibrated_candidates,
                key=lambda candidate: _candidate_key(cell[candidate], candidate),
            )

    result: dict[str, Any] = {
        "version": "phase2-routing-cost-selection-v1",
        "status": "public_development_reproduction",
        "test_set_opened": False,
        "input_hashes": {
            "validation_inputs_sha256": _sha256_file(validation_path),
            "oos_inputs_sha256": _sha256_file(oos_path),
        },
        "cost_matrix_version": cost_config["version"],
        "primary_matrix_id": primary_matrix_id,
        "primary_oos_prevalence": primary_prevalence,
        "primary_results": primary,
        "final_development_selection": {
            "candidate": final_candidate,
            **final_result,
            "decision_hashes": _decision_hashes(
                validation_rows,
                oos_rows,
                final_model,
                float(final_result["threshold"]),
            ),
        },
        "H3_development": h3,
        "H4_development": h4,
        "sensitivity_scenarios": scenarios,
        "sensitivity_calibrated_winners": sensitivity_winners,
        "final_candidate_stable_across_registered_sensitivity": all(
            winner == final_candidate for winner in sensitivity_winners.values()
        ),
        "limitations": {
            "costs_are_synthetic_scenario_assumptions": True,
            "costs_are_real_bank_economics": False,
            "oos_prevalence_is_production_estimate": False,
            "confirmatory_test_pending": True,
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
    parser.add_argument("--validation-inputs", type=Path, required=True)
    parser.add_argument("--oos-inputs", type=Path, required=True)
    parser.add_argument(
        "--cost-config",
        type=Path,
        default=COST_CONFIG_PATH,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.validation_inputs,
        args.oos_inputs,
        args.cost_config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

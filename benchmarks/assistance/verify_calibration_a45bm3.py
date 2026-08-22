"""Reconstruct A4.5b-M3 parameter selection from immutable raw SCEC scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scec_calibration_a45bm2 import build_suite, manifest
from scec_calibration_core_a45bm3 import candidate_key, evaluate_candidate, requirement_checks

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm3_v1.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected JSON object row in {path}")
        output.append(value)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_distribution(values: dict[str, Any], expected: set[str]) -> None:
    if set(values) != expected:
        raise RuntimeError(f"A4.5b-M3 probability labels drifted: {set(values)}")
    numbers = [float(value) for value in values.values()]
    if any(value < 0.0 or value > 1.0 for value in numbers):
        raise RuntimeError("A4.5b-M3 probability outside [0,1]")
    if abs(sum(numbers) - 1.0) > 1e-9:
        raise RuntimeError(f"A4.5b-M3 probability simplex drifted: {sum(numbers)}")


def _probability_audit(
    raw_pairs: list[dict[str, Any]], raw_sets: list[dict[str, Any]]
) -> None:
    dimensions = (
        "entity",
        "predicate",
        "target_slot",
        "temporal_scope",
        "location_scope",
        "organizational_scope",
        "conditional_scope",
        "modality_quantification_scope",
    )
    slots = (
        "entity",
        "predicate",
        "target_slot_identity",
        "target_value",
        "temporal_scope",
        "location_scope",
        "organizational_scope",
        "conditional_scope",
        "modality_quantification_scope",
    )
    for row in [*raw_pairs, *raw_sets]:
        for span in row["spans"]:
            for dimension in dimensions:
                _check_distribution(
                    span["dimensions"][dimension], {"MATCH", "MISMATCH", "UNSPECIFIED"}
                )
            for slot in slots:
                _check_distribution(span["coverage"][slot], {"COVERED", "MISSING"})
            _check_distribution(span["polarity"], {"SUPPORTS", "REFUTES"})
    for row in raw_sets:
        for values in row["subset_polarity"].values():
            _check_distribution(values, {"SUPPORTS", "REFUTES"})


def _reconstruct(
    suite: dict[str, Any],
    raw_pairs: list[dict[str, Any]],
    raw_sets: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    calibration = config["calibration"]
    requirements = calibration["calibration_readiness_requirements"]
    selected: dict[str, Any] | None = None
    feasible = 0
    count = 0
    for mismatch in calibration["mismatch_threshold_grid"]:
        for coverage in calibration["coverage_threshold_grid"]:
            candidate = evaluate_candidate(
                suite, raw_pairs, raw_sets, float(mismatch), float(coverage)
            )
            checks = requirement_checks(candidate["metrics"], requirements)
            candidate["requirement_checks"] = checks
            candidate["requirements_passed"] = sum(checks.values())
            candidate["calibration_ready"] = all(checks.values())
            feasible += int(candidate["calibration_ready"])
            count += 1
            if selected is None or candidate_key(candidate) > candidate_key(selected):
                selected = candidate
    if selected is None or count != 609:
        raise RuntimeError("A4.5b-M3 reconstruction did not evaluate all 609 candidates")
    for key in ("pair_predictions", "set_predictions", "claim_predictions"):
        selected.pop(key, None)
    return {
        "candidate_count": count,
        "feasible_candidate_count": feasible,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = _json(CONFIG)
    results = _json(args.output_dir / "results.json")
    raw_pair_path = args.output_dir / "calibration_pair_raw_scores.jsonl"
    raw_set_path = args.output_dir / "calibration_set_raw_scores.jsonl"
    raw_pairs = _jsonl(raw_pair_path)
    raw_sets = _jsonl(raw_set_path)

    if manifest()["sha256"] != config["calibration"]["sha256"]:
        raise RuntimeError("A4.5b-M3 reconstruction observed M2 calibration hash drift")
    if len(raw_pairs) != 768 or len(raw_sets) != 384:
        raise RuntimeError("A4.5b-M3 raw score cardinality drifted")
    if len({str(row["pair_id"]) for row in raw_pairs}) != 768:
        raise RuntimeError("A4.5b-M3 raw pair IDs are not unique")
    if len({str(row["set_id"]) for row in raw_sets}) != 384:
        raise RuntimeError("A4.5b-M3 raw set IDs are not unique")
    _probability_audit(raw_pairs, raw_sets)

    reconstructed = _reconstruct(build_suite(), raw_pairs, raw_sets, config)
    expected = results["parameter_selection"]
    if reconstructed != expected:
        raise RuntimeError("A4.5b-M3 registered parameter selection failed reconstruction")
    scientific_pass = bool(reconstructed["selected"]["calibration_ready"])
    if bool(results["scientific_pass"]) != scientific_pass:
        raise RuntimeError("A4.5b-M3 scientific pass/fail failed reconstruction")

    audit = {
        "status": "PASSED_A45BM3_DETERMINISTIC_RECONSTRUCTION",
        "scientific_pass": scientific_pass,
        "candidate_count_reconstructed": 609,
        "feasible_candidate_count": reconstructed["feasible_candidate_count"],
        "selected_mismatch_threshold": reconstructed["selected"]["mismatch_threshold"],
        "selected_coverage_threshold": reconstructed["selected"]["coverage_threshold"],
        "raw_pair_rows": len(raw_pairs),
        "raw_set_rows": len(raw_sets),
        "raw_pair_scores_sha256": _sha256(raw_pair_path),
        "raw_set_scores_sha256": _sha256(raw_set_path),
        "probability_simplex_verified": True,
        "a45a_fresh_validation_rows_scored": 0,
        "confirmatory_queries_scored": 0,
        "a45b_closed_rows_scored": 0,
        "post_result_rescue_authorized": False,
        "next_checkpoint_authorized": False,
    }
    (args.output_dir / "post_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "post_audit.md").write_text(
        "\n".join(
            [
                "# A4.5b-M3 deterministic reconstruction",
                "",
                f"Status: **{audit['status']}**",
                f"Scientific pass: **{audit['scientific_pass']}**",
                f"Candidates reconstructed: **{audit['candidate_count_reconstructed']}**",
                f"Feasible candidates: **{audit['feasible_candidate_count']}**",
                "",
                "No fresh validation, confirmatory, or closed A4.5b rows were scored.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

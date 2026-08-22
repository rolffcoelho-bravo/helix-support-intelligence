"""Finalize A4.5b from immutable calibration scores after a deterministic parser defect.

This recovery performs no model loading or inference. It consumes the exact raw
360-row calibration score artifact from A4.5b run 32581433921 and applies the
already registered threshold grid, readiness requirements, and selection rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import aerf_calibration_core_a45b as core  # noqa: E402
from calibration_cases_a45b import build_calibration_only, calibration_manifest  # noqa: E402

CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"
PROVENANCE = ROOT / "benchmarks" / "assistance" / "results" / "a45b_partial_attempt1_v1.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected JSON object row in {path}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_metric_name(requirement_name: str) -> tuple[str, str]:
    """Map a frozen requirement key to its frozen metric without changing semantics."""
    if "_min_" in requirement_name:
        return requirement_name.replace("_min_", "_", 1), "min"
    if requirement_name.endswith("_min"):
        return requirement_name.removesuffix("_min"), "min"
    if "_max_" in requirement_name:
        return requirement_name.replace("_max_", "_", 1), "max"
    if requirement_name.endswith("_max"):
        return requirement_name.removesuffix("_max"), "max"
    raise RuntimeError(f"Unsupported registered A4.5b requirement: {requirement_name}")


def requirement_checks_registered(
    metrics: dict[str, float], requirements: dict[str, Any]
) -> tuple[dict[str, bool], bool]:
    checks: dict[str, bool] = {}
    for requirement_name, raw_threshold in requirements.items():
        metric_name, direction = registered_metric_name(requirement_name)
        if metric_name not in metrics:
            raise RuntimeError(
                "Registered A4.5b requirement maps to absent metric: "
                f"{requirement_name} -> {metric_name}"
            )
        threshold = float(raw_threshold)
        observed = float(metrics[metric_name])
        if direction == "min":
            checks[requirement_name] = observed >= threshold - 1e-12
        else:
            checks[requirement_name] = observed <= threshold + 1e-12
    return checks, all(checks.values())


def _claim_composition_accuracy(claims: list[dict[str, Any]]) -> float:
    correct = 0
    for row in claims:
        gate = str(row["deterministic_gate"])
        relations = [str(value) for value in row["atom_relations"]]
        if gate == "CITATION_INVALID":
            predicted = "CITATION_INVALID"
        elif gate == "STALE_EVIDENCE":
            predicted = "STALE_EVIDENCE"
        elif gate == "REGISTERED_CONFLICT" or (
            "ENTAILED" in relations and "CONTRADICTED" in relations
        ):
            predicted = "CONFLICTING_EVIDENCE"
        elif relations and all(value == "ENTAILED" for value in relations):
            predicted = "SUPPORTED"
        else:
            predicted = "UNSUPPORTED"
        correct += predicted == row["expected_verdict"]
    return correct / len(claims)


def select_thresholds_registered(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    config: dict[str, Any],
    claim_composition_accuracy: float,
) -> dict[str, Any]:
    setup = config["threshold_calibration"]
    relevance_values = core.grid_values(setup["relevance_grid"])
    sufficiency_values = core.grid_values(setup["sufficiency_grid"])
    expected_candidates = int(setup["joint_candidates"])
    if len(relevance_values) * len(sufficiency_values) != expected_candidates:
        raise RuntimeError("A4.5b recovery threshold-grid cardinality drifted")
    requirements = config["calibration_readiness_requirements"]
    best_any: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    feasible_count = 0
    for relevance_threshold in relevance_values:
        for sufficiency_threshold in sufficiency_values:
            metrics = core.calibration_metrics(
                pairs,
                scores,
                relevance_threshold,
                sufficiency_threshold,
                claim_composition_accuracy,
            )
            checks, passed = requirement_checks_registered(metrics, requirements)
            candidate = {
                "relevance_threshold": relevance_threshold,
                "sufficiency_threshold": sufficiency_threshold,
                "metrics": metrics,
                "requirement_checks": checks,
                "calibration_ready": passed,
            }
            if best_any is None or core.selection_key(candidate) > core.selection_key(best_any):
                best_any = candidate
            if passed:
                feasible_count += 1
                if best_feasible is None or core.selection_key(candidate) > core.selection_key(
                    best_feasible
                ):
                    best_feasible = candidate
    if best_any is None:
        raise RuntimeError("A4.5b recovery threshold grid is empty")
    selected = best_feasible if best_feasible is not None else best_any
    return {
        "selected": selected,
        "feasible_candidate_count": feasible_count,
        "joint_candidates_evaluated": expected_candidates,
        "selection_used_feasible_set": best_feasible is not None,
    }


def _validate_partial(input_dir: Path, provenance: dict[str, Any]) -> None:
    artifact = provenance["artifact"]
    expected_files = artifact["files"]
    observed_names = sorted(path.name for path in input_dir.iterdir() if path.is_file())
    if observed_names != sorted(expected_files):
        raise RuntimeError(f"A4.5b partial artifact file set drifted: {observed_names}")
    for name, expected_hash in expected_files.items():
        observed_hash = _sha256(input_dir / name)
        if observed_hash != expected_hash:
            raise RuntimeError(f"A4.5b partial file hash drifted for {name}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(results: dict[str, Any]) -> str:
    selected = results["threshold_selection"]["selected"]
    metrics = selected["metrics"]
    feasible_count = results["threshold_selection"]["feasible_candidate_count"]
    sufficiency_f1 = metrics["sufficiency_macro_f1_on_relevant_pairs"]
    polarity_f1 = metrics["polarity_macro_f1_on_relevant_sufficient_pairs"]
    return "\n".join(
        [
            "# A4.5b deterministic calibration recovery",
            "",
            f"Scientific status: **{results['scientific_status']}**",
            "",
            "No second model inference was performed. The threshold decision uses the immutable",
            "360-row calibration score artifact from run 32581433921.",
            "",
            f"- relevance threshold: `{selected['relevance_threshold']}`",
            f"- sufficiency threshold: `{selected['sufficiency_threshold']}`",
            f"- feasible candidates: `{feasible_count}`",
            f"- final relation macro F1: `{metrics['final_relation_macro_f1']:.6f}`",
            f"- ENTAILED recall: `{metrics['entailed_recall']:.6f}`",
            f"- CONTRADICTED recall: `{metrics['contradicted_recall']:.6f}`",
            f"- UNKNOWN recall: `{metrics['unknown_recall']:.6f}`",
            f"- relevance macro F1: `{metrics['relevance_macro_f1']:.6f}`",
            f"- sufficiency macro F1: `{sufficiency_f1:.6f}`",
            f"- polarity macro F1: `{polarity_f1:.6f}`",
            "",
            "Fresh validation and confirmatory queries remain unopened and unscored.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _json(CONFIG)
    provenance = _json(PROVENANCE)
    _validate_partial(args.input_dir, provenance)
    if provenance["threshold_selected"] is not False:
        raise RuntimeError("A4.5b recovery requires a partial artifact with no threshold selected")
    if provenance["scientific_pass_fail_computed"] is not False:
        raise RuntimeError("A4.5b recovery requires no prior scientific pass/fail result")

    materialized = build_calibration_only()
    manifest = calibration_manifest()
    pairs = materialized["pair_rows"]
    claims = materialized["claim_rows"]
    scores = _jsonl(args.input_dir / "calibration_pair_scores.jsonl")
    pair_ids = [str(row["pair_id"]) for row in pairs]
    score_ids = [str(row["pair_id"]) for row in scores]
    if len(scores) != 360 or score_ids != pair_ids or len(set(score_ids)) != 360:
        raise RuntimeError("A4.5b recovery score rows do not exactly match calibration pairs")
    if any(str(row["split"]) != "calibration" for row in scores):
        raise RuntimeError("A4.5b recovery encountered a non-calibration score row")

    claim_accuracy = _claim_composition_accuracy(claims)
    threshold_selection = select_thresholds_registered(
        pairs,
        scores,
        config,
        claim_accuracy,
    )
    selected = threshold_selection["selected"]
    scientific_pass = bool(selected["calibration_ready"])
    scientific_status = (
        "PASSED_CALIBRATION_READINESS_THRESHOLDS_FROZEN"
        if scientific_pass
        else "FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED"
    )
    results = {
        "binding_id": config["binding_id"],
        "checkpoint": "A4.5b",
        "scientific_pass": scientific_pass,
        "scientific_status": scientific_status,
        "recovery_mode": "DETERMINISTIC_FROM_IMMUTABLE_PARTIAL_ARTIFACT",
        "source_inference_run_id": provenance["workflow_run_id"],
        "source_inference_sha": provenance["scientific_execution_sha"],
        "source_artifact_id": provenance["artifact"]["id"],
        "source_artifact_zip_sha256": provenance["artifact"]["zip_sha256"],
        "second_model_inference_performed": False,
        "calibration_units": manifest["calibration_units"],
        "calibration_pairs": manifest["calibration_pairs"],
        "calibration_claims": manifest["calibration_claims"],
        "calibration_pairs_sha256": manifest["calibration_pairs_sha256"],
        "calibration_claims_sha256": manifest["calibration_claims_sha256"],
        "threshold_selection": threshold_selection,
        "model_weight_verification": _json(args.input_dir / "model_weight_verification.json"),
        "validation_rows_materialized": 0,
        "validation_rows_scored": 0,
        "confirmatory_queries_inspected": 0,
        "confirmatory_queries_scored": 0,
        "a44d_rows_rescored": 0,
        "a44a_rows_rescored": 0,
        "post_result_rescue_authorized": False,
        "next_checkpoint_authorized": False,
    }
    _write_json(args.output_dir / "results.json", results)
    (args.output_dir / "report.md").write_text(_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

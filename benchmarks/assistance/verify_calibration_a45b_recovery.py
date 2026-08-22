"""Independently verify the deterministic A4.5b recovery from immutable raw scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import verify_calibration_a45b as audit  # noqa: E402
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
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object row in {path}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_name(requirement_name: str) -> tuple[str, str]:
    if "_min_" in requirement_name:
        return requirement_name.replace("_min_", "_", 1), "min"
    if requirement_name.endswith("_min"):
        return requirement_name.removesuffix("_min"), "min"
    if "_max_" in requirement_name:
        return requirement_name.replace("_max_", "_", 1), "max"
    if requirement_name.endswith("_max"):
        return requirement_name.removesuffix("_max"), "max"
    raise RuntimeError(f"Unsupported registered recovery requirement {requirement_name}")


def _checks_registered(
    metrics: dict[str, float], requirements: dict[str, Any]
) -> tuple[dict[str, bool], bool]:
    checks: dict[str, bool] = {}
    for requirement_name, raw_threshold in requirements.items():
        metric_name, direction = _metric_name(requirement_name)
        if metric_name not in metrics:
            raise RuntimeError(f"Recovery verifier missing metric {metric_name}")
        observed = float(metrics[metric_name])
        threshold = float(raw_threshold)
        checks[requirement_name] = (
            observed >= threshold - 1e-12
            if direction == "min"
            else observed <= threshold + 1e-12
        )
    return checks, all(checks.values())


def _reconstruct(
    pairs: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    setup = config["threshold_calibration"]
    relevance_values = audit._grid(setup["relevance_grid"])
    sufficiency_values = audit._grid(setup["sufficiency_grid"])
    if len(relevance_values) * len(sufficiency_values) != int(setup["joint_candidates"]):
        raise RuntimeError("Independent recovery grid cardinality drifted")
    requirements = config["calibration_readiness_requirements"]
    claim_accuracy = audit._claim_accuracy(claims)
    best_any: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    feasible_count = 0
    for relevance_threshold in relevance_values:
        for sufficiency_threshold in sufficiency_values:
            metrics = audit._metrics(
                pairs,
                scores,
                relevance_threshold,
                sufficiency_threshold,
                claim_accuracy,
            )
            checks, passed = _checks_registered(metrics, requirements)
            candidate = {
                "relevance_threshold": relevance_threshold,
                "sufficiency_threshold": sufficiency_threshold,
                "metrics": metrics,
                "requirement_checks": checks,
                "calibration_ready": passed,
            }
            if best_any is None or audit._selection_key(candidate) > audit._selection_key(best_any):
                best_any = candidate
            if passed:
                feasible_count += 1
                if best_feasible is None or audit._selection_key(candidate) > audit._selection_key(
                    best_feasible
                ):
                    best_feasible = candidate
    if best_any is None:
        raise RuntimeError("Independent recovery grid is empty")
    selected = best_feasible if best_feasible is not None else best_any
    return {
        "selected": selected,
        "feasible_candidate_count": feasible_count,
        "joint_candidates_evaluated": int(setup["joint_candidates"]),
        "selection_used_feasible_set": best_feasible is not None,
    }


def _assert_close(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"Key mismatch at {path}")
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"List length mismatch at {path}")
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_close(left_value, right_value, f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Numeric mismatch at {path}: {left} != {right}")
        return
    if left != right:
        raise RuntimeError(f"Mismatch at {path}: {left!r} != {right!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = _json(CONFIG)
    provenance = _json(PROVENANCE)
    results = _json(args.output_dir / "results.json")
    scores = _jsonl(args.input_dir / "calibration_pair_scores.jsonl")
    expected_score_hash = provenance["artifact"]["files"]["calibration_pair_scores.jsonl"]
    if _sha256(args.input_dir / "calibration_pair_scores.jsonl") != expected_score_hash:
        raise RuntimeError("Independent recovery raw score hash mismatch")

    materialized = build_calibration_only()
    manifest = calibration_manifest()
    pairs = materialized["pair_rows"]
    claims = materialized["claim_rows"]
    pair_ids = [str(row["pair_id"]) for row in pairs]
    score_ids = [str(row["pair_id"]) for row in scores]
    if len(scores) != 360 or score_ids != pair_ids or len(set(score_ids)) != 360:
        raise RuntimeError("Independent recovery score IDs do not match the frozen calibration pairs")
    if any(str(row["split"]) != "calibration" for row in scores):
        raise RuntimeError("Independent recovery found non-calibration score data")
    for score in scores:
        probabilities = score["nli_probabilities"]
        values = [float(probabilities[name]) for name in ("entailment", "neutral", "contradiction")]
        if any(value < 0.0 or value > 1.0 for value in values):
            raise RuntimeError(f"Invalid probability in {score['pair_id']}")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"Probability simplex failed for {score['pair_id']}")

    reconstructed = _reconstruct(pairs, claims, scores, config)
    _assert_close(reconstructed, results["threshold_selection"], "threshold_selection")
    scientific_pass = bool(reconstructed["selected"]["calibration_ready"])
    expected_status = (
        "PASSED_CALIBRATION_READINESS_THRESHOLDS_FROZEN"
        if scientific_pass
        else "FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED"
    )
    if results["scientific_pass"] is not scientific_pass:
        raise RuntimeError("A4.5b recovery scientific pass bit failed reconstruction")
    if results["scientific_status"] != expected_status:
        raise RuntimeError("A4.5b recovery scientific status failed reconstruction")
    if results["second_model_inference_performed"] is not False:
        raise RuntimeError("A4.5b recovery cannot perform a second model inference")
    if results["source_artifact_id"] != provenance["artifact"]["id"]:
        raise RuntimeError("A4.5b recovery source artifact drifted")
    if results["calibration_pairs_sha256"] != manifest["calibration_pairs_sha256"]:
        raise RuntimeError("A4.5b recovery calibration pair hash drifted")
    if results["calibration_claims_sha256"] != manifest["calibration_claims_sha256"]:
        raise RuntimeError("A4.5b recovery calibration claim hash drifted")
    for field in (
        "validation_rows_materialized",
        "validation_rows_scored",
        "confirmatory_queries_inspected",
        "confirmatory_queries_scored",
        "a44d_rows_rescored",
        "a44a_rows_rescored",
    ):
        if int(results[field]) != 0:
            raise RuntimeError(f"Forbidden A4.5b recovery counter is nonzero: {field}")
    if results["post_result_rescue_authorized"] is not False:
        raise RuntimeError("A4.5b recovery must not authorize post-result rescue")
    if results["next_checkpoint_authorized"] is not False:
        raise RuntimeError("A4.5c remains separately gated")

    post_audit = {
        "status": "PASSED_A45B_DETERMINISTIC_RECOVERY_RECONSTRUCTION",
        "scientific_status": expected_status,
        "scientific_pass": scientific_pass,
        "source_artifact_id": provenance["artifact"]["id"],
        "source_score_sha256": expected_score_hash,
        "calibration_pairs": 360,
        "raw_score_ids_exact_and_unique": True,
        "raw_probability_simplex_verified": True,
        "threshold_candidates_reconstructed": 12050,
        "threshold_selection_reconstructed": True,
        "registered_metrics_reconstructed": True,
        "registered_requirement_checks_reconstructed": True,
        "second_model_inference_performed": False,
        "validation_rows_materialized": 0,
        "validation_rows_scored": 0,
        "confirmatory_queries_inspected": 0,
        "confirmatory_queries_scored": 0,
        "next_checkpoint_authorized": False,
    }
    (args.output_dir / "post_audit.json").write_text(
        json.dumps(post_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "post_audit.md").write_text(
        "\n".join(
            [
                "# A4.5b deterministic recovery audit",
                "",
                f"Audit status: **{post_audit['status']}**",
                f"Scientific status: **{post_audit['scientific_status']}**",
                "",
                "The independent path reconstructed all 12,050 registered threshold candidates",
                "from the immutable 360-row score artifact. No second model inference was used.",
                "Fresh validation and confirmatory counters remain zero.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Deterministically reconstruct the registered A4.5b-M6 calibration result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tpag_calibration_a45bm5 import build_suite, manifest
from tpag_core_a45bm6 import candidate_record, select_candidate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm6_v1.json"
M5_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm5_v1.json"
M5_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm5_manifest_v1.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object row in {path}")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir

    config = _json(CONFIG)
    m5_config = _json(M5_CONFIG)
    frozen_manifest = _json(M5_MANIFEST)
    results = _json(output_dir / "results.json")
    raw_manifest = _json(output_dir / "raw_inference_manifest.json")
    raw_path = output_dir / "residual_raw_scores.jsonl"
    raw_rows = _jsonl(raw_path)

    if manifest() != frozen_manifest:
        raise RuntimeError("A4.5b-M6 post-audit M5 manifest drifted")
    if raw_manifest["status"] != "RAW_LEARNED_OUTPUTS_FROZEN_BEFORE_GOLD_EVALUATION":
        raise RuntimeError("A4.5b-M6 raw-output freeze marker drifted")
    if raw_manifest["raw_scores_sha256"] != _sha256(raw_path):
        raise RuntimeError("A4.5b-M6 raw score checksum mismatch")
    if results["raw_scores_sha256"] != _sha256(raw_path):
        raise RuntimeError("A4.5b-M6 results do not bind the raw score artifact")

    raw_scores: dict[str, dict[str, float]] = {}
    for row in raw_rows:
        probabilities = {str(name): float(value) for name, value in row["probabilities"].items()}
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise RuntimeError("A4.5b-M6 raw residual probabilities do not sum to one")
        raw_scores[str(row["request_id"])] = probabilities

    suite = build_suite()
    requirements = {
        str(name): float(value)
        for name, value in m5_config["calibration_readiness_requirements"].items()
    }
    thresholds = [
        float(value)
        for value in config["calibration_parameter_grid"]["alignment_confidence_values"]
    ]
    candidates = [
        candidate_record(suite, raw_scores, threshold, requirements) for threshold in thresholds
    ]
    selected = select_candidate(candidates)
    if candidates != results["candidates"]:
        raise RuntimeError("A4.5b-M6 candidate arithmetic did not reconstruct exactly")
    if selected != results["selected_candidate"]:
        raise RuntimeError("A4.5b-M6 selected candidate did not reconstruct exactly")
    feasible = sum(bool(candidate["feasible"]) for candidate in candidates)
    if feasible != int(results["feasible_candidate_count"]):
        raise RuntimeError("A4.5b-M6 feasible candidate count did not reconstruct")
    scientific_pass = feasible > 0
    if scientific_pass is not bool(results["scientific_pass"]):
        raise RuntimeError("A4.5b-M6 scientific pass flag did not reconstruct")

    audit = {
        "status": "PASSED_A45BM6_DETERMINISTIC_RECONSTRUCTION",
        "candidate_count": len(candidates),
        "feasible_candidate_count": feasible,
        "residual_request_count": len(raw_rows),
        "raw_scores_sha256": _sha256(raw_path),
        "selected_alignment_confidence_min": selected["alignment_confidence_min"],
        "scientific_pass": scientific_pass,
        "a45a_fresh_validation_rows_scored": 0,
        "confirmatory_queries_scored": 0,
        "closed_a45bm2_m3_rows_scored": 0,
    }
    (output_dir / "post_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "post_audit.md").write_text(
        "# A4.5b-M6 deterministic post-audit\n\n"
        "Status: `PASSED_A45BM6_DETERMINISTIC_RECONSTRUCTION`\n\n"
        f"- candidates reconstructed: **{len(candidates)}**\n"
        f"- feasible candidates: **{feasible}**\n"
        f"- residual learned requests: **{len(raw_rows)}**\n"
        f"- selected alignment threshold: **{selected['alignment_confidence_min']}**\n"
        "- fresh validation rows scored: **0**\n"
        "- confirmatory queries scored: **0**\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

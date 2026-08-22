"""Fail-closed preflight for A4.5b-M3 SCEC calibration-only execution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm3_v1.json"
M2_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm2_v1.json"
M2_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm2_manifest_v1.json"
BUILDER = ROOT / "benchmarks" / "assistance" / "scec_calibration_a45bm2.py"
EXECUTOR = ROOT / "benchmarks" / "assistance" / "calibrate_scec_a45bm3.py"
SOURCE_MAIN_SHA = "e44fc9d53d217b6bf506a2253d06da876f904c26"
MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
MODEL_REVISION = "91562024e753ad76646a2d0dfcbb26801aa945fe"
MODEL_SHA256 = "6e8f2af78c828dcbd5243aac40fb87430376f0b8a9c288f4993df3ea3558d557"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("scec_calibration_a45bm2", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen A4.5b-M2 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = _json(CONFIG)
    m2 = _json(M2_CONFIG)
    frozen_manifest = _json(M2_MANIFEST)
    generated_manifest = _builder_module().manifest()

    if config["checkpoint"] != "A4.5b-M3":
        raise RuntimeError("A4.5b-M3 checkpoint identity drifted")
    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b-M3 source main SHA drifted")
    if config["source_protocol_id"] != m2["protocol_id"]:
        raise RuntimeError("A4.5b-M3 no longer binds the frozen M2 protocol")
    if generated_manifest != frozen_manifest:
        raise RuntimeError("A4.5b-M2 fresh calibration manifest is no longer reproducible")

    calibration = config["calibration"]
    if calibration["sha256"] != frozen_manifest["sha256"]:
        raise RuntimeError("A4.5b-M3 calibration hashes differ from frozen M2 hashes")
    if (
        int(calibration["units"]) != 48
        or int(calibration["pair_rows"]) != 768
        or int(calibration["evidence_set_rows"]) != 384
        or int(calibration["claim_rows"]) != 384
    ):
        raise RuntimeError("A4.5b-M3 calibration cardinality drifted")

    implementation = config["authoritative_implementation"]
    if int(implementation["count"]) != 1:
        raise RuntimeError("A4.5b-M3 must bind exactly one authoritative implementation")
    model = implementation["semantic_model"]
    if (
        model["model_id"] != MODEL_ID
        or model["revision"] != MODEL_REVISION
        or model["weights_sha256"] != MODEL_SHA256
    ):
        raise RuntimeError("A4.5b-M3 authoritative model pin drifted")
    if model["native_labels"] != {"0": "entailment", "1": "not_entailment"}:
        raise RuntimeError("A4.5b-M3 native label contract drifted")

    mismatch_grid = calibration["mismatch_threshold_grid"]
    coverage_grid = calibration["coverage_threshold_grid"]
    expected_mismatch = [round(0.34 + 0.02 * index, 2) for index in range(29)]
    expected_coverage = [round(0.50 + 0.02 * index, 2) for index in range(21)]
    if mismatch_grid != expected_mismatch or coverage_grid != expected_coverage:
        raise RuntimeError("A4.5b-M3 registered parameter grids drifted")
    if int(calibration["joint_candidate_count"]) != 609:
        raise RuntimeError("A4.5b-M3 joint parameter count must remain 609")
    if calibration["calibration_readiness_requirements"] != m2[
        "calibration_readiness_requirements"
    ]:
        raise RuntimeError("A4.5b-M3 weakened or altered M2 readiness requirements")

    dimensions = implementation["compatibility"]["dimensions"]
    hypotheses = implementation["compatibility"]["hypotheses"]
    if len(dimensions) != 8 or set(dimensions) != set(hypotheses):
        raise RuntimeError("A4.5b-M3 compatibility dimension registry drifted")
    for dimension in dimensions:
        if set(hypotheses[dimension]) != {"MATCH", "MISMATCH", "UNSPECIFIED"}:
            raise RuntimeError(f"A4.5b-M3 hypotheses drifted for {dimension}")
    if len(implementation["coverage"]["slots"]) != 9:
        raise RuntimeError("A4.5b-M3 decisive coverage slot registry drifted")

    forbidden = config["forbidden_after_binding"]
    if not all(bool(value) for value in forbidden.values()):
        raise RuntimeError("A4.5b-M3 post-binding prohibitions must remain fail-closed")

    scope = config["execution_scope"]
    if int(scope["semantic_inference_authorized"]) != 1:
        raise RuntimeError("A4.5b-M3 calibration inference authorization drifted")
    if int(scope["calibration_fit_authorized"]) != 1:
        raise RuntimeError("A4.5b-M3 calibration fitting authorization drifted")
    if int(scope["model_binding_count"]) != 1:
        raise RuntimeError("A4.5b-M3 model binding count drifted")
    if int(scope["validation_scoring_authorized"]) != 0:
        raise RuntimeError("A4.5b-M3 fresh validation must remain unauthorized")
    if int(scope["confirmatory_scoring_authorized"]) != 0:
        raise RuntimeError("A4.5b-M3 confirmatory scoring must remain unauthorized")

    if any(int(value) != 0 for value in config["sealed_partitions"].values()):
        raise RuntimeError("A4.5b-M3 sealed partition counter is nonzero")

    executor_text = EXECUTOR.read_text(encoding="utf-8")
    forbidden_sources = (
        "aerf_validity_a45a",
        "calibration_cases_a45b",
        "calibration_pair_scores.jsonl",
        "a45b_calibration_postresult",
    )
    for token in forbidden_sources:
        if token in executor_text:
            raise RuntimeError(f"A4.5b-M3 executor references forbidden prior data: {token}")

    print(
        json.dumps(
            {
                "status": "PASSED_A45BM3_PRE_EXECUTION_CALIBRATION_ONLY",
                "authoritative_implementation_count": 1,
                "model_id": MODEL_ID,
                "calibration_units": 48,
                "pair_rows": 768,
                "evidence_set_rows": 384,
                "claim_rows": 384,
                "joint_parameter_candidates": 609,
                "fresh_validation_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "a45b_closed_rows_authorized": 0,
                "next_checkpoint_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

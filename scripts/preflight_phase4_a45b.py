"""Fail-closed pre-execution audit for Phase 4 A4.5b."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"
A45A_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45a_manifest_v1.json"
CALIBRATION_CASES = ROOT / "benchmarks" / "assistance" / "calibration_cases_a45b.py"
SOURCE_MAIN_SHA = "35305f3568e0842bc7327dfebb58ffc975ea1cee"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _load_calibration_module() -> Any:
    spec = importlib.util.spec_from_file_location("calibration_cases_a45b", CALIBRATION_CASES)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load A4.5b calibration materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grid_count(grid: dict[str, Any]) -> int:
    start = int(grid["integer_start"])
    stop = int(grid["integer_stop"])
    step = int(grid["integer_step"])
    return len(range(start, stop + step, step))


def main() -> None:
    config = _load_json(CONFIG)
    a45a_manifest = _load_json(A45A_MANIFEST)
    materializer = _load_calibration_module()
    calibration = materializer.calibration_manifest()

    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b must descend from the exact A4.5a closure")
    if config["status"] != "REGISTERED_PRE_EXECUTION_CALIBRATION_ONLY":
        raise RuntimeError("A4.5b must remain registered pre-execution")
    if config["architecture"]["short_name"] != "AERF":
        raise RuntimeError("A4.5b architecture must remain AERF")
    if "never directly mapped to UNKNOWN" not in config["architecture"]["unknown_construction"]:
        raise RuntimeError("A4.5b must not map native NLI neutral directly to UNKNOWN")

    binding = config["binding"]
    relevance = binding["alignment_relevance"]
    if relevance["model_id"] != "cross-encoder/ms-marco-MiniLM-L6-v2":
        raise RuntimeError("A4.5b relevance model drifted")
    if relevance["revision"] != "c5f2b386de279a97c53a702dd5189d1c407160dc":
        raise RuntimeError("A4.5b relevance revision drifted")
    if relevance["weights_sha256"] != (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise RuntimeError("A4.5b relevance weight hash drifted")

    nli = binding["sufficiency_polarity"]
    if nli["model_id"] != "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli":
        raise RuntimeError("A4.5b sufficiency/polarity model drifted")
    if nli["revision"] != "0e2603d5d3d3ef9b2910814b34eebe1a2101da65":
        raise RuntimeError("A4.5b NLI revision drifted")
    if nli["weights_sha256"] != (
        "06d6fd89edd4f97816831626daafbdb0b029cf63bae8edc0bccab1d64e2e7707"
    ):
        raise RuntimeError("A4.5b NLI weight hash drifted")
    if nli["native_labels"] != {
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
    }:
        raise RuntimeError("A4.5b NLI native labels drifted")

    partition = config["calibration_partition"]
    if calibration["calibration_units"] != partition["unit_count"]:
        raise RuntimeError("A4.5b calibration unit count drifted")
    if calibration["calibration_pairs"] != partition["pair_rows"]:
        raise RuntimeError("A4.5b calibration pair count drifted")
    if calibration["calibration_claims"] != partition["claim_rows"]:
        raise RuntimeError("A4.5b calibration claim count drifted")
    if calibration["calibration_pairs_sha256"] != partition["pairs_sha256"]:
        raise RuntimeError("A4.5b calibration pair hash drifted")
    if calibration["calibration_claims_sha256"] != partition["claims_sha256"]:
        raise RuntimeError("A4.5b calibration claim hash drifted")
    if partition["pairs_sha256"] != a45a_manifest["sha256"]["calibration_pairs"]:
        raise RuntimeError("A4.5b pair hash no longer matches A4.5a")
    if partition["claims_sha256"] != a45a_manifest["sha256"]["calibration_claims"]:
        raise RuntimeError("A4.5b claim hash no longer matches A4.5a")

    setup = config["threshold_calibration"]
    candidates = _grid_count(setup["relevance_grid"]) * _grid_count(setup["sufficiency_grid"])
    if candidates != 12050 or candidates != setup["joint_candidates"]:
        raise RuntimeError("A4.5b registered threshold grid drifted")

    sealed = config["sealed_partitions"]
    if sealed["validation_scoring_authorized"] != 0:
        raise RuntimeError("A4.5b cannot authorize fresh validation")
    if sealed["validation_rows_materialized_by_a45b"] != 0:
        raise RuntimeError("A4.5b cannot materialize fresh validation rows")
    if sealed["confirmatory_query_records_inspected"] != 0:
        raise RuntimeError("A4.5b confirmatory records must remain unopened")
    if sealed["confirmatory_scoring_authorized"] != 0:
        raise RuntimeError("A4.5b confirmatory scoring must remain unauthorized")
    scope = config["scope"]
    if scope["candidate_model_comparison_authorized"] != 0:
        raise RuntimeError("A4.5b binds one implementation; comparison is forbidden")
    if scope["validation_model_inference_authorized"] != 0:
        raise RuntimeError("A4.5b validation model inference must remain forbidden")
    if config["next_checkpoint"]["authorized_by_a45b"] is not False:
        raise RuntimeError("A4.5c requires separate approval")

    print(
        json.dumps(
            {
                "status": "PASSED_A45B_PRE_EXECUTION_CALIBRATION_ONLY_NO_RESULTS",
                "calibration_units": calibration["calibration_units"],
                "calibration_pairs": calibration["calibration_pairs"],
                "joint_threshold_candidates": candidates,
                "validation_rows_materialized": 0,
                "validation_scoring_authorized": 0,
                "confirmatory_queries_inspected": 0,
                "confirmatory_scoring_authorized": 0,
                "next_checkpoint_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

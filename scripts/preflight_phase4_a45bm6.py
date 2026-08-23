"""Fail-closed preflight for A4.5b-M6 TPAG implementation binding and calibration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm6_v1.json"
M5_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm5_v1.json"
M5_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm5_manifest_v1.json"
BUILDER = ROOT / "benchmarks" / "assistance" / "tpag_calibration_a45bm5.py"
SOURCE_MAIN_SHA = "9070fc4cf0447077a20c7e576e49e9ba5f0536ba"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("tpag_calibration_a45bm5", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load A4.5b-M5 TPAG calibration builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = _json(CONFIG)
    m5 = _json(M5_CONFIG)
    frozen_manifest = _json(M5_MANIFEST)
    module = _builder_module()

    if config["checkpoint"] != "A4.5b-M6":
        raise RuntimeError("A4.5b-M6 checkpoint identity drifted")
    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b-M6 source main SHA drifted")
    if m5["checkpoint"] != "A4.5b-M5":
        raise RuntimeError("A4.5b-M6 requires the frozen M5 protocol")
    if module.manifest() != frozen_manifest:
        raise RuntimeError("A4.5b-M6 requires the exact frozen M5 calibration manifest")
    suite = module.build_suite()
    if len(suite["units"]) != 64:
        raise RuntimeError("A4.5b-M6 calibration unit count drifted")
    if len(suite["proposition_rows"]) != 512:
        raise RuntimeError("A4.5b-M6 proposition row count drifted")
    if len(suite["alignment_rows"]) != 1280:
        raise RuntimeError("A4.5b-M6 alignment row count drifted")
    if len(suite["evidence_group_rows"]) != 768:
        raise RuntimeError("A4.5b-M6 evidence-group row count drifted")
    if len(suite["claim_rows"]) != 640:
        raise RuntimeError("A4.5b-M6 claim row count drifted")

    implementation = config["authoritative_implementation"]
    if implementation["count"] != 1:
        raise RuntimeError("A4.5b-M6 must bind exactly one authoritative TPAG pipeline")
    if implementation["short_name"] != "TPAG":
        raise RuntimeError("A4.5b-M6 authoritative architecture drifted")
    if implementation["novelty_claim"] is not False:
        raise RuntimeError("A4.5b-M6 may not claim TPAG novelty")
    extraction = implementation["proposition_and_frame_extraction"]
    if extraction["type"] != "deterministic":
        raise RuntimeError("A4.5b-M6 extraction binding drifted")
    if extraction["tunable_parameters"] != 0:
        raise RuntimeError("A4.5b-M6 deterministic extraction cannot expose tunable parameters")
    if implementation["polarity"]["type"] != "deterministic_after_scope_and_coverage":
        raise RuntimeError("A4.5b-M6 polarity binding drifted")

    model = implementation["semantic_model"]
    if model["model_id"] != "cross-encoder/nli-deberta-v3-base":
        raise RuntimeError("A4.5b-M6 semantic model identity drifted")
    if model["revision"] != "6c749ce3425cd33b46d187e45b92bbf96ee12ec7":
        raise RuntimeError("A4.5b-M6 semantic model revision drifted")
    if (
        model["weights_sha256"]
        != "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa"
    ):
        raise RuntimeError("A4.5b-M6 semantic model weight hash drifted")
    if (
        model["tokenizer_sha256"]
        != "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"
    ):
        raise RuntimeError("A4.5b-M6 tokenizer hash drifted")
    if model["native_labels"] != {
        "0": "contradiction",
        "1": "entailment",
        "2": "neutral",
    }:
        raise RuntimeError("A4.5b-M6 native label mapping drifted")
    if model["role"] != "residual query-conditioned typed-edge semantic equivalence only":
        raise RuntimeError("A4.5b-M6 learned role expanded beyond the registered residual")

    grid = config["calibration_parameter_grid"]
    if grid["active_thresholds"] != ["alignment_confidence_min"]:
        raise RuntimeError("A4.5b-M6 active threshold set drifted")
    if grid["candidate_count"] != 7:
        raise RuntimeError("A4.5b-M6 must evaluate exactly seven registered candidates")
    if grid["candidate_count"] > int(grid["m5_maximum_joint_candidates"]):
        raise RuntimeError("A4.5b-M6 parameter grid exceeds the M5 budget")
    if grid["class_specific_thresholds_authorized"] is not False:
        raise RuntimeError("A4.5b-M6 class-specific thresholds are prohibited")
    if grid["slot_specific_thresholds_authorized"] is not False:
        raise RuntimeError("A4.5b-M6 slot-specific thresholds are prohibited")
    if grid["post_result_grid_expansion_authorized"] is not False:
        raise RuntimeError("A4.5b-M6 post-result grid expansion is prohibited")
    if grid["all_raw_learned_outputs_frozen_before_threshold_selection"] is not True:
        raise RuntimeError("A4.5b-M6 must freeze raw learned outputs before parameter selection")

    execution = config["execution_contract"]
    expected_nonzero = {
        "m5_calibration_units_authorized": 64,
        "m5_proposition_rows_authorized": 512,
        "m5_alignment_rows_authorized": 1280,
        "m5_evidence_group_rows_authorized": 768,
        "m5_claim_rows_authorized": 640,
        "semantic_model_bindings_authorized": 1,
        "authoritative_pipeline_count": 1,
        "threshold_candidates_authorized": 7,
    }
    for name, value in expected_nonzero.items():
        if int(execution[name]) != value:
            raise RuntimeError(f"A4.5b-M6 execution contract drifted: {name}")
    for name in (
        "model_family_comparisons_authorized",
        "prompt_searches_authorized",
        "closed_a45bm2_m3_rows_authorized",
        "a45a_fresh_validation_rows_authorized",
        "confirmatory_queries_authorized",
        "future_validation_construction_authorized",
        "post_result_rescue_authorized",
    ):
        if int(execution[name]) != 0:
            raise RuntimeError(f"A4.5b-M6 forbidden execution scope is nonzero: {name}")

    if len(m5["calibration_readiness_requirements"]) != 56:
        raise RuntimeError("A4.5b-M6 requires all 56 frozen M5 readiness requirements")
    if config["gold_separation"]["inference_stage_may_read_gold"] is not False:
        raise RuntimeError("A4.5b-M6 inference stage may not read gold")
    if (
        config["gold_separation"]["raw_learned_outputs_must_be_written_before_gold_evaluation"]
        is not True
    ):
        raise RuntimeError("A4.5b-M6 raw-output/gold boundary drifted")
    governance = config["governance"]
    if governance["a45a_fresh_validation_remains_sealed"] is not True:
        raise RuntimeError("A4.5a validation must remain sealed")
    if governance["confirmatory_partition_remains_sealed"] is not True:
        raise RuntimeError("Confirmatory partition must remain sealed")
    if governance["a45c_repurposed"] is not False:
        raise RuntimeError("A4.5c must not be repurposed")
    if config["next_checkpoint"]["authorized"] is not False:
        raise RuntimeError("No checkpoint after M6 may be pre-authorized")

    print(
        json.dumps(
            {
                "status": "PASSED_A45BM6_PRE_EXECUTION_CALIBRATION_ONLY",
                "authoritative_pipeline_count": 1,
                "model_id": model["model_id"],
                "calibration_units": 64,
                "proposition_rows": 512,
                "alignment_rows": 1280,
                "evidence_group_rows": 768,
                "claim_rows": 640,
                "readiness_requirements": 56,
                "joint_parameter_candidates": 7,
                "model_family_comparisons_authorized": 0,
                "prompt_searches_authorized": 0,
                "fresh_validation_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "next_checkpoint_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Fail-closed zero-result preflight for Phase 4 A4.5d."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45d_v1.json"
A45A_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json"
A45A_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45a_manifest_v1.json"
M6_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm6_v1.json"
M6_CLOSURE = ROOT / "benchmarks" / "assistance" / "a45bm6_closure_v1.json"
M6_CORE = ROOT / "benchmarks" / "assistance" / "tpag_core_a45bm6.py"
A45A_GENERATOR = ROOT / "benchmarks" / "assistance" / "aerf_validity_a45a.py"
SOURCE_MAIN_SHA = "40d6bdb417e798a7c0ead7709bdcec5d8241a989"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("tpag_core_a45bm6_for_a45d", M6_CORE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen M6 TPAG core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = _json(CONFIG)
    a45a = _json(A45A_CONFIG)
    manifest = _json(A45A_MANIFEST)
    m6 = _json(M6_CONFIG)
    closure = _json(M6_CLOSURE)

    if config["checkpoint"] != "A4.5d":
        raise RuntimeError("A4.5d checkpoint identity drifted")
    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5d must bind the exact M6 closure main SHA")
    if config["status"] != "REGISTERED_ZERO_RESULT_VALIDATION_BLOCKED_INPUT_CONTRACT_MISMATCH":
        raise RuntimeError("A4.5d zero-result blocker status drifted")
    if config["scientific_result_exposed"] is not False:
        raise RuntimeError("A4.5d may not expose a scientific validation result")

    if closure["closure_id"] != config["source_m6_closure_id"]:
        raise RuntimeError("A4.5d M6 closure identity drifted")
    if closure["scientific_status"] != "PASSED_TPAG_CALIBRATION_READINESS_PARAMETERS_FROZEN":
        raise RuntimeError("A4.5d requires the frozen positive M6 calibration closure")
    if closure["scientific_pass"] is not True:
        raise RuntimeError("A4.5d requires M6 scientific_pass=true")
    if float(closure["registered_calibration"]["selected_alignment_confidence_min"]) != 0.6:
        raise RuntimeError("A4.5d requires the frozen M6 alignment threshold 0.60")
    if closure["validation_authorized"] is not False:
        raise RuntimeError("M6 closure may not already authorize validation")

    bound = config["authoritative_implementation"]
    m6_impl = m6["authoritative_implementation"]
    if bound["implementation_id"] != m6_impl["implementation_id"]:
        raise RuntimeError("A4.5d TPAG implementation identity drifted")
    if float(bound["selected_alignment_confidence_min"]) != 0.6:
        raise RuntimeError("A4.5d may not change the frozen M6 threshold")
    if bound["model"] != m6_impl["semantic_model"]:
        raise RuntimeError("A4.5d model binding must match M6 exactly")
    if bound["runtime"] != m6["runtime"]:
        raise RuntimeError("A4.5d runtime binding must match M6 exactly")

    if a45a["protocol_id"] != config["source_a45a_protocol_id"]:
        raise RuntimeError("A4.5d A4.5a protocol identity drifted")
    fresh = config["fresh_validation_contract"]
    a45a_fresh = a45a["fresh_validity_construction"]
    if int(fresh["validation_units"]) != 20 or int(manifest["counts"]["validation_units"]) != 20:
        raise RuntimeError("A4.5d validation-unit count drifted")
    if int(fresh["validation_pair_rows"]) != 180:
        raise RuntimeError("A4.5d validation-pair count drifted")
    if int(fresh["validation_claim_rows"]) != 180:
        raise RuntimeError("A4.5d validation-claim count drifted")
    if fresh["validation_pairs_sha256"] != a45a_fresh["validation_pairs_sha256"]:
        raise RuntimeError("A4.5d validation-pair hash drifted")
    if fresh["validation_pairs_sha256"] != manifest["sha256"]["validation_pairs"]:
        raise RuntimeError("A4.5d manifest validation-pair hash drifted")
    if fresh["validation_claims_sha256"] != a45a_fresh["validation_claims_sha256"]:
        raise RuntimeError("A4.5d validation-claim hash drifted")
    if fresh["validation_claims_sha256"] != manifest["sha256"]["validation_claims"]:
        raise RuntimeError("A4.5d manifest validation-claim hash drifted")
    if fresh["component_requirements"] != a45a["registered_component_requirements"]:
        raise RuntimeError("A4.5d component validity requirements drifted")
    if fresh["claim_requirements"] != a45a["registered_claim_requirements"]:
        raise RuntimeError("A4.5d claim validity requirements drifted")
    if int(fresh["requirement_count"]) != 29:
        raise RuntimeError("A4.5d must preserve all 29 A4.5a hard validity requirements")

    audit = config["pre_execution_contract_audit"]
    if audit["direct_application_ready"] is not False:
        raise RuntimeError("A4.5d must fail closed on direct validation application")
    if audit["existing_registered_adapter_found"] is not False:
        raise RuntimeError("A4.5d found no previously registered A4.5a-to-TPAG adapter")
    if audit["adapter_or_parser_extension_would_change_scientific_implementation"] is not True:
        raise RuntimeError("A4.5d must treat a new adapter/parser extension as a scientific change")
    if audit["validation_execution_authorized"] is not False:
        raise RuntimeError("A4.5d may not authorize validation execution")

    # Contract audit only: inspect source grammar and use non-validation synthetic probes.
    # The A4.5a validation builder is deliberately never imported or executed here.
    a45a_source = A45A_GENERATOR.read_text(encoding="utf-8")
    if 'subject = f"Orchid case {number:03d}"' not in a45a_source:
        raise RuntimeError("A4.5a registered subject grammar changed")
    if "requests are handled by the {queue} queue" not in a45a_source:
        raise RuntimeError("A4.5a registered support grammar changed")

    core = _module()
    queue_probe = core.parse_frame(
        "Orchid case 999 requests are handled by the access_review queue.", {}
    )
    requirement_probe = core.parse_frame(
        "Orchid case 999 review requires transaction reference.", {}
    )
    for probe in (queue_probe, requirement_probe):
        if probe["entity_or_subject"] is not None:
            raise RuntimeError("Frozen M6 parser unexpectedly accepts A4.5a subject grammar")
        if probe["predicate_or_event"] is not None:
            raise RuntimeError("Frozen M6 parser unexpectedly accepts A4.5a predicate grammar")
        if probe["target_slot_identity"] is not None:
            raise RuntimeError("Frozen M6 parser unexpectedly accepts A4.5a target-slot grammar")

    scope = config["execution_scope"]
    if any(int(value) != 0 for value in scope.values()):
        raise RuntimeError("A4.5d is zero-result and must authorize no execution")
    if any(
        int(fresh[name]) != 0
        for name in (
            "validation_records_materialized",
            "validation_records_inspected",
            "validation_records_scored",
        )
    ):
        raise RuntimeError("A4.5d fresh validation must remain unopened")

    governance = config["governance"]
    if governance["a45a_validation_remains_sealed"] is not True:
        raise RuntimeError("A4.5a fresh validation must remain sealed")
    if governance["confirmatory_partition_remains_sealed"] is not True:
        raise RuntimeError("Confirmatory partition must remain sealed")
    if governance["a45c_repurposed"] is not False:
        raise RuntimeError("A4.5c must remain permanently ineligible")
    if config["next_checkpoint"]["authorized"] is not False:
        raise RuntimeError("A4.5d may not pre-authorize its successor")

    print(
        json.dumps(
            {
                "status": "PASSED_A45D_ZERO_RESULT_PREEXECUTION_BLOCKER",
                "m6_alignment_confidence_min": 0.6,
                "a45a_validation_units": 20,
                "a45a_validation_pairs": 180,
                "a45a_validation_claims": 180,
                "registered_validity_requirements": 29,
                "validation_records_materialized": 0,
                "validation_records_inspected": 0,
                "validation_records_scored": 0,
                "confirmatory_records_inspected": 0,
                "validation_execution_authorized": False,
                "next_checkpoint_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

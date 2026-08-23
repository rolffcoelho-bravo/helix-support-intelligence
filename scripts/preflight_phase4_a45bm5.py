"""Fail-closed preflight for A4.5b-M5 TPAG calibration protocol registration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm5_v1.json"
MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm5_manifest_v1.json"
BUILDER = ROOT / "benchmarks" / "assistance" / "tpag_calibration_a45bm5.py"
M4 = ROOT / "configs" / "models" / "assistance_grounding_a45bm4_v1.json"
M2_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm2_manifest_v1.json"
SOURCE_MAIN_SHA = "2299935cdf940bc0f3774e7d5c35d4a5cd297d87"

SCOPE_MISMATCH_SUBTYPES = {
    "entity_mismatch",
    "predicate_mismatch",
    "target_slot_identity_mismatch",
    "temporal_scope_mismatch",
    "location_scope_mismatch",
    "organizational_scope_mismatch",
    "conditional_scope_mismatch",
    "modality_quantification_mismatch",
}
UNSPECIFIED_SUBTYPES = {
    "missing_target_value",
    "missing_temporal_scope",
    "missing_conditional_scope",
    "missing_location_scope",
    "missing_organizational_scope",
    "missing_modality_quantification",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("tpag_calibration_a45bm5", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load A4.5b-M5 calibration builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _by_subtype(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["subtype"]), []).append(row)
    return result


def main() -> None:
    config = _json(CONFIG)
    frozen_manifest = _json(MANIFEST)
    m4 = _json(M4)
    m2_manifest = _json(M2_MANIFEST)

    if config["checkpoint"] != "A4.5b-M5":
        raise RuntimeError("A4.5b-M5 checkpoint identity drifted")
    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b-M5 source main SHA drifted")
    if config["source_methodology_decision_id"] != m4["decision_id"]:
        raise RuntimeError("A4.5b-M5 source methodology decision drifted")
    if m4["selected_methodology"]["short_name"] != "TPAG":
        raise RuntimeError("A4.5b-M5 requires the frozen TPAG methodology")
    if m4["selected_methodology"]["binding_status"] != "UNBOUND":
        raise RuntimeError("TPAG must remain unbound before A4.5b-M5")

    module = _builder_module()
    current_manifest = module.manifest()
    if current_manifest != frozen_manifest:
        raise RuntimeError("A4.5b-M5 fresh calibration manifest drifted")

    suite = module.build_suite()
    expected_counts = {
        "units": 64,
        "proposition_rows": 512,
        "alignment_rows": 1280,
        "evidence_group_rows": 768,
        "claim_rows": 640,
    }
    for key, expected in expected_counts.items():
        if len(suite[key]) != expected:
            raise RuntimeError(f"A4.5b-M5 {key} count drifted")

    for key in (
        "proposition_rows",
        "alignment_rows",
        "evidence_group_rows",
        "claim_rows",
    ):
        if any(row["split"] != "calibration" for row in suite[key]):
            raise RuntimeError(f"A4.5b-M5 {key} must be calibration-only")

    serialized_suite = json.dumps(suite, sort_keys=True)
    if "Cobalt case" in serialized_suite or "SCEC-C" in serialized_suite:
        raise RuntimeError("A4.5b-M5 must not reuse M2/M3 calibration units")
    if any(not str(row["unit_id"]).startswith("TPAG-C") for row in suite["units"]):
        raise RuntimeError("A4.5b-M5 unit namespace drifted")
    if frozen_manifest["corpus_id"] == m2_manifest["corpus_id"]:
        raise RuntimeError("A4.5b-M5 corpus must differ from M2/M3 corpus")
    if frozen_manifest["seed"] == m2_manifest["seed"]:
        raise RuntimeError("A4.5b-M5 seed must differ from M2/M3 seed")

    alignments = _by_subtype(suite["alignment_rows"])
    if set(alignments) != set(frozen_manifest["counts"]["alignment_subtypes"]):
        raise RuntimeError("A4.5b-M5 alignment subtype registry drifted")

    value_refutes = alignments["direct_refutation_value_conflict"]
    if len(value_refutes) != 64:
        raise RuntimeError("A4.5b-M5 target-value refutation count drifted")
    for row in value_refutes:
        gold = row["gold"]
        if gold["slot_relations"]["target_value"] != "MISMATCH":
            raise RuntimeError("Target-value refutation must register target_value MISMATCH")
        if gold["scope_compatibility"] != "COMPATIBLE":
            raise RuntimeError("Target-value refutation must remain scope-compatible")
        if gold["polarity"] != "REFUTES" or gold["final_relation"] != "CONTRADICTED":
            raise RuntimeError("Target-value refutation semantics drifted")

    for subtype in SCOPE_MISMATCH_SUBTYPES:
        rows = alignments[subtype]
        if len(rows) != 64:
            raise RuntimeError(f"A4.5b-M5 {subtype} count drifted")
        if any(row["gold"]["scope_compatibility"] != "INCOMPATIBLE" for row in rows):
            raise RuntimeError(f"A4.5b-M5 {subtype} must be incompatible")

    for subtype in UNSPECIFIED_SUBTYPES:
        rows = alignments[subtype]
        if len(rows) != 64:
            raise RuntimeError(f"A4.5b-M5 {subtype} count drifted")
        for row in rows:
            gold = row["gold"]
            if gold["scope_compatibility"] != "COMPATIBLE":
                raise RuntimeError(f"A4.5b-M5 {subtype} must remain compatible")
            if gold["coverage_status"] != "INCOMPLETE":
                raise RuntimeError(f"A4.5b-M5 {subtype} must remain incomplete")
            if "UNSPECIFIED" not in gold["slot_relations"].values():
                raise RuntimeError(f"A4.5b-M5 {subtype} must expose UNSPECIFIED")

    groups = _by_subtype(suite["evidence_group_rows"])
    same_scope = groups["same_scope_support_refute_conflict"]
    different_condition = groups["different_condition_not_conflict"]
    if any(row["gold"]["final_relation"] != "CONFLICTING_EVIDENCE" for row in same_scope):
        raise RuntimeError("Same-scope support/refute groups must conflict")
    if any(row["gold"]["final_relation"] == "CONFLICTING_EVIDENCE" for row in different_condition):
        raise RuntimeError("Different-condition evidence must not create false conflict")

    method = config["methodology"]
    if method["binding_status"] != "UNBOUND":
        raise RuntimeError("A4.5b-M5 cannot bind a TPAG implementation")
    if method["novelty_claim"] is not False:
        raise RuntimeError("A4.5b-M5 must not claim TPAG novelty")

    requirements = config["calibration_readiness_requirements"]
    if len(requirements) != 56:
        raise RuntimeError("A4.5b-M5 readiness requirement count drifted")

    budget = config["future_calibration_parameter_budget"]
    if budget["maximum_scalar_thresholds"] != 3:
        raise RuntimeError("A4.5b-M5 threshold-count budget drifted")
    if len(budget["values"]) != 7:
        raise RuntimeError("A4.5b-M5 threshold-value grid drifted")
    if budget["maximum_joint_candidates"] != 343:
        raise RuntimeError("A4.5b-M5 joint parameter budget drifted")
    if budget["maximum_joint_candidates"] != len(budget["values"]) ** 3:
        raise RuntimeError("A4.5b-M5 parameter arithmetic is inconsistent")
    if budget["class_specific_thresholds_authorized"] is not False:
        raise RuntimeError("A4.5b-M5 forbids class-specific thresholds")
    if budget["slot_specific_thresholds_authorized"] is not False:
        raise RuntimeError("A4.5b-M5 forbids slot-specific thresholds")
    if budget["post_result_grid_expansion_authorized"] is not False:
        raise RuntimeError("A4.5b-M5 forbids post-result grid expansion")

    governance = config["data_governance"]
    if governance["a45a_fresh_validation_remains_sealed"] is not True:
        raise RuntimeError("A4.5a fresh validation must remain sealed")
    if governance["a45a_fresh_validation_pairs_scored"] != 0:
        raise RuntimeError("A4.5a fresh validation pairs must remain unscored")
    if governance["a45a_fresh_validation_claims_scored"] != 0:
        raise RuntimeError("A4.5a fresh validation claims must remain unscored")
    if governance["confirmatory_records_inspected"] != 0:
        raise RuntimeError("Confirmatory records must remain unopened")
    if governance["confirmatory_queries_scored"] != 0:
        raise RuntimeError("Confirmatory queries must remain unscored")
    if governance["a45c_repurposed"] is not False:
        raise RuntimeError("A4.5c must remain ineligible and unrepurposed")

    if any(int(value) != 0 for value in frozen_manifest["governance"].values()):
        raise RuntimeError("A4.5b-M5 manifest governance must remain zero")
    for name, value in config["scope"].items():
        if int(value) != 0:
            raise RuntimeError(f"A4.5b-M5 forbidden execution scope is nonzero: {name}")

    if config["next_action"]["checkpoint"] != "A4.5b-M6":
        raise RuntimeError("A4.5b-M5 next checkpoint identity drifted")
    if config["next_action"]["authorized"] is not False:
        raise RuntimeError("A4.5b-M6 requires separate approval")

    print(
        json.dumps(
            {
                "status": "PASSED_A45BM5_TPAG_CALIBRATION_PROTOCOL_NO_INFERENCE",
                "calibration_units": 64,
                "proposition_rows": 512,
                "alignment_rows": 1280,
                "evidence_group_rows": 768,
                "claim_rows": 640,
                "readiness_requirements": 56,
                "maximum_joint_parameter_candidates": 343,
                "semantic_inference_authorized": 0,
                "model_bindings_authorized": 0,
                "model_family_comparisons_authorized": 0,
                "prompt_searches_authorized": 0,
                "threshold_searches_authorized": 0,
                "fresh_validation_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "next_action_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

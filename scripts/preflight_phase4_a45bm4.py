"""Fail-closed preflight for the A4.5b-M4 TPAG methodology decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm4_v1.json"
DECISION = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a45bm4_methodology_v1"
    / "decision_summary.json"
)
CLOSED_M3 = ROOT / "benchmarks" / "assistance" / "a45bm3_closure_v1.json"
SOURCE_MAIN_SHA = "595a1df9381b5c7d6e79042b0438eb12d39bff9b"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def main() -> None:
    config = _json(CONFIG)
    decision = _json(DECISION)
    closed = _json(CLOSED_M3)

    if config["checkpoint"] != "A4.5b-M4":
        raise RuntimeError("A4.5b-M4 checkpoint identity drifted")
    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b-M4 source main SHA drifted")
    if closed["scientific_pass"] is not False:
        raise RuntimeError("A4.5b-M4 requires the frozen negative M3 result")
    if closed["scientific_status"] != "FAILED_SCEC_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED":
        raise RuntimeError("A4.5b-M4 source scientific status drifted")
    calibration = closed["calibration"]
    if calibration["candidate_count"] != 609:
        raise RuntimeError("A4.5b-M4 requires the registered 609-candidate M3 grid")
    if calibration["feasible_candidate_count"] != 0:
        raise RuntimeError("A4.5b-M4 requires zero feasible M3 candidates")

    selected = config["selected_methodology"]
    if selected["name"] != "Typed Proposition Alignment Graph":
        raise RuntimeError("A4.5b-M4 selected methodology drifted")
    if selected["short_name"] != "TPAG":
        raise RuntimeError("A4.5b-M4 selected methodology short name drifted")
    if selected["novelty_claim"] is not False:
        raise RuntimeError("A4.5b-M4 must not claim TPAG novelty")
    if selected["binding_status"] != "UNBOUND":
        raise RuntimeError("A4.5b-M4 must remain implementation-unbound")

    retained = config["retained_semantic_principles"]
    if retained["compatibility_precedes_sufficiency"] is not True:
        raise RuntimeError("TPAG must preserve compatibility before sufficiency")
    if retained["sufficiency_is_set_level"] is not True:
        raise RuntimeError("TPAG must preserve set-level sufficiency")
    if retained["existing_aerf_atoms_preserved"] is not True:
        raise RuntimeError("TPAG must preserve existing AERF atoms")

    alignment = config["alignment_policy"]
    if alignment["whole_pair_nli_may_not_authoritatively_decide_all_slots"] is not True:
        raise RuntimeError("A4.5b-M4 must retire universal whole-pair NLI authority")
    if alignment["query_conditioning_required_for_any_future_residual_relation_model"] is not True:
        raise RuntimeError("Future learned residual relation models must be query-conditioned")

    governance = config["data_governance"]
    if governance["fresh_replacement_calibration_required_before_future_validation"] is not True:
        raise RuntimeError("A4.5b-M4 must require fresh TPAG calibration")
    if governance["a45a_fresh_validation_pairs_scored"] != 0:
        raise RuntimeError("A4.5a fresh validation pairs must remain unscored")
    if governance["a45a_fresh_validation_claims_scored"] != 0:
        raise RuntimeError("A4.5a fresh validation claims must remain unscored")
    if governance["confirmatory_records_inspected"] != 0:
        raise RuntimeError("Confirmatory records must remain unopened")
    if governance["confirmatory_queries_scored"] != 0:
        raise RuntimeError("Confirmatory queries must remain unscored")
    if governance["a45c_repurposed"] is not False:
        raise RuntimeError("A4.5c must not be repurposed")

    for name, value in config["scope"].items():
        if int(value) != 0:
            raise RuntimeError(f"A4.5b-M4 forbidden execution scope is nonzero: {name}")

    if decision["status"] != "CLOSED_METHODOLOGY_SELECTED_NO_INFERENCE":
        raise RuntimeError("A4.5b-M4 decision summary status drifted")
    method = decision["selected_methodology"]
    if method["implementation_bound"] is not False:
        raise RuntimeError("A4.5b-M4 cannot bind an implementation")
    if method["model_bound"] is not False:
        raise RuntimeError("A4.5b-M4 cannot bind a model")
    if method["thresholds_bound"] is not False:
        raise RuntimeError("A4.5b-M4 cannot bind thresholds")
    if decision["next_action"]["authorized"] is not False:
        raise RuntimeError("The fresh TPAG protocol requires separate approval")

    print(
        json.dumps(
            {
                "status": "PASSED_A45BM4_TPAG_METHODOLOGY_DECISION_NO_INFERENCE",
                "selected_methodology": "TPAG",
                "source_m3_feasible_candidates": 0,
                "semantic_inference_authorized": 0,
                "model_bindings_authorized": 0,
                "model_family_comparisons_authorized": 0,
                "threshold_searches_authorized": 0,
                "prompt_searches_authorized": 0,
                "fresh_validation_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "next_action_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

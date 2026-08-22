"""Fail-closed preflight for the A4.5b-M1 SCEC methodology decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm1_v1.json"
DECISION = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a45bm1_methodology_v1"
    / "decision_summary.json"
)
CLOSED_A45B = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a45b_calibration_postresult_v1"
    / "result_summary.json"
)
SOURCE_MAIN_SHA = "e27f7652032ad72c3bc526067a403b58bb50636c"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def main() -> None:
    config = _json(CONFIG)
    decision = _json(DECISION)
    closed = _json(CLOSED_A45B)

    if config["checkpoint"] != "A4.5b-M1":
        raise RuntimeError("A4.5b-M1 checkpoint identity drifted")
    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b-M1 source main SHA drifted")
    if closed["scientific_pass"] is not False:
        raise RuntimeError("A4.5b-M1 requires the frozen negative A4.5b result")
    if closed["threshold_selection"]["feasible_candidate_count"] != 0:
        raise RuntimeError("A4.5b-M1 requires zero feasible A4.5b threshold candidates")

    selected = config["selected_methodology"]
    if selected["name"] != "Scope-Conditioned Evidence Compatibility":
        raise RuntimeError("A4.5b-M1 selected methodology drifted")
    if selected["short_name"] != "SCEC":
        raise RuntimeError("A4.5b-M1 selected methodology short name drifted")
    if selected["novelty_claim"] is not False:
        raise RuntimeError("A4.5b-M1 must not claim SCEC novelty")
    if selected["binding_status"] != "UNBOUND":
        raise RuntimeError("A4.5b-M1 must remain implementation-unbound")

    constraints = config["design_constraints"]
    if constraints["single_scalar_retrieval_relevance_as_authoritative_gate"] is not False:
        raise RuntimeError("A4.5b-M1 cannot retain scalar retrieval relevance as authoritative gate")
    if constraints["sufficiency_is_set_level_property"] is not True:
        raise RuntimeError("A4.5b-M1 must preserve set-level sufficiency")
    if constraints["free_form_claim_redecomposition"] is not False:
        raise RuntimeError("A4.5b-M1 must preserve existing atomization")

    for name, value in config["scope"].items():
        if int(value) != 0:
            raise RuntimeError(f"A4.5b-M1 forbidden execution scope is nonzero: {name}")

    governance = config["data_governance"]
    if governance["fresh_replacement_calibration_required_before_future_validation"] is not True:
        raise RuntimeError("A4.5b-M1 must require fresh replacement calibration")
    if governance["a45a_fresh_validation_pairs_scored"] != 0:
        raise RuntimeError("A4.5a fresh validation must remain unscored")
    if governance["confirmatory_records_inspected"] != 0:
        raise RuntimeError("Confirmatory records must remain unopened")
    if governance["a45c_repurposed_for_scec"] is not False:
        raise RuntimeError("A4.5c must not be repurposed for SCEC")

    if decision["status"] != "CLOSED_METHODOLOGY_SELECTED_NO_INFERENCE":
        raise RuntimeError("A4.5b-M1 decision summary status drifted")
    if decision["selected_methodology"]["model_bound"] is not False:
        raise RuntimeError("A4.5b-M1 cannot bind a model")
    if decision["next_action"]["authorized"] is not False:
        raise RuntimeError("The next SCEC protocol requires separate approval")

    print(
        json.dumps(
            {
                "status": "PASSED_A45BM1_SCEC_METHODOLOGY_DECISION_NO_INFERENCE",
                "selected_methodology": "SCEC",
                "semantic_inference_authorized": 0,
                "model_bindings_authorized": 0,
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

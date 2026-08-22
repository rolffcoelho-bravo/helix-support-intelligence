"""Static tests for the A4.5b-M1 SCEC methodology decision."""

from __future__ import annotations

import json
from pathlib import Path

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


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_scec_is_selected_but_unbound() -> None:
    config = _json(CONFIG)
    selected = config["selected_methodology"]
    assert isinstance(selected, dict)
    assert selected["name"] == "Scope-Conditioned Evidence Compatibility"
    assert selected["short_name"] == "SCEC"
    assert selected["novelty_claim"] is False
    assert selected["binding_status"] == "UNBOUND"


def test_scec_separates_compatibility_from_sufficiency() -> None:
    config = _json(CONFIG)
    construct = config["construct_definition"]
    assert isinstance(construct, dict)
    excluded = construct["excluded_from_relevance_decision"]
    assert isinstance(excluded, list)
    assert "whether all decisive evidence is present" in excluded
    constraints = config["design_constraints"]
    assert isinstance(constraints, dict)
    assert constraints["sufficiency_is_set_level_property"] is True
    assert constraints["single_scalar_retrieval_relevance_as_authoritative_gate"] is False
    assert constraints["top1_span_only_as_authoritative_evidence_set"] is False


def test_scec_preserves_existing_atomization_and_seals_future_data() -> None:
    config = _json(CONFIG)
    constraints = config["design_constraints"]
    governance = config["data_governance"]
    assert isinstance(constraints, dict)
    assert isinstance(governance, dict)
    assert constraints["existing_aerf_atoms_preserved"] is True
    assert constraints["free_form_claim_redecomposition"] is False
    assert governance["fresh_replacement_calibration_required_before_future_validation"] is True
    assert governance["a45a_fresh_validation_pairs_scored"] == 0
    assert governance["a45a_fresh_validation_claims_scored"] == 0
    assert governance["confirmatory_records_inspected"] == 0
    assert governance["confirmatory_queries_scored"] == 0
    assert governance["a45c_repurposed_for_scec"] is False


def test_a45bm1_performs_no_scientific_execution() -> None:
    decision = _json(DECISION)
    scope = decision["scope"]
    assert isinstance(scope, dict)
    assert all(int(value) == 0 for value in scope.values())
    selected = decision["selected_methodology"]
    assert isinstance(selected, dict)
    assert selected["implementation_bound"] is False
    assert selected["model_bound"] is False
    assert selected["thresholds_bound"] is False
    next_action = decision["next_action"]
    assert isinstance(next_action, dict)
    assert next_action["authorized"] is False

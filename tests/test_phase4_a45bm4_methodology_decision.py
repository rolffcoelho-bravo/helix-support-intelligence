"""Static tests for the A4.5b-M4 TPAG methodology decision."""

from __future__ import annotations

import json
from pathlib import Path

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


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_tpag_is_selected_but_unbound() -> None:
    config = _json(CONFIG)
    selected = config["selected_methodology"]
    assert isinstance(selected, dict)
    assert selected["name"] == "Typed Proposition Alignment Graph"
    assert selected["short_name"] == "TPAG"
    assert selected["novelty_claim"] is False
    assert selected["binding_status"] == "UNBOUND"


def test_tpag_preserves_scec_semantics_and_retires_universal_nli() -> None:
    config = _json(CONFIG)
    retained = config["retained_semantic_principles"]
    alignment = config["alignment_policy"]
    assert isinstance(retained, dict)
    assert isinstance(alignment, dict)
    assert retained["compatibility_precedes_sufficiency"] is True
    assert retained["sufficiency_is_set_level"] is True
    assert retained["polarity_only_after_compatibility_and_sufficiency"] is True
    assert alignment["whole_pair_nli_may_not_authoritatively_decide_all_slots"] is True
    assert (
        alignment["learned_semantic_matching_must_be_restricted_to_unresolved_typed_edges"]
        is True
    )


def test_tpag_has_explicit_typed_frames_and_graph_coverage() -> None:
    config = _json(CONFIG)
    frame = config["typed_proposition_frame"]
    graph = config["coverage_graph"]
    assert isinstance(frame, dict)
    assert isinstance(graph, dict)
    assert frame["relation_states"] == ["MATCH", "MISMATCH", "UNSPECIFIED"]
    assert "target_value" in frame["core_slots"]
    assert "conditional_scope" in frame["qualifier_slots"]
    assert "modality_or_quantification" in frame["qualifier_slots"]
    assert graph["minimal_evidence_group_compatible"] is True
    assert "COVERS" in graph["edge_types"]


def test_tpag_is_justified_by_the_frozen_m3_failure() -> None:
    closed = _json(CLOSED_M3)
    calibration = closed["calibration"]
    assert isinstance(calibration, dict)
    assert closed["scientific_pass"] is False
    assert calibration["candidate_count"] == 609
    assert calibration["feasible_candidate_count"] == 0
    metrics = calibration["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["refutes_recall"] == 0.0
    assert metrics["contradicted_recall"] == 0.0
    assert metrics["conflict_detection_accuracy"] == 0.0


def test_tpag_keeps_future_evidence_sealed() -> None:
    config = _json(CONFIG)
    governance = config["data_governance"]
    assert isinstance(governance, dict)
    assert governance["fresh_replacement_calibration_required_before_future_validation"] is True
    assert governance["a45a_fresh_validation_pairs_scored"] == 0
    assert governance["a45a_fresh_validation_claims_scored"] == 0
    assert governance["confirmatory_records_inspected"] == 0
    assert governance["confirmatory_queries_scored"] == 0
    assert governance["a45c_repurposed"] is False


def test_a45bm4_performs_no_scientific_execution() -> None:
    decision = _json(DECISION)
    scope = decision["scope"]
    assert isinstance(scope, dict)
    assert all(int(value) == 0 for value in scope.values())
    method = decision["selected_methodology"]
    assert isinstance(method, dict)
    assert method["implementation_bound"] is False
    assert method["model_bound"] is False
    assert method["thresholds_bound"] is False
    next_action = decision["next_action"]
    assert isinstance(next_action, dict)
    assert next_action["authorized"] is False

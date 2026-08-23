"""Static and deterministic tests for the A4.5b-M5 TPAG calibration protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm5_v1.json"
MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm5_manifest_v1.json"
M2_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm2_manifest_v1.json"
BUILDER = ROOT / "benchmarks" / "assistance" / "tpag_calibration_a45bm5.py"

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
    assert isinstance(value, dict)
    return value


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("tpag_calibration_a45bm5", BUILDER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _by_subtype(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["subtype"]), []).append(row)
    return result


def test_fresh_manifest_reconstructs_exactly() -> None:
    module = _module()
    frozen = _json(MANIFEST)
    assert module.manifest() == frozen
    suite = module.build_suite()
    assert len(suite["units"]) == 64
    assert len(suite["proposition_rows"]) == 512
    assert len(suite["alignment_rows"]) == 1280
    assert len(suite["evidence_group_rows"]) == 768
    assert len(suite["claim_rows"]) == 640


def test_m5_corpus_is_distinct_from_m2_m3_calibration() -> None:
    module = _module()
    suite = module.build_suite()
    frozen = _json(MANIFEST)
    m2 = _json(M2_MANIFEST)
    assert frozen["corpus_id"] != m2["corpus_id"]
    assert frozen["seed"] != m2["seed"]
    assert frozen["governance"]["a45bm2_m3_rows_reused"] == 0
    serialized = json.dumps(suite, sort_keys=True)
    assert "Cobalt case" not in serialized
    assert "SCEC-C" not in serialized
    assert all(str(row["unit_id"]).startswith("TPAG-C") for row in suite["units"])


def test_four_measurement_layers_are_balanced_and_registered() -> None:
    frozen = _json(MANIFEST)
    counts = frozen["counts"]
    assert len(counts["proposition_subtypes"]) == 8
    assert len(counts["alignment_subtypes"]) == 20
    assert len(counts["evidence_group_subtypes"]) == 12
    assert len(counts["claim_categories"]) == 10
    assert set(counts["proposition_subtypes"].values()) == {64}
    assert set(counts["alignment_subtypes"].values()) == {64}
    assert set(counts["evidence_group_subtypes"].values()) == {64}
    assert set(counts["claim_categories"].values()) == {64}
    assert counts["alignment_compatibility"] == {
        "COMPATIBLE": 640,
        "INCOMPATIBLE": 640,
    }


def test_target_value_mismatch_is_compatible_refutation() -> None:
    module = _module()
    rows = _by_subtype(module.build_suite()["alignment_rows"])["direct_refutation_value_conflict"]
    assert len(rows) == 64
    for row in rows:
        gold = row["gold"]
        assert gold["slot_relations"]["target_value"] == "MISMATCH"
        assert gold["scope_compatibility"] == "COMPATIBLE"
        assert gold["coverage_status"] == "COMPLETE"
        assert gold["polarity"] == "REFUTES"
        assert gold["final_relation"] == "CONTRADICTED"


def test_target_slot_mismatch_is_surface_isolated() -> None:
    module = _module()
    suite = module.build_suite()
    units = {str(row["unit_id"]): row for row in suite["units"]}
    rows = _by_subtype(suite["alignment_rows"])["target_slot_identity_mismatch"]
    assert len(rows) == 64
    for row in rows:
        unit = units[str(row["unit_id"])]
        text = str(row["evidence_proposition"])
        assert str(unit["predicate"]) in text
        assert str(unit["alternate_slot"]) in text
        assert row["gold"]["slot_relations"]["predicate_or_event"] == "MATCH"
        assert row["gold"]["slot_relations"]["target_slot_identity"] == "MISMATCH"


def test_scope_identity_mismatches_veto_compatibility() -> None:
    module = _module()
    rows_by_subtype = _by_subtype(module.build_suite()["alignment_rows"])
    for subtype in SCOPE_MISMATCH_SUBTYPES:
        rows = rows_by_subtype[subtype]
        assert len(rows) == 64
        assert all(row["gold"]["scope_compatibility"] == "INCOMPATIBLE" for row in rows)
        assert all(row["gold"]["final_relation"] == "UNKNOWN" for row in rows)


def test_unspecified_required_slots_preserve_compatibility_but_not_coverage() -> None:
    module = _module()
    rows_by_subtype = _by_subtype(module.build_suite()["alignment_rows"])
    for subtype in UNSPECIFIED_SUBTYPES:
        rows = rows_by_subtype[subtype]
        assert len(rows) == 64
        for row in rows:
            gold = row["gold"]
            assert gold["scope_compatibility"] == "COMPATIBLE"
            assert gold["coverage_status"] == "INCOMPLETE"
            assert gold["polarity"] == "UNRESOLVED"
            assert gold["final_relation"] == "UNKNOWN"
            assert "UNSPECIFIED" in gold["slot_relations"].values()


def test_group_conflict_requires_same_scope() -> None:
    module = _module()
    groups = _by_subtype(module.build_suite()["evidence_group_rows"])
    same_scope = groups["same_scope_support_refute_conflict"]
    different_condition = groups["different_condition_not_conflict"]
    assert len(same_scope) == 64
    assert len(different_condition) == 64
    assert all(row["gold"]["sufficiency"] == "CONFLICTING" for row in same_scope)
    assert all(row["gold"]["final_relation"] == "CONFLICTING_EVIDENCE" for row in same_scope)
    assert all(row["gold"]["sufficiency"] == "SUFFICIENT" for row in different_condition)
    assert all(row["gold"]["final_relation"] == "ENTAILED" for row in different_condition)


def test_cross_span_scope_incoherence_cannot_fill_location() -> None:
    module = _module()
    rows = _by_subtype(module.build_suite()["evidence_group_rows"])["cross_span_scope_incoherence"]
    assert len(rows) == 64
    for row in rows:
        gold = row["gold"]
        assert gold["cross_proposition_scope_coherence"] == "INCOHERENT"
        assert gold["missing_decisive_slots"] == ["location_scope"]
        assert "location_scope" not in gold["covered_decisive_slots"]
        assert gold["sufficiency"] == "INSUFFICIENT"
        assert gold["final_relation"] == "UNKNOWN"


def test_predicate_paraphrase_is_surface_isolated_from_condition() -> None:
    module = _module()
    suite = module.build_suite()
    units = {str(row["unit_id"]): row for row in suite["units"]}
    rows = _by_subtype(suite["alignment_rows"])["predicate_paraphrase_match"]
    assert len(rows) == 64
    for row in rows:
        unit = units[str(row["unit_id"])]
        text = str(row["evidence_proposition"])
        assert str(unit["predicate_paraphrase"]) in text
        assert f"when {unit['condition']}" in text
        assert row["gold"]["slot_relations"]["predicate_or_event"] == "MATCH"
        assert row["gold"]["slot_relations"]["conditional_scope"] == "MATCH"


def test_parameter_budget_is_small_and_nonadaptive() -> None:
    config = _json(CONFIG)
    budget = config["future_calibration_parameter_budget"]
    assert budget["scalar_thresholds"] == [
        "extraction_confidence_min",
        "alignment_confidence_min",
        "polarity_confidence_min",
    ]
    assert budget["values"] == [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    assert budget["maximum_scalar_thresholds"] == 3
    assert budget["maximum_joint_candidates"] == 7**3 == 343
    assert budget["class_specific_thresholds_authorized"] is False
    assert budget["slot_specific_thresholds_authorized"] is False
    assert budget["post_result_grid_expansion_authorized"] is False
    assert budget["freeze_all_raw_outputs_before_threshold_selection"] is True


def test_protocol_keeps_all_execution_and_future_evidence_sealed() -> None:
    config = _json(CONFIG)
    manifest = _json(MANIFEST)
    assert config["methodology"]["binding_status"] == "UNBOUND"
    assert len(config["calibration_readiness_requirements"]) == 56
    assert all(int(value) == 0 for value in config["scope"].values())
    assert all(int(value) == 0 for value in manifest["governance"].values())
    governance = config["data_governance"]
    assert governance["a45a_fresh_validation_remains_sealed"] is True
    assert governance["a45a_fresh_validation_pairs_scored"] == 0
    assert governance["a45a_fresh_validation_claims_scored"] == 0
    assert governance["confirmatory_records_inspected"] == 0
    assert governance["confirmatory_queries_scored"] == 0
    assert governance["a45c_repurposed"] is False
    assert config["next_action"]["checkpoint"] == "A4.5b-M6"
    assert config["next_action"]["authorized"] is False

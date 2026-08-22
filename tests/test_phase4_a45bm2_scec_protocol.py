"""Registration tests for the A4.5b-M2 SCEC calibration protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "benchmarks" / "assistance" / "scec_calibration_a45bm2.py"
MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm2_manifest_v1.json"
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm2_v1.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("scec_calibration_a45bm2", BUILDER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_exactly_reproducible() -> None:
    module = _module()
    assert module.manifest() == _json(MANIFEST)


def test_fresh_calibration_counts_and_hashes_are_frozen() -> None:
    manifest = _json(MANIFEST)
    assert manifest["partition"] == "calibration_only"
    assert manifest["counts"]["units"] == 48
    assert manifest["counts"]["pair_rows"] == 768
    assert manifest["counts"]["evidence_set_rows"] == 384
    assert manifest["counts"]["claim_rows"] == 384
    assert set(manifest["counts"]["pair_subtypes"].values()) == {48}
    assert set(manifest["counts"]["evidence_set_subtypes"].values()) == {48}
    assert set(manifest["counts"]["claim_categories"].values()) == {48}
    assert manifest["sha256"] == {
        "units": "ece3b03fe215cb4847ec1e8ed71f05885bddc6807fe7a87691af55b05dc75d84",
        "pair_rows": "2ee18830fb2510aae85a936368adea72de145f82b2282831e0d2ee841546e12f",
        "evidence_set_rows": (
            "698b348b4bb5d5b00e597a7fcea144ce10ecd62e40036602ae5fa25577606d61"
        ),
        "claim_rows": "3dc8b01f797ca1b97ae5330929d8b462bf3b85b3d0788433c23026bf8bef262e",
    }


def test_relevant_but_insufficient_is_not_relabelled_irrelevant() -> None:
    rows = _module().build_suite()["pair_rows"]
    target = [
        row
        for row in rows
        if str(row["subtype"]).startswith("relevant_but_insufficient_")
    ]
    assert len(target) == 144
    assert all(row["gold"]["compatibility"] == "COMPATIBLE" for row in target)
    assert all(row["gold"]["pair_sufficiency"] == "INSUFFICIENT" for row in target)
    assert all(row["gold"]["final_relation"] == "UNKNOWN" for row in target)


def test_scope_mismatches_are_explicitly_incompatible() -> None:
    rows = _module().build_suite()["pair_rows"]
    mismatch_subtypes = {
        "entity_scope_mismatch",
        "predicate_scope_mismatch",
        "target_slot_mismatch",
        "temporal_scope_mismatch",
        "location_scope_mismatch",
        "organizational_scope_mismatch",
        "conditional_scope_mismatch",
        "modality_quantification_scope_mismatch",
    }
    target = [row for row in rows if row["subtype"] in mismatch_subtypes]
    assert len(target) == 384
    assert all(row["gold"]["compatibility"] == "INCOMPATIBLE" for row in target)
    assert all(row["gold"]["final_relation"] == "UNKNOWN" for row in target)


def test_complementary_evidence_and_scope_gap_are_distinct() -> None:
    rows = _module().build_suite()["evidence_set_rows"]
    complementary = [
        row for row in rows if row["subtype"] == "complementary_two_span_support"
    ]
    scope_gap = [
        row
        for row in rows
        if row["subtype"] == "compatible_multi_span_unresolved_scope_gap"
    ]
    assert len(complementary) == 48
    assert len(scope_gap) == 48
    assert all(row["gold"]["sufficiency"] == "SUFFICIENT" for row in complementary)
    assert all(row["gold"]["final_relation"] == "ENTAILED" for row in complementary)
    assert all(row["gold"]["sufficiency"] == "INSUFFICIENT" for row in scope_gap)
    assert all(row["gold"]["final_relation"] == "UNKNOWN" for row in scope_gap)


def test_protocol_remains_unbound_and_sealed() -> None:
    config = _json(CONFIG)
    manifest = _json(MANIFEST)
    assert config["methodology"]["binding_status"] == "UNBOUND"
    assert all(int(value) == 0 for value in config["scope"].values())
    assert config["data_governance"]["a45a_fresh_validation_remains_sealed"] is True
    assert config["data_governance"]["confirmatory_records_inspected"] == 0
    assert config["data_governance"]["a45c_repurposed_for_scec"] is False
    assert manifest["governance"]["a45b_calibration_rows_reused"] == 0
    assert manifest["governance"]["a45a_fresh_validation_rows_materialized"] == 0
    assert manifest["governance"]["confirmatory_query_records_inspected"] == 0
    assert config["next_action"]["authorized"] is False

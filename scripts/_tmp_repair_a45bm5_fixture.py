"""Temporary deterministic maintenance helper for the M5 pre-inference fixture erratum."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "benchmarks" / "assistance" / "tpag_calibration_a45bm5.py"
MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm5_manifest_v1.json"
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm5_v1.json"
DOC = ROOT / "docs" / "assistance-a45bm5-tpag-calibration-protocol.md"
TESTS = ROOT / "tests" / "test_phase4_a45bm5_calibration_protocol.py"
ERRATUM = ROOT / "benchmarks" / "assistance" / "a45bm5_erratum_v1.json"

SOURCE_MAIN_SHA = "9070fc4cf0447077a20c7e576e49e9ba5f0536ba"
OLD_ALIGNMENT_SHA = "c2b0bb66dc13a8c09560730337152ee5d0978b5a995bc7af3beb7782988c2845"

OLD_LINE = '    paraphrase_evidence = support.replace(unit["predicate"], unit["predicate_paraphrase"])'
NEW_BLOCK = '''    paraphrase_evidence = support.replace(
        f" and {unit['predicate']} them within",
        f" and {unit['predicate_paraphrase']} them within",
        1,
    )'''

TEST_MARKER = "def test_parameter_budget_is_small_and_nonadaptive() -> None:\n"
TEST_BLOCK = '''def test_predicate_paraphrase_is_surface_isolated_from_condition() -> None:
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


'''


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("tpag_calibration_a45bm5_erratum", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load corrected M5 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    old_manifest = _json(MANIFEST)
    if old_manifest["sha256"]["alignment_rows"] != OLD_ALIGNMENT_SHA:
        raise RuntimeError("M5 erratum expected the original frozen alignment hash")

    builder_text = BUILDER.read_text(encoding="utf-8")
    if OLD_LINE not in builder_text:
        raise RuntimeError("M5 A18 global replacement target was not found exactly once")
    if builder_text.count(OLD_LINE) != 1:
        raise RuntimeError("M5 A18 global replacement target was not unique")
    BUILDER.write_text(builder_text.replace(OLD_LINE, NEW_BLOCK), encoding="utf-8")

    module = _load_builder()
    new_manifest = module.manifest()
    new_alignment_sha = str(new_manifest["sha256"]["alignment_rows"])
    if new_alignment_sha == OLD_ALIGNMENT_SHA:
        raise RuntimeError("M5 erratum did not change the affected alignment hash")
    for key in ("units", "proposition_rows", "evidence_group_rows", "claim_rows"):
        if new_manifest["sha256"][key] != old_manifest["sha256"][key]:
            raise RuntimeError(f"M5 erratum unexpectedly changed {key}")
    if new_manifest["counts"] != old_manifest["counts"]:
        raise RuntimeError("M5 erratum unexpectedly changed corpus counts")
    if new_manifest["governance"] != old_manifest["governance"]:
        raise RuntimeError("M5 erratum unexpectedly changed governance")
    _write_json(MANIFEST, new_manifest)

    config = _json(CONFIG)
    config_hashes = config["fresh_calibration_construction"]["sha256"]
    if config_hashes["alignment_rows"] != OLD_ALIGNMENT_SHA:
        raise RuntimeError("M5 config did not contain the expected original alignment hash")
    config["fresh_calibration_construction"]["sha256"] = dict(new_manifest["sha256"])
    _write_json(CONFIG, config)

    doc_text = DOC.read_text(encoding="utf-8")
    if OLD_ALIGNMENT_SHA not in doc_text:
        raise RuntimeError("M5 protocol doc did not contain the original alignment hash")
    doc_text = doc_text.replace(OLD_ALIGNMENT_SHA, new_alignment_sha)
    erratum_section = f'''\n## Pre-inference fixture erratum\n\nBefore any A4.5b-M6 semantic inference, static structural testing found that the A18\n`predicate_paraphrase_match` generator used unrestricted string replacement. When a\npredicate token also appeared inside the conditional qualifier, that operation changed\nboth the intended predicate span and the condition while gold still marked\n`conditional_scope` as `MATCH`. The first observed case was `TPAG-C004-A18`.\n\nThe correction changes only A18 surface construction: the replacement is now restricted\nto the registered operation span `and <predicate> them within`, exactly once. Counts, gold\nlabels, readiness floors, units, proposition rows, evidence-group rows, claim rows, and all\ngovernance boundaries are unchanged.\n\nOriginal alignment SHA-256:\n\n`{OLD_ALIGNMENT_SHA}`\n\nCorrected alignment SHA-256:\n\n`{new_alignment_sha}`\n\nNo M6 model inference, threshold evaluation, A4.5a validation access, M2/M3 rescoring,\nor confirmatory access occurred before this repair. The abandoned M6 PR #50 was closed\nunmerged after the defect was detected by static CI.\n'''
    if "## Pre-inference fixture erratum" in doc_text:
        raise RuntimeError("M5 erratum section already exists")
    DOC.write_text(doc_text.rstrip() + "\n" + erratum_section, encoding="utf-8")

    tests_text = TESTS.read_text(encoding="utf-8")
    if "test_predicate_paraphrase_is_surface_isolated_from_condition" in tests_text:
        raise RuntimeError("M5 A18 erratum regression test already exists")
    if TEST_MARKER not in tests_text:
        raise RuntimeError("M5 test insertion marker was not found")
    TESTS.write_text(tests_text.replace(TEST_MARKER, TEST_BLOCK + TEST_MARKER), encoding="utf-8")

    erratum = {
        "erratum_id": "phase4-assistance-a4.5b-m5-preinference-fixture-erratum-v1",
        "status": "CLOSED_PREINFERENCE_FIXTURE_ERRATUM_NO_SEMANTIC_INFERENCE",
        "source_main_sha": SOURCE_MAIN_SHA,
        "affected_corpus_id": str(new_manifest["corpus_id"]),
        "affected_subtype": "predicate_paraphrase_match",
        "first_observed_case": "TPAG-C004-A18",
        "defect": (
            "A18 used unrestricted predicate string replacement, which could also mutate "
            "conditional-scope text containing the same token while gold remained MATCH."
        ),
        "repair": (
            "Restrict replacement to the registered operation span 'and <predicate> them within' "
            "and replace exactly once."
        ),
        "old_alignment_sha256": OLD_ALIGNMENT_SHA,
        "new_alignment_sha256": new_alignment_sha,
        "unchanged_sha256": {
            key: str(new_manifest["sha256"][key])
            for key in ("units", "proposition_rows", "evidence_group_rows", "claim_rows")
        },
        "counts_unchanged": True,
        "gold_labels_changed": 0,
        "readiness_requirements_changed": 0,
        "semantic_inference_before_repair": 0,
        "m6_model_calls_before_repair": 0,
        "m6_threshold_candidates_evaluated_before_repair": 0,
        "a45a_fresh_validation_rows_scored": 0,
        "confirmatory_records_inspected": 0,
        "confirmatory_queries_scored": 0,
        "closed_a45bm2_m3_rows_scored": 0,
        "a45c_repurposed": False,
        "abandoned_m6_pr": 50,
    }
    _write_json(ERRATUM, erratum)

    print(json.dumps(erratum, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

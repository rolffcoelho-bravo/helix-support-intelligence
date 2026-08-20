"""Preflight the frozen A4.4a compositional grounding protocol without inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from compositional_cases_a44a import (  # type: ignore[import-not-found]  # noqa: E402
    compositional_partition,
    generate_cases,
    suite_summary,
)
from grounding_anchors_a43a import (  # type: ignore[import-not-found]  # noqa: E402
    development_intents,
)

from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"
A43A_AUDIT_PATH = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a43a_validity_postresult_v1"
    / "forensic_audit.json"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def preflight() -> dict[str, Any]:
    """Validate A4.4a registration with zero model or candidate execution."""
    config = _load(CONFIG_PATH)
    predecessor = _load(A43A_AUDIT_PATH)
    bundle = generate_bundle()
    development = development_intents(bundle)
    partition = compositional_partition(bundle)
    cases = generate_cases(bundle)
    summary = suite_summary()

    if predecessor["status"] != "CLOSED_FAILED_EVALUATOR_VALIDITY":
        raise RuntimeError("A4.4a requires the closed negative A4.3a disposition.")
    disposition = predecessor["scientific_disposition"]
    if disposition["evaluation_verifier_rejected_for_future_candidate_selection"] is not True:
        raise RuntimeError("A4.4a requires the A4.3a evaluator rejection to remain frozen.")
    if disposition["confirmatory_partition_remains_unopened"] is not True:
        raise RuntimeError("A4.4a requires the confirmatory partition to remain unopened.")

    guards = config["execution_guards"]
    zero_guard_names = (
        "generator_calls",
        "openai_calls",
        "candidate_calls",
        "candidate_scoring",
        "semantic_verifier_calls",
        "model_family_searches",
        "confirmatory_query_scoring",
        "confirmatory_query_inspection",
    )
    for name in zero_guard_names:
        if int(guards[name]) != 0:
            raise RuntimeError(f"A4.4a execution guard {name} must remain zero.")
    if guards["a44b_not_authorized_by_this_gate"] is not True:
        raise RuntimeError("A4.4a must not authorize A4.4b.")

    semantic = config["component_boundaries"]["atomic_semantic_relation"]
    if semantic["binding_status"] != "UNBOUND":
        raise RuntimeError("A4.4a semantic verifier must remain unbound.")
    for field in ("model_family", "model_revision", "thresholds"):
        if semantic[field] is not None:
            raise RuntimeError(f"A4.4a semantic verifier field {field} must remain null.")
    if semantic["separate_future_approval_required"] is not True:
        raise RuntimeError("A4.4a future semantic binding must require separate approval.")

    if partition["calibration"] & partition["validation"]:
        raise RuntimeError("A4.4a calibration and validation intents overlap.")
    if partition["calibration"] | partition["validation"] != development:
        raise RuntimeError("A4.4a partitions must cover exactly the 60 development intents.")

    expected_counts = config["validation_suite"]["expected_counts"]
    if summary["rows"] != int(expected_counts["total"]):
        raise RuntimeError("A4.4a validation-suite total mismatch.")
    if summary["split_counts"]["calibration"] != int(expected_counts["calibration"]):
        raise RuntimeError("A4.4a calibration row count mismatch.")
    if summary["split_counts"]["validation"] != int(expected_counts["validation"]):
        raise RuntimeError("A4.4a validation row count mismatch.")

    case_ids = [str(row["case_id"]) for row in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("A4.4a case identifiers must be unique.")
    forbidden_keys = {"query_id", "query_text", "candidate_id", "candidate_output"}
    for row in cases:
        if forbidden_keys & set(row):
            raise RuntimeError("A4.4a cases must not carry query or candidate records.")

    documents = {str(row["document_id"]): dict(row) for row in bundle.documents}
    for row in cases:
        presented = {str(value) for value in row["presented_document_ids"]}
        cited = {str(value) for value in row["cited_document_ids"]}
        category = str(row["category"])
        verdict = str(row["expected_verdict"])

        if category == "citation_invalid":
            if cited <= presented:
                raise RuntimeError("A4.4a citation-invalid cases must contain an invalid citation.")
            if verdict != "CITATION_INVALID":
                raise RuntimeError("A4.4a citation-invalid verdict mismatch.")
        elif not cited <= presented:
            raise RuntimeError("Only A4.4a citation-invalid cases may cite absent documents.")

        if category == "stale_current_evidence":
            if verdict != "STALE_EVIDENCE":
                raise RuntimeError("A4.4a stale-evidence verdict mismatch.")
            if not all(documents[doc_id]["status"] == "archived" for doc_id in presented):
                raise RuntimeError("A4.4a stale cases must use archived presented evidence.")

        if category == "unresolved_conflict":
            if verdict != "CONFLICTING_EVIDENCE":
                raise RuntimeError("A4.4a conflict verdict mismatch.")
            if not any(bool(documents[doc_id]["conflict_fixture"]) for doc_id in presented):
                raise RuntimeError("A4.4a conflict cases must retain a conflict fixture.")

    future = config["future_binding_rules"]
    if future["semantic_verifier_model_search_allowed_in_a44a"] is not False:
        raise RuntimeError("A4.4a may not search replacement semantic verifier families.")
    if future["semantic_verifier_inference_allowed_in_a44a"] is not False:
        raise RuntimeError("A4.4a may not run semantic verifier inference.")
    if future["a44a_validation_split_may_influence_binding_or_thresholds"] is not False:
        raise RuntimeError("A4.4a validation data may not influence future binding.")

    return {
        "protocol_id": config["protocol_id"],
        "development_intents": len(development),
        "calibration_intents": len(partition["calibration"]),
        "validation_intents": len(partition["validation"]),
        "case_rows": summary["rows"],
        "calibration_rows": summary["split_counts"]["calibration"],
        "validation_rows": summary["split_counts"]["validation"],
        "suite_sha256": summary["sha256"],
        "category_counts": summary["category_counts"],
        "candidate_outputs_used": False,
        "candidate_calls_made": 0,
        "openai_calls_made": 0,
        "semantic_verifier_calls_made": 0,
        "model_family_searches_made": 0,
        "confirmatory_query_records_inspected": 0,
        "confirmatory_queries_scored": 0,
        "semantic_verifier_binding_status": semantic["binding_status"],
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))

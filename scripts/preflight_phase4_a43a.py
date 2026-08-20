"""Preflight the frozen A4.3a evidence and grounding-validity protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from grounding_anchors_a43a import (  # noqa: E402
    anchor_partition,
    development_intents,
    generate_anchors,
    suite_summary,
)

from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "models" / "assistance_validity_a43a_v1.json"
A42_AUDIT_PATH = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a42_development_postresult_v1"
    / "forensic_audit.json"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _eligible(document: dict[str, object]) -> bool:
    valid_to = document["valid_to"]
    return bool(
        document["status"] == "current"
        and document["permission"] == "public_support"
        and document["audience"] == "customer_support"
        and document["jurisdiction"] == "fictional-global"
        and str(document["valid_from"]) <= "2026-08-19"
        and (valid_to is None or str(valid_to) >= "2026-08-19")
    )


def _evidence_pack(
    query_id: str,
    documents: dict[str, dict[str, object]],
    judgments: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    ids = {
        str(row["document_id"])
        for row in judgments
        if row["query_id"] == query_id
        and int(row["relevance"]) >= 2
        and _eligible(documents[str(row["document_id"])])
    }
    return [documents[document_id] for document_id in sorted(ids)]


def _effective_decision(
    base_decision: str,
    case_type: str,
    evidence: list[dict[str, object]],
) -> str:
    if case_type == "ambiguous":
        return "ASK_FOR_CLARIFICATION"
    if case_type == "missing_evidence":
        return "ESCALATE_LOW_CONFIDENCE"
    if case_type == "conflicting_evidence":
        return "ESCALATE_CONFLICTING_EVIDENCE"
    if base_decision == "ANSWER_WITH_EVIDENCE" and any(
        bool(row["conflict_fixture"]) for row in evidence
    ):
        return "ESCALATE_CONFLICTING_EVIDENCE"
    return base_decision


def preflight() -> dict[str, Any]:
    """Validate A4.3a without model inference, candidate calls, or confirmatory scoring."""
    config = _load(CONFIG_PATH)
    audit = _load(A42_AUDIT_PATH)
    bundle = generate_bundle()
    development = development_intents(bundle)
    partitions = anchor_partition(bundle)
    anchors = generate_anchors(bundle)
    summary = suite_summary()

    if audit["status"] != "FAILED_SCIENTIFIC_VALIDITY_NO_SELECTION":
        raise RuntimeError("A4.3a requires the closed A4.2 no-selection disposition.")
    if audit["selection_admissible"] is not False:
        raise RuntimeError("A4.2 selection must remain inadmissible.")
    if int(audit["confirmatory_queries_opened"]) != 0:
        raise RuntimeError("A4.2 confirmatory partition must remain unopened.")
    if config["execution_guards"]["candidate_calls"] != 0:
        raise RuntimeError("A4.3a candidate calls must be frozen to zero.")
    if config["execution_guards"]["confirmatory_query_scoring"] != 0:
        raise RuntimeError("A4.3a confirmatory query scoring must be zero.")

    if partitions["calibration"] & partitions["validation"]:
        raise RuntimeError("A4.3a calibration and validation intents overlap.")
    if partitions["calibration"] | partitions["validation"] != development:
        raise RuntimeError("A4.3a anchor partitions must cover only development intents.")

    expected_counts = config["grounding_anchor_suite"]["expected_counts"]
    if summary["rows"] != int(expected_counts["total"]):
        raise RuntimeError("A4.3a anchor total does not match the frozen protocol.")
    if summary["split_counts"]["calibration"] != int(expected_counts["calibration"]):
        raise RuntimeError("A4.3a calibration anchor count mismatch.")
    if summary["split_counts"]["validation"] != int(expected_counts["validation"]):
        raise RuntimeError("A4.3a validation anchor count mismatch.")
    if any("query" in row for row in anchors):
        raise RuntimeError("A4.3a grounding anchors must not contain query records or text.")

    documents = {str(row["document_id"]): dict(row) for row in bundle.documents}
    reclassified = 0
    answer_with_conflict_after_resolution = 0
    development_queries_seen = 0
    for query in bundle.queries:
        intent = str(query["intent"])
        if intent not in development:
            continue
        development_queries_seen += 1
        evidence = _evidence_pack(str(query["query_id"]), documents, bundle.judgments)
        base = str(query["expected_decision"])
        effective = _effective_decision(base, str(query["case_type"]), evidence)
        if base == "ANSWER_WITH_EVIDENCE" and effective == "ESCALATE_CONFLICTING_EVIDENCE":
            reclassified += 1
        if effective == "ANSWER_WITH_EVIDENCE" and any(
            bool(row["conflict_fixture"]) for row in evidence
        ):
            answer_with_conflict_after_resolution += 1

    contract = config["evidence_contract_v2"]
    if reclassified != int(contract["known_development_reclassification_count_expected"]):
        raise RuntimeError("A4.3a evidence-contract reclassification count mismatch.")
    if answer_with_conflict_after_resolution != int(
        contract["answer_with_evidence_packs_with_current_conflict_after_resolution_expected"]
    ):
        raise RuntimeError("A4.3a still permits an answer decision with unresolved conflict.")

    return {
        "validity_id": config["validity_id"],
        "evidence_contract_id": contract["contract_id"],
        "development_intents": len(development),
        "development_queries_inspected": development_queries_seen,
        "confirmatory_queries_scored": 0,
        "confirmatory_query_metrics_computed": 0,
        "answer_queries_reclassified_to_conflict": reclassified,
        "answer_with_unresolved_conflict_after_resolution": answer_with_conflict_after_resolution,
        "anchor_rows": summary["rows"],
        "anchor_calibration_rows": summary["split_counts"]["calibration"],
        "anchor_validation_rows": summary["split_counts"]["validation"],
        "anchor_sha256": summary["sha256"],
        "candidate_outputs_used": False,
        "candidate_calls_made": 0,
        "openai_calls_made": 0,
        "nli_calls_made": 0,
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))

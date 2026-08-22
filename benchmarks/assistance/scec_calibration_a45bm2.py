"""Candidate-independent SCEC calibration construction for Phase 4 A4.5b-M2.

This module performs no learned inference and creates calibration-only fictional
support-policy fixtures. It does not materialize A4.5a validation or confirmatory data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEED = 20260822
CORPUS_ID = "helix-scec-calibration-corpus-v1"
PROTOCOL_ID = "phase4-assistance-a4.5b-m2-scec-calibration-v1"

QUEUES = (
    "access_review",
    "payments_review",
    "transfers_review",
    "cash_operations",
    "identity_review",
    "account_services",
)
REQUIREMENTS = (
    "identity confirmation",
    "transaction reference",
    "device confirmation",
    "recipient details",
    "statement excerpt",
    "account ownership proof",
)
REGIONS = (
    "north region",
    "south region",
    "east region",
    "west region",
)
CONDITIONS = (
    "priority review applies",
    "manual escalation applies",
    "enhanced verification applies",
    "supervisor review applies",
)
WINDOWS = (1, 2, 3, 4, 5, 7)
YEARS = (2026, 2027)

DIMENSIONS = (
    "entity",
    "predicate",
    "target_slot",
    "temporal_scope",
    "location_scope",
    "organizational_scope",
    "conditional_scope",
    "modality_quantification_scope",
)


def _alt(value: str, values: tuple[str, ...]) -> str:
    index = values.index(value)
    return values[(index + 1) % len(values)]


def build_units() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for index in range(48):
        number = index + 1
        unit_id = f"SCEC-C{number:03d}"
        queue = QUEUES[index % len(QUEUES)]
        requirement = REQUIREMENTS[(index * 5 + 2) % len(REQUIREMENTS)]
        region = REGIONS[(index * 3 + 1) % len(REGIONS)]
        condition = CONDITIONS[(index * 7 + 2) % len(CONDITIONS)]
        window = WINDOWS[(index * 5 + 3) % len(WINDOWS)]
        year = YEARS[index % len(YEARS)]
        subject = f"Cobalt case {number:03d}"
        units.append(
            {
                "unit_id": unit_id,
                "subject": subject,
                "queue": queue,
                "alternate_queue": _alt(queue, QUEUES),
                "requirement": requirement,
                "alternate_requirement": _alt(requirement, REQUIREMENTS),
                "region": region,
                "alternate_region": _alt(region, REGIONS),
                "condition": condition,
                "alternate_condition": _alt(condition, CONDITIONS),
                "window_days": window,
                "alternate_window_days": WINDOWS[(WINDOWS.index(window) + 1) % len(WINDOWS)],
                "year": year,
                "alternate_year": YEARS[(YEARS.index(year) + 1) % len(YEARS)],
            }
        )
    return units


def _dims(**overrides: str) -> dict[str, str]:
    values = {dimension: "MATCH" for dimension in DIMENSIONS}
    values.update(overrides)
    return values


def _pair(
    unit: dict[str, Any],
    suffix: str,
    subtype: str,
    claim: str,
    evidence_text: str,
    minimal_span: str | None,
    dimensions: dict[str, str],
    compatibility: str,
    expected_pair_sufficiency: str,
    expected_pair_polarity: str,
    expected_relation: str,
) -> dict[str, Any]:
    return {
        "pair_id": f"{unit['unit_id']}-{suffix}",
        "split": "calibration",
        "unit_id": unit["unit_id"],
        "subtype": subtype,
        "claim": claim,
        "evidence_document_id": f"{unit['unit_id']}-{suffix}-D",
        "evidence_text": evidence_text,
        "gold": {
            "compatibility_dimensions": dimensions,
            "compatibility": compatibility,
            "pair_sufficiency": expected_pair_sufficiency,
            "pair_polarity": expected_pair_polarity,
            "final_relation": expected_relation,
            "minimal_compatible_span": minimal_span,
        },
    }


def _claim_text(unit: dict[str, Any]) -> str:
    return (
        f"During {unit['year']}, the {unit['queue']} queue must complete "
        f"all {unit['subject'].lower()} requests from the {unit['region']} within "
        f"{unit['window_days']} business days when {unit['condition']}."
    )


def _unit_pairs(unit: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    claim = _claim_text(unit)
    subject = unit["subject"]
    queue = unit["queue"]
    region = unit["region"]
    condition = unit["condition"]
    window = unit["window_days"]
    year = unit["year"]
    support = (
        f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
        f"from the {region} within {window} business days when {condition}."
    )
    refute = (
        f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
        f"from the {region} within {unit['alternate_window_days']} business days "
        f"when {condition}."
    )
    missing_value = (
        f"During {year}, the {queue} queue's completion window for {subject.lower()} "
        f"requests from the {region} applies when {condition}; the duration is listed "
        "in the escalation schedule."
    )
    missing_time = (
        f"The {queue} queue must complete all {subject.lower()} requests from the {region} "
        f"within {window} business days when {condition}."
    )
    distractor = f"{subject} review also requires {unit['requirement']}."
    rows = [
        _pair(
            unit,
            "P01",
            "direct_support_full_scope",
            claim,
            f"{distractor} {support}",
            support,
            _dims(),
            "COMPATIBLE",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _pair(
            unit,
            "P02",
            "direct_refutation_full_scope",
            claim,
            f"{refute} {distractor}",
            refute,
            _dims(),
            "COMPATIBLE",
            "SUFFICIENT",
            "REFUTES",
            "CONTRADICTED",
        ),
        _pair(
            unit,
            "P03",
            "entity_scope_mismatch",
            claim,
            (
                f"During {year}, the {queue} queue must complete "
                f"all {other['subject'].lower()} requests from the {region} within {window} "
                f"business days when {condition}."
            ),
            None,
            _dims(entity="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P04",
            "predicate_scope_mismatch",
            claim,
            (
                f"During {year}, the {queue} queue must archive all {subject.lower()} requests "
                f"from the {region} within {window} business days when {condition}."
            ),
            None,
            _dims(predicate="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P05",
            "target_slot_mismatch",
            claim,
            (
                f"During {year}, the {queue} queue must begin all {subject.lower()} requests "
                f"from the {region} within {window} business days when {condition}."
            ),
            None,
            _dims(target_slot="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P06",
            "temporal_scope_mismatch",
            claim,
            (
                f"During {unit['alternate_year']}, the {queue} queue must complete "
                f"all {subject.lower()} requests from the {region} within {window} "
                f"business days when {condition}."
            ),
            None,
            _dims(temporal_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P07",
            "location_scope_mismatch",
            claim,
            (
                f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
                f"from the {unit['alternate_region']} within {window} business days "
                f"when {condition}."
            ),
            None,
            _dims(location_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P08",
            "organizational_scope_mismatch",
            claim,
            (
                f"During {year}, the {unit['alternate_queue']} queue must complete all "
                f"{subject.lower()} requests from the {region} within {window} business days "
                f"when {condition}."
            ),
            None,
            _dims(organizational_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P09",
            "conditional_scope_mismatch",
            claim,
            (
                f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
                f"from the {region} within {window} business days when "
                f"{unit['alternate_condition']}."
            ),
            None,
            _dims(conditional_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P10",
            "modality_quantification_scope_mismatch",
            claim,
            (
                f"During {year}, the {queue} queue may complete some {subject.lower()} requests "
                f"from the {region} within {window} business days when {condition}."
            ),
            None,
            _dims(modality_quantification_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P11",
            "relevant_but_insufficient_missing_value",
            claim,
            missing_value,
            missing_value,
            _dims(),
            "COMPATIBLE",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P12",
            "relevant_but_insufficient_missing_temporal_scope",
            claim,
            missing_time,
            missing_time,
            _dims(temporal_scope="UNSPECIFIED"),
            "COMPATIBLE",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P13",
            "relevant_but_insufficient_missing_conditional_scope",
            claim,
            (
                f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
                f"from the {region} within {window} business days under the applicable "
                "escalation condition."
            ),
            (
                f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
                f"from the {region} within {window} business days under the applicable "
                "escalation condition."
            ),
            _dims(conditional_scope="UNSPECIFIED"),
            "COMPATIBLE",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P14",
            "same_domain_irrelevance",
            claim,
            (
                f"During {year}, {subject.lower()} requests from the {region} require "
                f"{unit['requirement']} when {condition}."
            ),
            None,
            _dims(predicate="MISMATCH", target_slot="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P15",
            "cross_document_irrelevance",
            claim,
            (
                f"During {other['year']}, the {other['queue']} queue must complete "
                f"{other['subject'].lower()} requests from the {other['region']} within "
                f"{other['window_days']} business days when {other['condition']}."
            ),
            None,
            _dims(
                entity="MISMATCH",
                temporal_scope="MISMATCH" if other["year"] != year else "MATCH",
                location_scope="MISMATCH" if other["region"] != region else "MATCH",
                organizational_scope="MISMATCH" if other["queue"] != queue else "MATCH",
                conditional_scope="MISMATCH" if other["condition"] != condition else "MATCH",
            ),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _pair(
            unit,
            "P16",
            "context_contamination_support",
            claim,
            f"{support} {subject} requests are not automatically approved without review.",
            support,
            _dims(),
            "COMPATIBLE",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
    ]
    return rows


def _set_row(
    unit: dict[str, Any],
    suffix: str,
    subtype: str,
    claim: str,
    evidence_spans: list[str],
    covered_slots: list[str],
    missing_slots: list[str],
    compatibility: str,
    sufficiency: str,
    polarity: str,
    final_relation: str,
) -> dict[str, Any]:
    return {
        "set_id": f"{unit['unit_id']}-{suffix}",
        "split": "calibration",
        "unit_id": unit["unit_id"],
        "subtype": subtype,
        "claim": claim,
        "evidence_spans": evidence_spans,
        "gold": {
            "set_compatibility": compatibility,
            "covered_decisive_slots": covered_slots,
            "missing_decisive_slots": missing_slots,
            "sufficiency": sufficiency,
            "polarity": polarity,
            "final_relation": final_relation,
        },
    }


def _unit_sets(unit: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    claim = _claim_text(unit)
    subject = unit["subject"]
    queue = unit["queue"]
    region = unit["region"]
    condition = unit["condition"]
    window = unit["window_days"]
    year = unit["year"]

    support = (
        f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
        f"from the {region} within {window} business days when {condition}."
    )
    refute = (
        f"During {year}, the {queue} queue must complete all {subject.lower()} requests "
        f"from the {region} within {unit['alternate_window_days']} business days "
        f"when {condition}."
    )
    partial_scope = (
        f"During {year}, {subject.lower()} completion windows from the {region} "
        f"apply when {condition}."
    )
    partial_value = (
        f"The {queue} queue must use a {window}-business-day completion window for all "
        f"{subject.lower()} requests."
    )
    missing_value = (
        f"During {year}, the {queue} queue's completion window for {subject.lower()} "
        f"requests from the {region} applies when {condition}; the duration is listed "
        "in the escalation schedule."
    )
    missing_time = (
        f"The {queue} queue must complete all {subject.lower()} requests from the {region} "
        f"within {window} business days when {condition}."
    )
    irrelevant = (
        f"During {other['year']}, the {other['queue']} queue must complete "
        f"{other['subject'].lower()} requests within {other['window_days']} business days."
    )
    all_slots = [
        "entity",
        "predicate",
        "target_slot_identity",
        "target_value",
        "temporal_scope",
        "location_scope",
        "organizational_scope",
        "conditional_scope",
        "modality_quantification_scope",
    ]
    return [
        _set_row(
            unit,
            "S01",
            "single_span_complete_support",
            claim,
            [support],
            all_slots,
            [],
            "COMPATIBLE",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _set_row(
            unit,
            "S02",
            "single_span_complete_refutation",
            claim,
            [refute],
            all_slots,
            [],
            "COMPATIBLE",
            "SUFFICIENT",
            "REFUTES",
            "CONTRADICTED",
        ),
        _set_row(
            unit,
            "S03",
            "compatible_incomplete_missing_value",
            claim,
            [missing_value],
            [slot for slot in all_slots if slot != "target_value"],
            ["target_value"],
            "COMPATIBLE",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _set_row(
            unit,
            "S04",
            "compatible_incomplete_scope_gap",
            claim,
            [missing_time],
            [slot for slot in all_slots if slot != "temporal_scope"],
            ["temporal_scope"],
            "COMPATIBLE",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _set_row(
            unit,
            "S05",
            "complementary_two_span_support",
            claim,
            [partial_scope, partial_value],
            all_slots,
            [],
            "COMPATIBLE",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _set_row(
            unit,
            "S06",
            "complete_support_with_irrelevant_distractor",
            claim,
            [irrelevant, support],
            all_slots,
            [],
            "COMPATIBLE",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _set_row(
            unit,
            "S07",
            "support_refute_conflict",
            claim,
            [support, refute],
            all_slots,
            [],
            "COMPATIBLE",
            "CONFLICTING",
            "CONFLICTING",
            "CONFLICTING_EVIDENCE",
        ),
        _set_row(
            unit,
            "S08",
            "compatible_multi_span_unresolved_scope_gap",
            claim,
            [partial_value, missing_time],
            [slot for slot in all_slots if slot != "temporal_scope"],
            ["temporal_scope"],
            "COMPATIBLE",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
    ]


def _claim_row(
    unit: dict[str, Any],
    suffix: str,
    category: str,
    set_relations: list[str],
    expected_verdict: str,
    deterministic_gate: str = "NONE",
) -> dict[str, Any]:
    return {
        "case_id": f"{unit['unit_id']}-{suffix}",
        "split": "calibration",
        "unit_id": unit["unit_id"],
        "category": category,
        "set_relations": set_relations,
        "deterministic_gate": deterministic_gate,
        "expected_verdict": expected_verdict,
    }


def _unit_claims(unit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _claim_row(unit, "C01", "single_supported", ["ENTAILED"], "SUPPORTED"),
        _claim_row(unit, "C02", "single_refuted", ["CONTRADICTED"], "UNSUPPORTED"),
        _claim_row(unit, "C03", "compatible_insufficient", ["UNKNOWN"], "UNSUPPORTED"),
        _claim_row(unit, "C04", "complementary_multi_span_supported", ["ENTAILED"], "SUPPORTED"),
        _claim_row(
            unit,
            "C05",
            "support_refute_conflict",
            ["CONFLICTING_EVIDENCE"],
            "CONFLICTING_EVIDENCE",
        ),
        _claim_row(
            unit,
            "C06",
            "citation_invalid",
            ["ENTAILED"],
            "CITATION_INVALID",
            "CITATION_INVALID",
        ),
        _claim_row(
            unit,
            "C07",
            "stale_evidence",
            ["ENTAILED"],
            "STALE_EVIDENCE",
            "STALE_EVIDENCE",
        ),
        _claim_row(
            unit,
            "C08",
            "registered_conflict",
            ["ENTAILED"],
            "CONFLICTING_EVIDENCE",
            "REGISTERED_CONFLICT",
        ),
    ]


def build_suite() -> dict[str, Any]:
    units = build_units()
    pairs: list[dict[str, Any]] = []
    evidence_sets: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        other = units[(index + 1) % len(units)]
        pairs.extend(_unit_pairs(unit, other))
        evidence_sets.extend(_unit_sets(unit, other))
        claims.extend(_unit_claims(unit))
    pairs.sort(key=lambda row: str(row["pair_id"]))
    evidence_sets.sort(key=lambda row: str(row["set_id"]))
    claims.sort(key=lambda row: str(row["case_id"]))
    return {
        "protocol_id": PROTOCOL_ID,
        "corpus_id": CORPUS_ID,
        "units": units,
        "pair_rows": pairs,
        "evidence_set_rows": evidence_sets,
        "claim_rows": claims,
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def manifest() -> dict[str, Any]:
    suite = build_suite()
    units = suite["units"]
    pairs = suite["pair_rows"]
    sets = suite["evidence_set_rows"]
    claims = suite["claim_rows"]
    pair_subtypes = sorted({str(row["subtype"]) for row in pairs})
    set_subtypes = sorted({str(row["subtype"]) for row in sets})
    claim_categories = sorted({str(row["category"]) for row in claims})
    return {
        "protocol_id": PROTOCOL_ID,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "partition": "calibration_only",
        "counts": {
            "units": len(units),
            "pair_rows": len(pairs),
            "evidence_set_rows": len(sets),
            "claim_rows": len(claims),
            "pair_subtypes": {
                name: sum(row["subtype"] == name for row in pairs) for name in pair_subtypes
            },
            "evidence_set_subtypes": {
                name: sum(row["subtype"] == name for row in sets) for name in set_subtypes
            },
            "claim_categories": {
                name: sum(row["category"] == name for row in claims) for name in claim_categories
            },
            "pair_compatibility": {
                label: sum(row["gold"]["compatibility"] == label for row in pairs)
                for label in ("COMPATIBLE", "INCOMPATIBLE")
            },
            "pair_relations": {
                label: sum(row["gold"]["final_relation"] == label for row in pairs)
                for label in ("ENTAILED", "CONTRADICTED", "UNKNOWN")
            },
            "evidence_set_relations": {
                label: sum(row["gold"]["final_relation"] == label for row in sets)
                for label in ("ENTAILED", "CONTRADICTED", "UNKNOWN", "CONFLICTING_EVIDENCE")
            },
        },
        "sha256": {
            "units": hashlib.sha256(_jsonl(units)).hexdigest(),
            "pair_rows": hashlib.sha256(_jsonl(pairs)).hexdigest(),
            "evidence_set_rows": hashlib.sha256(_jsonl(sets)).hexdigest(),
            "claim_rows": hashlib.sha256(_jsonl(claims)).hexdigest(),
        },
        "governance": {
            "candidate_model_calls": 0,
            "model_bindings": 0,
            "threshold_searches": 0,
            "a45a_fresh_validation_rows_materialized": 0,
            "a45a_fresh_validation_rows_scored": 0,
            "confirmatory_query_records_inspected": 0,
            "confirmatory_queries_scored": 0,
            "a45b_calibration_rows_reused": 0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))

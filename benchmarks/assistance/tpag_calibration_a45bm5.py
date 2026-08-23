"""Candidate-independent TPAG calibration construction for Phase 4 A4.5b-M5.

This module performs no learned inference and creates calibration-only fictional
support-policy fixtures. It does not materialize A4.5a validation or confirmatory data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEED = 20260823
CORPUS_ID = "helix-tpag-calibration-corpus-v1"
PROTOCOL_ID = "phase4-assistance-a4.5b-m5-tpag-calibration-v1"

SERVICE_LINES = (
    "benefits_intake",
    "merchant_services",
    "document_review",
    "account_maintenance",
    "claims_support",
    "identity_operations",
    "payment_operations",
    "case_resolution",
)
PREDICATES = (
    "resolve",
    "acknowledge",
    "review",
    "release",
    "verify",
    "escalate",
    "record",
    "finalize",
)
TARGET_SLOTS = (
    "resolution_deadline",
    "acknowledgement_deadline",
    "review_deadline",
    "release_deadline",
    "verification_deadline",
    "escalation_deadline",
    "recording_deadline",
    "finalization_deadline",
)
REGIONS = (
    "amber district",
    "birch district",
    "cinder district",
    "delta district",
    "ember district",
    "fjord district",
    "grove district",
    "harbor district",
)
ORGANIZATIONS = (
    "Atlas desk",
    "Beacon desk",
    "Cedar desk",
    "Drift desk",
    "Elm desk",
    "Flint desk",
    "Gale desk",
    "Haven desk",
)
CONDITIONS = (
    "priority handling is active",
    "enhanced review is active",
    "manual verification is active",
    "supervisor review is active",
    "exception handling is active",
    "secondary review is active",
    "risk review is active",
    "document escalation is active",
)
WINDOWS = (2, 3, 4, 5, 6, 7, 8, 9)
YEARS = (2028, 2029, 2030, 2031)
MODALITIES = (
    "must process every",
    "must process all",
    "is required to process every",
    "is required to process all",
)

SCOPE_SLOTS = (
    "entity_or_subject",
    "predicate_or_event",
    "target_slot_identity",
    "temporal_scope",
    "location_scope",
    "organizational_scope",
    "conditional_scope",
    "modality_or_quantification",
)
CONTENT_SLOTS = ("target_value",)
ALL_SLOTS = (
    "entity_or_subject",
    "predicate_or_event",
    "target_slot_identity",
    "target_value",
    "temporal_scope",
    "location_scope",
    "organizational_scope",
    "conditional_scope",
    "modality_or_quantification",
)
RELATION_STATES = ("MATCH", "MISMATCH", "UNSPECIFIED")


def _alt(value: str, values: tuple[str, ...]) -> str:
    index = values.index(value)
    return values[(index + 1) % len(values)]


def build_units() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for index in range(64):
        number = index + 1
        service_line = SERVICE_LINES[index % len(SERVICE_LINES)]
        predicate = PREDICATES[(index * 3 + 1) % len(PREDICATES)]
        slot = TARGET_SLOTS[PREDICATES.index(predicate)]
        region = REGIONS[(index * 5 + 2) % len(REGIONS)]
        organization = ORGANIZATIONS[(index * 7 + 3) % len(ORGANIZATIONS)]
        condition = CONDITIONS[(index * 3 + 4) % len(CONDITIONS)]
        window = WINDOWS[(index * 5 + 1) % len(WINDOWS)]
        year = YEARS[(index * 3 + 2) % len(YEARS)]
        modality = MODALITIES[index % len(MODALITIES)]
        subject = f"Orchid request {number:03d}"
        units.append(
            {
                "unit_id": f"TPAG-C{number:03d}",
                "subject": subject,
                "subject_alias": f"OR-{number:03d}",
                "service_line": service_line,
                "predicate": predicate,
                "predicate_paraphrase": f"complete {predicate} handling",
                "target_slot_identity": slot,
                "target_value": window,
                "alternate_target_value": WINDOWS[(WINDOWS.index(window) + 1) % len(WINDOWS)],
                "year": year,
                "alternate_year": YEARS[(YEARS.index(year) + 1) % len(YEARS)],
                "region": region,
                "alternate_region": _alt(region, REGIONS),
                "organization": organization,
                "alternate_organization": _alt(organization, ORGANIZATIONS),
                "condition": condition,
                "alternate_condition": _alt(condition, CONDITIONS),
                "modality": modality,
                "alternate_modality": "may process some",
                "alternate_predicate": _alt(predicate, PREDICATES),
                "alternate_slot": _alt(slot, TARGET_SLOTS),
            }
        )
    return units


def _claim_text(unit: dict[str, Any]) -> str:
    return (
        f"During {unit['year']}, {unit['organization']} in the {unit['region']} "
        f"{unit['modality']} {unit['subject'].lower()} records in "
        f"{unit['service_line']} and {unit['predicate']} them within "
        f"{unit['target_value']} business days when {unit['condition']}."
    )


def _frame(unit: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "entity_or_subject": unit["subject"],
        "predicate_or_event": unit["predicate"],
        "target_slot_identity": unit["target_slot_identity"],
        "target_value": unit["target_value"],
        "temporal_scope": unit["year"],
        "location_scope": unit["region"],
        "organizational_scope": unit["organization"],
        "conditional_scope": unit["condition"],
        "modality_or_quantification": unit["modality"],
    }
    frame.update(overrides)
    return frame


def _relations(**overrides: str) -> dict[str, str]:
    values = {slot: "MATCH" for slot in ALL_SLOTS}
    values.update(overrides)
    return values


def _support(unit: dict[str, Any]) -> str:
    return _claim_text(unit)


def _refute(unit: dict[str, Any]) -> str:
    return (
        f"During {unit['year']}, {unit['organization']} in the {unit['region']} "
        f"{unit['modality']} {unit['subject'].lower()} records in "
        f"{unit['service_line']} and {unit['predicate']} them within "
        f"{unit['alternate_target_value']} business days when {unit['condition']}."
    )


def _proposition_row(
    unit: dict[str, Any],
    suffix: str,
    subtype: str,
    document_text: str,
    surface_propositions: list[str],
    target_indices: list[int],
    decontextualized_targets: list[str],
) -> dict[str, Any]:
    return {
        "proposition_case_id": f"{unit['unit_id']}-{suffix}",
        "split": "calibration",
        "unit_id": unit["unit_id"],
        "subtype": subtype,
        "claim": _claim_text(unit),
        "document_text": document_text,
        "gold": {
            "surface_propositions": surface_propositions,
            "target_proposition_indices": target_indices,
            "decontextualized_target_propositions": decontextualized_targets,
            "target_frames": [_frame(unit) for _ in decontextualized_targets],
        },
    }


def _unit_propositions(unit: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    support = _support(unit)
    unrelated = (
        f"{other['organization']} maintains {other['subject'].lower()} records for "
        f"{other['service_line']}."
    )
    alias_context = f"{unit['subject']} is referenced internally as {unit['subject_alias']}."
    alias_target = support.replace(unit["subject"].lower(), unit["subject_alias"])
    pronoun_context = f"The active record is {unit['subject']}."
    pronoun_target = support.replace(unit["subject"].lower(), "it")
    coordinated_a = support
    coordinated_b = f"{unit['organization']} must archive the audit note after the case is closed."
    parenthetical = support.replace(
        f"{unit['organization']} in the {unit['region']}",
        f"{unit['organization']} (the assigned unit) in the {unit['region']}",
    )
    return [
        _proposition_row(
            unit,
            "X01",
            "single_clean_target",
            support,
            [support],
            [0],
            [support],
        ),
        _proposition_row(
            unit,
            "X02",
            "target_after_unrelated_prefix",
            f"{unrelated} {support}",
            [unrelated, support],
            [1],
            [support],
        ),
        _proposition_row(
            unit,
            "X03",
            "target_before_unrelated_suffix",
            f"{support} {unrelated}",
            [support, unrelated],
            [0],
            [support],
        ),
        _proposition_row(
            unit,
            "X04",
            "alias_requires_decontextualization",
            f"{alias_context} {alias_target}",
            [alias_context, alias_target],
            [1],
            [support],
        ),
        _proposition_row(
            unit,
            "X05",
            "pronoun_requires_decontextualization",
            f"{pronoun_context} {pronoun_target}",
            [pronoun_context, pronoun_target],
            [1],
            [support],
        ),
        _proposition_row(
            unit,
            "X06",
            "coordinated_independent_propositions",
            f"{coordinated_a} {coordinated_b}",
            [coordinated_a, coordinated_b],
            [0],
            [coordinated_a],
        ),
        _proposition_row(
            unit,
            "X07",
            "parenthetical_context_target",
            parenthetical,
            [parenthetical],
            [0],
            [support],
        ),
        _proposition_row(
            unit,
            "X08",
            "no_target_proposition",
            unrelated,
            [unrelated],
            [],
            [],
        ),
    ]


def _alignment_row(
    unit: dict[str, Any],
    suffix: str,
    subtype: str,
    evidence_proposition: str,
    evidence_frame: dict[str, Any],
    relations: dict[str, str],
    compatibility: str,
    coverage: str,
    polarity: str,
    final_relation: str,
) -> dict[str, Any]:
    return {
        "alignment_id": f"{unit['unit_id']}-{suffix}",
        "split": "calibration",
        "unit_id": unit["unit_id"],
        "subtype": subtype,
        "claim": _claim_text(unit),
        "claim_frame": _frame(unit),
        "evidence_proposition": evidence_proposition,
        "evidence_frame": evidence_frame,
        "gold": {
            "slot_relations": relations,
            "scope_compatibility": compatibility,
            "coverage_status": coverage,
            "polarity": polarity,
            "final_relation": final_relation,
        },
    }


def _unit_alignments(unit: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    support = _support(unit)
    refute = _refute(unit)
    base = _frame(unit)
    rows = [
        _alignment_row(
            unit,
            "A01",
            "direct_support",
            support,
            base,
            _relations(),
            "COMPATIBLE",
            "COMPLETE",
            "SUPPORTS",
            "ENTAILED",
        ),
        _alignment_row(
            unit,
            "A02",
            "direct_refutation_value_conflict",
            refute,
            _frame(unit, target_value=unit["alternate_target_value"]),
            _relations(target_value="MISMATCH"),
            "COMPATIBLE",
            "COMPLETE",
            "REFUTES",
            "CONTRADICTED",
        ),
        _alignment_row(
            unit,
            "A03",
            "entity_mismatch",
            support.replace(unit["subject"].lower(), other["subject"].lower()),
            _frame(unit, entity_or_subject=other["subject"]),
            _relations(entity_or_subject="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _alignment_row(
            unit,
            "A04",
            "predicate_mismatch",
            support.replace(unit["predicate"], unit["alternate_predicate"]),
            _frame(unit, predicate_or_event=unit["alternate_predicate"]),
            _relations(predicate_or_event="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _alignment_row(
            unit,
            "A05",
            "target_slot_identity_mismatch",
            (
                f"During {unit['year']}, {unit['organization']} in the {unit['region']} "
                f"{unit['modality']} {unit['subject'].lower()} records in "
                f"{unit['service_line']}. The governing action is {unit['predicate']}, "
                f"and the {unit['alternate_slot']} is {unit['target_value']} business days "
                f"when {unit['condition']}."
            ),
            _frame(unit, target_slot_identity=unit["alternate_slot"]),
            _relations(target_slot_identity="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _alignment_row(
            unit,
            "A06",
            "temporal_scope_mismatch",
            support.replace(str(unit["year"]), str(unit["alternate_year"]), 1),
            _frame(unit, temporal_scope=unit["alternate_year"]),
            _relations(temporal_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _alignment_row(
            unit,
            "A07",
            "location_scope_mismatch",
            support.replace(unit["region"], unit["alternate_region"]),
            _frame(unit, location_scope=unit["alternate_region"]),
            _relations(location_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _alignment_row(
            unit,
            "A08",
            "organizational_scope_mismatch",
            support.replace(unit["organization"], unit["alternate_organization"]),
            _frame(unit, organizational_scope=unit["alternate_organization"]),
            _relations(organizational_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _alignment_row(
            unit,
            "A09",
            "conditional_scope_mismatch",
            support.replace(unit["condition"], unit["alternate_condition"]),
            _frame(unit, conditional_scope=unit["alternate_condition"]),
            _relations(conditional_scope="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        _alignment_row(
            unit,
            "A10",
            "modality_quantification_mismatch",
            support.replace(unit["modality"], unit["alternate_modality"]),
            _frame(unit, modality_or_quantification=unit["alternate_modality"]),
            _relations(modality_or_quantification="MISMATCH"),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
    ]

    def missing(slot: str, suffix: str, subtype: str, evidence: str) -> None:
        frame = _frame(unit)
        frame[slot] = None
        rows.append(
            _alignment_row(
                unit,
                suffix,
                subtype,
                evidence,
                frame,
                _relations(**{slot: "UNSPECIFIED"}),
                "COMPATIBLE",
                "INCOMPLETE",
                "UNRESOLVED",
                "UNKNOWN",
            )
        )

    missing(
        "target_value",
        "A11",
        "missing_target_value",
        support.replace(
            f"within {unit['target_value']} business days",
            "within the registered deadline",
        ),
    )
    missing(
        "temporal_scope",
        "A12",
        "missing_temporal_scope",
        support.replace(f"During {unit['year']}, ", ""),
    )
    missing(
        "conditional_scope",
        "A13",
        "missing_conditional_scope",
        support.replace(
            f" when {unit['condition']}",
            " under the applicable condition",
        ),
    )
    missing(
        "location_scope",
        "A14",
        "missing_location_scope",
        support.replace(f" in the {unit['region']}", ""),
    )
    missing(
        "organizational_scope",
        "A15",
        "missing_organizational_scope",
        support.replace(f"{unit['organization']} in", "The assigned desk in"),
    )
    missing(
        "modality_or_quantification",
        "A16",
        "missing_modality_quantification",
        support.replace(unit["modality"], "handles"),
    )

    alias_evidence = support.replace(unit["subject"].lower(), unit["subject_alias"])
    rows.append(
        _alignment_row(
            unit,
            "A17",
            "explicit_entity_alias_match",
            alias_evidence,
            _frame(unit, entity_or_subject=unit["subject_alias"]),
            _relations(),
            "COMPATIBLE",
            "COMPLETE",
            "SUPPORTS",
            "ENTAILED",
        )
    )
    paraphrase_evidence = support.replace(unit["predicate"], unit["predicate_paraphrase"])
    rows.append(
        _alignment_row(
            unit,
            "A18",
            "predicate_paraphrase_match",
            paraphrase_evidence,
            _frame(unit, predicate_or_event=unit["predicate_paraphrase"]),
            _relations(),
            "COMPATIBLE",
            "COMPLETE",
            "SUPPORTS",
            "ENTAILED",
        )
    )
    rows.append(
        _alignment_row(
            unit,
            "A19",
            "same_domain_near_miss",
            (
                f"During {unit['year']}, {unit['organization']} in the {unit['region']} "
                f"must archive every {unit['subject'].lower()} audit note when "
                f"{unit['condition']}."
            ),
            _frame(
                unit,
                predicate_or_event="archive",
                target_slot_identity="audit_note_retention",
                target_value=None,
            ),
            _relations(
                predicate_or_event="MISMATCH",
                target_slot_identity="MISMATCH",
                target_value="UNSPECIFIED",
            ),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        )
    )
    rows.append(
        _alignment_row(
            unit,
            "A20",
            "cross_unit_distractor",
            _support(other),
            _frame(other),
            _relations(
                entity_or_subject="MISMATCH",
                predicate_or_event=(
                    "MATCH" if other["predicate"] == unit["predicate"] else "MISMATCH"
                ),
                target_slot_identity=(
                    "MATCH"
                    if other["target_slot_identity"] == unit["target_slot_identity"]
                    else "MISMATCH"
                ),
                target_value=(
                    "MATCH" if other["target_value"] == unit["target_value"] else "MISMATCH"
                ),
                temporal_scope=("MATCH" if other["year"] == unit["year"] else "MISMATCH"),
                location_scope=("MATCH" if other["region"] == unit["region"] else "MISMATCH"),
                organizational_scope=(
                    "MATCH" if other["organization"] == unit["organization"] else "MISMATCH"
                ),
                conditional_scope=(
                    "MATCH" if other["condition"] == unit["condition"] else "MISMATCH"
                ),
                modality_or_quantification=(
                    "MATCH" if other["modality"] == unit["modality"] else "MISMATCH"
                ),
            ),
            "INCOMPATIBLE",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        )
    )
    return rows


def _group_row(
    unit: dict[str, Any],
    suffix: str,
    subtype: str,
    evidence_propositions: list[str],
    covered_slots: list[str],
    missing_slots: list[str],
    minimal_groups: list[list[int]],
    coherence: str,
    sufficiency: str,
    polarity: str,
    final_relation: str,
) -> dict[str, Any]:
    return {
        "group_id": f"{unit['unit_id']}-{suffix}",
        "split": "calibration",
        "unit_id": unit["unit_id"],
        "subtype": subtype,
        "claim": _claim_text(unit),
        "evidence_propositions": evidence_propositions,
        "gold": {
            "covered_decisive_slots": covered_slots,
            "missing_decisive_slots": missing_slots,
            "minimal_sufficient_groups_zero_based": minimal_groups,
            "cross_proposition_scope_coherence": coherence,
            "sufficiency": sufficiency,
            "polarity": polarity,
            "final_relation": final_relation,
        },
    }


def _unit_groups(unit: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    support = _support(unit)
    refute = _refute(unit)
    scope = (
        f"During {unit['year']}, {unit['organization']} in the {unit['region']} handles "
        f"{unit['subject'].lower()} records in {unit['service_line']} when "
        f"{unit['condition']}."
    )
    value = (
        f"The {unit['target_slot_identity']} for {unit['subject'].lower()} is "
        f"{unit['target_value']} business days and the governing action is "
        f"{unit['predicate']}."
    )
    modality = f"The governing rule {unit['modality']} covered records."
    missing_value = support.replace(
        f"within {unit['target_value']} business days",
        "within the registered deadline",
    )
    missing_condition = support.replace(
        f" when {unit['condition']}",
        " under the applicable condition",
    )
    distractor = _support(other)
    coherent_except_location = (
        f"During {unit['year']}, {unit['organization']} {unit['modality']} "
        f"{unit['subject'].lower()} records in {unit['service_line']} and "
        f"{unit['predicate']} them within {unit['target_value']} business days "
        f"when {unit['condition']}."
    )
    incoherent_location = f"The applicable location is {unit['alternate_region']}."
    support_duplicate = support.replace("records in", "records assigned to")
    different_condition_refute = refute.replace(
        unit["condition"],
        unit["alternate_condition"],
    )
    all_slots = list(ALL_SLOTS)
    return [
        _group_row(
            unit,
            "G01",
            "single_support",
            [support],
            all_slots,
            [],
            [[0]],
            "COHERENT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _group_row(
            unit,
            "G02",
            "single_refutation",
            [refute],
            all_slots,
            [],
            [[0]],
            "COHERENT",
            "SUFFICIENT",
            "REFUTES",
            "CONTRADICTED",
        ),
        _group_row(
            unit,
            "G03",
            "single_incomplete_missing_value",
            [missing_value],
            [slot for slot in all_slots if slot != "target_value"],
            ["target_value"],
            [],
            "COHERENT",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _group_row(
            unit,
            "G04",
            "single_incomplete_missing_condition",
            [missing_condition],
            [slot for slot in all_slots if slot != "conditional_scope"],
            ["conditional_scope"],
            [],
            "COHERENT",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _group_row(
            unit,
            "G05",
            "complementary_two_span_support",
            [scope, value + " " + modality],
            all_slots,
            [],
            [[0, 1]],
            "COHERENT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _group_row(
            unit,
            "G06",
            "complementary_three_span_support",
            [scope, value, modality],
            all_slots,
            [],
            [[0, 1, 2]],
            "COHERENT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _group_row(
            unit,
            "G07",
            "support_with_cross_unit_distractor",
            [distractor, support],
            all_slots,
            [],
            [[1]],
            "COHERENT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _group_row(
            unit,
            "G08",
            "multi_span_unresolved_condition",
            [value, missing_condition],
            [slot for slot in all_slots if slot != "conditional_scope"],
            ["conditional_scope"],
            [],
            "COHERENT",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        _group_row(
            unit,
            "G09",
            "same_scope_support_refute_conflict",
            [support, refute],
            all_slots,
            [],
            [[0], [1]],
            "COHERENT",
            "CONFLICTING",
            "CONFLICTING",
            "CONFLICTING_EVIDENCE",
        ),
        _group_row(
            unit,
            "G10",
            "different_condition_not_conflict",
            [support, different_condition_refute],
            all_slots,
            [],
            [[0]],
            "COHERENT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _group_row(
            unit,
            "G11",
            "redundant_support_minimality",
            [support, support_duplicate],
            all_slots,
            [],
            [[0], [1]],
            "COHERENT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        _group_row(
            unit,
            "G12",
            "cross_span_scope_incoherence",
            [coherent_except_location, incoherent_location],
            [slot for slot in all_slots if slot != "location_scope"],
            ["location_scope"],
            [],
            "INCOHERENT",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
    ]


def _claim_row(
    unit: dict[str, Any],
    suffix: str,
    category: str,
    group_relations: list[str],
    expected_verdict: str,
    deterministic_gate: str = "NONE",
) -> dict[str, Any]:
    return {
        "case_id": f"{unit['unit_id']}-{suffix}",
        "split": "calibration",
        "unit_id": unit["unit_id"],
        "category": category,
        "group_relations": group_relations,
        "deterministic_gate": deterministic_gate,
        "expected_verdict": expected_verdict,
    }


def _unit_claims(unit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _claim_row(unit, "C01", "single_supported", ["ENTAILED"], "SUPPORTED"),
        _claim_row(unit, "C02", "single_refuted", ["CONTRADICTED"], "UNSUPPORTED"),
        _claim_row(unit, "C03", "compatible_incomplete", ["UNKNOWN"], "UNSUPPORTED"),
        _claim_row(
            unit,
            "C04",
            "complementary_group_supported",
            ["ENTAILED"],
            "SUPPORTED",
        ),
        _claim_row(
            unit,
            "C05",
            "same_scope_conflict",
            ["CONFLICTING_EVIDENCE"],
            "CONFLICTING_EVIDENCE",
        ),
        _claim_row(unit, "C06", "scope_mismatch_only", ["UNKNOWN"], "UNSUPPORTED"),
        _claim_row(
            unit,
            "C07",
            "citation_invalid",
            ["ENTAILED"],
            "CITATION_INVALID",
            "CITATION_INVALID",
        ),
        _claim_row(
            unit,
            "C08",
            "stale_evidence",
            ["ENTAILED"],
            "STALE_EVIDENCE",
            "STALE_EVIDENCE",
        ),
        _claim_row(
            unit,
            "C09",
            "registered_conflict",
            ["ENTAILED"],
            "CONFLICTING_EVIDENCE",
            "REGISTERED_CONFLICT",
        ),
        _claim_row(
            unit,
            "C10",
            "unresolved_alignment_abstention",
            ["UNKNOWN"],
            "UNSUPPORTED",
        ),
    ]


def build_suite() -> dict[str, Any]:
    units = build_units()
    propositions: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        other = units[(index + 11) % len(units)]
        propositions.extend(_unit_propositions(unit, other))
        alignments.extend(_unit_alignments(unit, other))
        groups.extend(_unit_groups(unit, other))
        claims.extend(_unit_claims(unit))
    propositions.sort(key=lambda row: str(row["proposition_case_id"]))
    alignments.sort(key=lambda row: str(row["alignment_id"]))
    groups.sort(key=lambda row: str(row["group_id"]))
    claims.sort(key=lambda row: str(row["case_id"]))
    return {
        "protocol_id": PROTOCOL_ID,
        "corpus_id": CORPUS_ID,
        "units": units,
        "proposition_rows": propositions,
        "alignment_rows": alignments,
        "evidence_group_rows": groups,
        "claim_rows": claims,
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for row in rows
    )


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    names = sorted({str(row[key]) for row in rows})
    return {name: sum(row[key] == name for row in rows) for name in names}


def manifest() -> dict[str, Any]:
    suite = build_suite()
    units = suite["units"]
    propositions = suite["proposition_rows"]
    alignments = suite["alignment_rows"]
    groups = suite["evidence_group_rows"]
    claims = suite["claim_rows"]
    return {
        "protocol_id": PROTOCOL_ID,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "partition": "calibration_only",
        "counts": {
            "units": len(units),
            "proposition_rows": len(propositions),
            "alignment_rows": len(alignments),
            "evidence_group_rows": len(groups),
            "claim_rows": len(claims),
            "proposition_subtypes": _counts(propositions, "subtype"),
            "alignment_subtypes": _counts(alignments, "subtype"),
            "evidence_group_subtypes": _counts(groups, "subtype"),
            "claim_categories": _counts(claims, "category"),
            "alignment_compatibility": {
                label: sum(row["gold"]["scope_compatibility"] == label for row in alignments)
                for label in ("COMPATIBLE", "INCOMPATIBLE")
            },
            "alignment_relations": {
                label: sum(row["gold"]["final_relation"] == label for row in alignments)
                for label in ("ENTAILED", "CONTRADICTED", "UNKNOWN")
            },
            "group_relations": {
                label: sum(row["gold"]["final_relation"] == label for row in groups)
                for label in (
                    "ENTAILED",
                    "CONTRADICTED",
                    "UNKNOWN",
                    "CONFLICTING_EVIDENCE",
                )
            },
        },
        "sha256": {
            "units": hashlib.sha256(_jsonl(units)).hexdigest(),
            "proposition_rows": hashlib.sha256(_jsonl(propositions)).hexdigest(),
            "alignment_rows": hashlib.sha256(_jsonl(alignments)).hexdigest(),
            "evidence_group_rows": hashlib.sha256(_jsonl(groups)).hexdigest(),
            "claim_rows": hashlib.sha256(_jsonl(claims)).hexdigest(),
        },
        "governance": {
            "candidate_model_calls": 0,
            "model_bindings": 0,
            "model_family_comparisons": 0,
            "prompt_searches": 0,
            "threshold_searches": 0,
            "calibration_fits": 0,
            "a45a_fresh_validation_rows_materialized": 0,
            "a45a_fresh_validation_rows_scored": 0,
            "confirmatory_query_records_inspected": 0,
            "confirmatory_queries_scored": 0,
            "a45bm2_m3_rows_reused": 0,
            "future_validation_rows_constructed": 0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))

"""Candidate-independent AERF validity construction for Phase 4 A4.5a.

This module performs no learned inference. It builds a fresh fictional auxiliary
support corpus and gold AERF component/claim cases before any model binding.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEED = 20260821
CORPUS_ID = "helix-aerf-validity-corpus-v1"
PROTOCOL_ID = "phase4-assistance-a4.5a-aerf-validity-v1"
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
WINDOWS = (1, 2, 3, 4, 5, 7)


def _order(unit_ids: list[str]) -> list[str]:
    return sorted(
        unit_ids,
        key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest(),
    )


def build_units() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for index in range(60):
        number = index + 1
        unit_id = f"AERF-U{number:03d}"
        queue = QUEUES[index % len(QUEUES)]
        requirement = REQUIREMENTS[(index * 5 + 1) % len(REQUIREMENTS)]
        window = WINDOWS[(index * 7 + 2) % len(WINDOWS)]
        alt_queue = QUEUES[(index + 1) % len(QUEUES)]
        subject = f"Orchid case {number:03d}"
        support_queue = f"{subject} requests are handled by the {queue} queue."
        support_requirement = f"{subject} review requires {requirement}."
        support_window = (
            f"The standard review window for {subject.lower()} is {window} business days."
        )
        units.append(
            {
                "unit_id": unit_id,
                "subject": subject,
                "queue": queue,
                "alternate_queue": alt_queue,
                "requirement": requirement,
                "window_days": window,
                "documents": {
                    "queue": {"document_id": f"{unit_id}-Q", "text": support_queue},
                    "requirement": {
                        "document_id": f"{unit_id}-R",
                        "text": support_requirement,
                    },
                    "window": {"document_id": f"{unit_id}-W", "text": support_window},
                    "refutation": {
                        "document_id": f"{unit_id}-X",
                        "text": (
                            f"{subject} requests are not handled by the {queue} queue. "
                            f"They are handled by the {alt_queue} queue."
                        ),
                    },
                    "contaminated": {
                        "document_id": f"{unit_id}-C",
                        "text": (
                            f"{support_queue} {subject} requests are not automatically "
                            "approved without review."
                        ),
                    },
                },
            }
        )
    return units


def split_units(units: list[dict[str, Any]]) -> dict[str, list[str]]:
    ordered = _order([str(unit["unit_id"]) for unit in units])
    calibration = ordered[:40]
    validation = ordered[40:]
    if len(calibration) != 40 or len(validation) != 20:
        raise RuntimeError("invalid A4.5a split")
    if set(calibration) & set(validation):
        raise RuntimeError("A4.5a calibration and validation units overlap")
    return {"calibration": calibration, "validation": validation}


def _pair(
    pair_id: str,
    split: str,
    unit_id: str,
    subtype: str,
    claim: str,
    document: dict[str, Any],
    relevance: str,
    sufficiency: str,
    polarity: str,
    final_relation: str,
) -> dict[str, Any]:
    evidence_text = str(document["text"])
    minimal = None
    if relevance != "IRRELEVANT":
        minimal = evidence_text.split(". ", 1)[0].rstrip(".") + "."
    return {
        "pair_id": pair_id,
        "split": split,
        "unit_id": unit_id,
        "subtype": subtype,
        "claim": claim,
        "evidence_document_id": str(document["document_id"]),
        "evidence_text": evidence_text,
        "gold": {
            "relevance": relevance,
            "sufficiency": sufficiency,
            "polarity": polarity,
            "final_relation": final_relation,
            "minimal_evidence_text": minimal,
        },
    }


def _claim(
    case_id: str,
    split: str,
    unit_id: str,
    category: str,
    atom_relations: list[str],
    expected_verdict: str,
    deterministic_gate: str = "NONE",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "split": split,
        "unit_id": unit_id,
        "category": category,
        "atom_relations": atom_relations,
        "deterministic_gate": deterministic_gate,
        "expected_verdict": expected_verdict,
    }


def _unit_pairs(
    split_name: str,
    unit_id: str,
    unit: dict[str, Any],
    other_queue: dict[str, Any],
) -> list[dict[str, Any]]:
    documents = unit["documents"]
    subject = str(unit["subject"])
    queue = str(unit["queue"])
    requirement = str(unit["requirement"])
    window = int(unit["window_days"])
    prefix = f"{split_name[:3].upper()}-{unit_id}"
    specs = [
        (
            "P01",
            "literal_support",
            f"{subject} requests are handled by the {queue} queue.",
            documents["queue"],
            "RELEVANT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        (
            "P02",
            "paraphrase_support",
            f"The {queue} queue handles {subject.lower()} requests.",
            documents["queue"],
            "RELEVANT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
        (
            "P03",
            "explicit_refutation",
            f"{subject} requests are handled by the {queue} queue.",
            documents["refutation"],
            "RELEVANT",
            "SUFFICIENT",
            "REFUTES",
            "CONTRADICTED",
        ),
        (
            "P04",
            "attribute_refutation",
            f"{subject} requests are not handled by the {queue} queue.",
            documents["queue"],
            "RELEVANT",
            "SUFFICIENT",
            "REFUTES",
            "CONTRADICTED",
        ),
        (
            "P05",
            "cross_document_irrelevance",
            f"{subject} requests are handled by the {queue} queue.",
            other_queue,
            "IRRELEVANT",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        (
            "P06",
            "same_domain_irrelevance",
            f"{subject} review requires {requirement}.",
            documents["window"],
            "IRRELEVANT",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "UNKNOWN",
        ),
        (
            "P07",
            "relevant_but_insufficient",
            (
                f"{subject} requests are handled by the {queue} queue and review "
                f"requires {requirement}."
            ),
            documents["queue"],
            "RELEVANT",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        (
            "P08",
            "temporal_insufficiency",
            f"{subject} review always finishes within {window} business days.",
            documents["window"],
            "RELEVANT",
            "INSUFFICIENT",
            "UNRESOLVED",
            "UNKNOWN",
        ),
        (
            "P09",
            "context_contamination_support",
            f"{subject} requests are handled by the {queue} queue.",
            documents["contaminated"],
            "RELEVANT",
            "SUFFICIENT",
            "SUPPORTS",
            "ENTAILED",
        ),
    ]
    return [
        _pair(f"{prefix}-{suffix}", split_name, unit_id, subtype, claim, *rest)
        for suffix, subtype, claim, *rest in specs
    ]


def _unit_claims(split_name: str, unit_id: str) -> list[dict[str, Any]]:
    prefix = f"{split_name[:3].upper()}-{unit_id}"
    specs = [
        ("C01", "single_supported", ["ENTAILED"], "SUPPORTED", "NONE"),
        ("C02", "single_refuted", ["CONTRADICTED"], "UNSUPPORTED", "NONE"),
        ("C03", "single_unknown", ["UNKNOWN"], "UNSUPPORTED", "NONE"),
        ("C04", "multi_document_supported", ["ENTAILED", "ENTAILED"], "SUPPORTED", "NONE"),
        (
            "C05",
            "partial_multi_document_unsupported",
            ["ENTAILED", "UNKNOWN"],
            "UNSUPPORTED",
            "NONE",
        ),
        (
            "C06",
            "support_refute_conflict",
            ["ENTAILED", "CONTRADICTED"],
            "CONFLICTING_EVIDENCE",
            "NONE",
        ),
        ("C07", "citation_invalid", ["ENTAILED"], "CITATION_INVALID", "CITATION_INVALID"),
        ("C08", "stale_evidence", ["ENTAILED"], "STALE_EVIDENCE", "STALE_EVIDENCE"),
        (
            "C09",
            "registered_conflict",
            ["ENTAILED"],
            "CONFLICTING_EVIDENCE",
            "REGISTERED_CONFLICT",
        ),
    ]
    return [
        _claim(
            f"{prefix}-{suffix}",
            split_name,
            unit_id,
            category,
            relations,
            verdict,
            gate,
        )
        for suffix, category, relations, verdict, gate in specs
    ]


def build_suite() -> dict[str, Any]:
    units = build_units()
    by_id = {str(unit["unit_id"]): unit for unit in units}
    split = split_units(units)
    pair_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for split_name in ("calibration", "validation"):
        ids = split[split_name]
        for offset, unit_id in enumerate(ids):
            unit = by_id[unit_id]
            other = by_id[ids[(offset + 1) % len(ids)]]
            pair_rows.extend(
                _unit_pairs(split_name, unit_id, unit, other["documents"]["queue"])
            )
            claim_rows.extend(_unit_claims(split_name, unit_id))
    pair_rows.sort(key=lambda row: str(row["pair_id"]))
    claim_rows.sort(key=lambda row: str(row["case_id"]))
    return {
        "protocol_id": PROTOCOL_ID,
        "corpus_id": CORPUS_ID,
        "split": split,
        "units": units,
        "pair_rows": pair_rows,
        "claim_rows": claim_rows,
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def manifest() -> dict[str, Any]:
    suite = build_suite()
    pairs = suite["pair_rows"]
    claims = suite["claim_rows"]
    units = suite["units"]
    relation_counts = {
        split_name: {
            relation: sum(
                row["split"] == split_name and row["gold"]["final_relation"] == relation
                for row in pairs
            )
            for relation in ("ENTAILED", "CONTRADICTED", "UNKNOWN")
        }
        for split_name in ("calibration", "validation")
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "counts": {
            "units": len(units),
            "calibration_units": 40,
            "validation_units": 20,
            "pair_rows": len(pairs),
            "pair_rows_by_split": {
                split_name: sum(row["split"] == split_name for row in pairs)
                for split_name in ("calibration", "validation")
            },
            "claim_rows": len(claims),
            "claim_rows_by_split": {
                split_name: sum(row["split"] == split_name for row in claims)
                for split_name in ("calibration", "validation")
            },
            "relation_counts_by_split": relation_counts,
        },
        "sha256": {
            "units": hashlib.sha256(_jsonl(units)).hexdigest(),
            "pair_rows": hashlib.sha256(_jsonl(pairs)).hexdigest(),
            "claim_rows": hashlib.sha256(_jsonl(claims)).hexdigest(),
            "calibration_pairs": hashlib.sha256(
                _jsonl([row for row in pairs if row["split"] == "calibration"])
            ).hexdigest(),
            "validation_pairs": hashlib.sha256(
                _jsonl([row for row in pairs if row["split"] == "validation"])
            ).hexdigest(),
            "calibration_claims": hashlib.sha256(
                _jsonl([row for row in claims if row["split"] == "calibration"])
            ).hexdigest(),
            "validation_claims": hashlib.sha256(
                _jsonl([row for row in claims if row["split"] == "validation"])
            ).hexdigest(),
        },
        "candidate_model_calls": 0,
        "confirmatory_query_records_inspected": 0,
        "a44d_rows_reused": 0,
    }


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))

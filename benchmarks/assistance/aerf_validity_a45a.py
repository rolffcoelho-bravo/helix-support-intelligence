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
                    "queue": {
                        "document_id": f"{unit_id}-Q",
                        "text": support_queue,
                    },
                    "requirement": {
                        "document_id": f"{unit_id}-R",
                        "text": support_requirement,
                    },
                    "window": {
                        "document_id": f"{unit_id}-W",
                        "text": support_window,
                    },
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
    evidence_document_id: str,
    evidence_text: str,
    relevance: str,
    sufficiency: str,
    polarity: str,
    final_relation: str,
) -> dict[str, Any]:
    minimal = None
    if relevance != "IRRELEVANT":
        minimal = evidence_text.split(". ", 1)[0].rstrip(".") + "."
    return {
        "pair_id": pair_id,
        "split": split,
        "unit_id": unit_id,
        "subtype": subtype,
        "claim": claim,
        "evidence_document_id": evidence_document_id,
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
            documents = unit["documents"]
            subject = str(unit["subject"])
            queue = str(unit["queue"])
            requirement = str(unit["requirement"])
            window = int(unit["window_days"])
            other = by_id[ids[(offset + 1) % len(ids)]]
            other_doc = other["documents"]["queue"]
            prefix = f"{split_name[:3].upper()}-{unit_id}"

            pair_rows.extend(
                [
                    _pair(
                        f"{prefix}-P01",
                        split_name,
                        unit_id,
                        "literal_support",
                        f"{subject} requests are handled by the {queue} queue.",
                        str(documents["queue"]["document_id"]),
                        str(documents["queue"]["text"]),
                        "RELEVANT",
                        "SUFFICIENT",
                        "SUPPORTS",
                        "ENTAILED",
                    ),
                    _pair(
                        f"{prefix}-P02",
                        split_name,
                        unit_id,
                        "paraphrase_support",
                        f"The {queue} queue handles {subject.lower()} requests.",
                        str(documents["queue"]["document_id"]),
                        str(documents["queue"]["text"]),
                        "RELEVANT",
                        "SUFFICIENT",
                        "SUPPORTS",
                        "ENTAILED",
                    ),
                    _pair(
                        f"{prefix}-P03",
                        split_name,
                        unit_id,
                        "explicit_refutation",
                        f"{subject} requests are handled by the {queue} queue.",
                        str(documents["refutation"]["document_id"]),
                        str(documents["refutation"]["text"]),
                        "RELEVANT",
                        "SUFFICIENT",
                        "REFUTES",
                        "CONTRADICTED",
                    ),
                    _pair(
                        f"{prefix}-P04",
                        split_name,
                        unit_id,
                        "attribute_refutation",
                        f"{subject} requests are not handled by the {queue} queue.",
                        str(documents["queue"]["document_id"]),
                        str(documents["queue"]["text"]),
                        "RELEVANT",
                        "SUFFICIENT",
                        "REFUTES",
                        "CONTRADICTED",
                    ),
                    _pair(
                        f"{prefix}-P05",
                        split_name,
                        unit_id,
                        "cross_document_irrelevance",
                        f"{subject} requests are handled by the {queue} queue.",
                        str(other_doc["document_id"]),
                        str(other_doc["text"]),
                        "IRRELEVANT",
                        "NOT_APPLICABLE",
                        "NOT_APPLICABLE",
                        "UNKNOWN",
                    ),
                    _pair(
                        f"{prefix}-P06",
                        split_name,
                        unit_id,
                        "same_domain_irrelevance",
                        f"{subject} review requires {requirement}.",
                        str(documents["window"]["document_id"]),
                        str(documents["window"]["text"]),
                        "IRRELEVANT",
                        "NOT_APPLICABLE",
                        "NOT_APPLICABLE",
                        "UNKNOWN",
                    ),
                    _pair(
                        f"{prefix}-P07",
                        split_name,
                        unit_id,
                        "relevant_but_insufficient",
                        (
                            f"{subject} requests are handled by the {queue} queue and review "
                            f"requires {requirement}."
                        ),
                        str(documents["queue"]["document_id"]),
                        str(documents["queue"]["text"]),
                        "RELEVANT",
                        "INSUFFICIENT",
                        "UNRESOLVED",
                        "UNKNOWN",
                    ),
                    _pair(
                        f"{prefix}-P08",
                        split_name,
                        unit_id,
                        "temporal_insufficiency",
                        f"{subject} review always finishes within {window} business days.",
                        str(documents["window"]["document_id"]),
                        str(documents["window"]["text"]),
                        "RELEVANT",
                        "INSUFFICIENT",
                        "UNRESOLVED",
                        "UNKNOWN",
                    ),
                    _pair(
                        f"{prefix}-P09",
                        split_name,
                        unit_id,
                        "context_contamination_support",
                        f"{subject} requests are handled by the {queue} queue.",
                        str(documents["contaminated"]["document_id"]),
                        str(documents["contaminated"]["text"]),
                        "RELEVANT",
                        "SUFFICIENT",
                        "SUPPORTS",
                        "ENTAILED",
                    ),
                ]
            )

            claim_rows.extend(
                [
                    _claim(
                        f"{prefix}-C01",
                        split_name,
                        unit_id,
                        "single_supported",
                        ["ENTAILED"],
                        "SUPPORTED",
                    ),
                    _claim(
                        f"{prefix}-C02",
                        split_name,
                        unit_id,
                        "single_refuted",
                        ["CONTRADICTED"],
                        "UNSUPPORTED",
                    ),
                    _claim(
                        f"{prefix}-C03",
                        split_name,
                        unit_id,
                        "single_unknown",
                        ["UNKNOWN"],
                        "UNSUPPORTED",
                    ),
                    _claim(
                        f"{prefix}-C04",
                        split_name,
                        unit_id,
                        "multi_document_supported",
                        ["ENTAILED", "ENTAILED"],
                        "SUPPORTED",
                    ),
                    _claim(
                        f"{prefix}-C05",
                        split_name,
                        unit_id,
                        "partial_multi_document_unsupported",
                        ["ENTAILED", "UNKNOWN"],
                        "UNSUPPORTED",
                    ),
                    _claim(
                        f"{prefix}-C06",
                        split_name,
                        unit_id,
                        "support_refute_conflict",
                        ["ENTAILED", "CONTRADICTED"],
                        "CONFLICTING_EVIDENCE",
                    ),
                    _claim(
                        f"{prefix}-C07",
                        split_name,
                        unit_id,
                        "citation_invalid",
                        ["ENTAILED"],
                        "CITATION_INVALID",
                        "CITATION_INVALID",
                    ),
                    _claim(
                        f"{prefix}-C08",
                        split_name,
                        unit_id,
                        "stale_evidence",
                        ["ENTAILED"],
                        "STALE_EVIDENCE",
                        "STALE_EVIDENCE",
                    ),
                    _claim(
                        f"{prefix}-C09",
                        split_name,
                        unit_id,
                        "registered_conflict",
                        ["ENTAILED"],
                        "CONFLICTING_EVIDENCE",
                        "REGISTERED_CONFLICT",
                    ),
                ]
            )

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
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
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
                row["split"] == split_name
                and row["gold"]["final_relation"] == relation
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

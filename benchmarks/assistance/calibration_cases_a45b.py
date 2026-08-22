"""Calibration-only AERF materializer for Phase 4 A4.5b.

This module reconstructs only the 40 A4.5a calibration units. It deliberately
contains no fresh-validation unit identifiers and performs no learned inference.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from benchmarks.assistance.aerf_validity_a45a import _unit_claims, _unit_pairs

CALIBRATION_UNIT_IDS = (
    "AERF-U052",
    "AERF-U033",
    "AERF-U058",
    "AERF-U016",
    "AERF-U025",
    "AERF-U045",
    "AERF-U003",
    "AERF-U017",
    "AERF-U057",
    "AERF-U010",
    "AERF-U050",
    "AERF-U046",
    "AERF-U041",
    "AERF-U038",
    "AERF-U013",
    "AERF-U026",
    "AERF-U047",
    "AERF-U053",
    "AERF-U043",
    "AERF-U037",
    "AERF-U051",
    "AERF-U008",
    "AERF-U048",
    "AERF-U014",
    "AERF-U009",
    "AERF-U028",
    "AERF-U005",
    "AERF-U002",
    "AERF-U022",
    "AERF-U034",
    "AERF-U060",
    "AERF-U015",
    "AERF-U029",
    "AERF-U042",
    "AERF-U044",
    "AERF-U049",
    "AERF-U054",
    "AERF-U011",
    "AERF-U027",
    "AERF-U007",
)

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

EXPECTED_PAIR_SHA256 = "2339b1328c1dd2854e712862f5e4183300581997a22a7000db2629313a49092f"
EXPECTED_CLAIM_SHA256 = "e04813b1aec29335ffdbfbd8baa2e6f021ca6aa6a64625c255b454139e948ee5"


def _build_unit(unit_id: str) -> dict[str, Any]:
    number = int(unit_id.removeprefix("AERF-U"))
    index = number - 1
    queue = QUEUES[index % len(QUEUES)]
    requirement = REQUIREMENTS[(index * 5 + 1) % len(REQUIREMENTS)]
    window = WINDOWS[(index * 7 + 2) % len(WINDOWS)]
    alternate_queue = QUEUES[(index + 1) % len(QUEUES)]
    subject = f"Orchid case {number:03d}"
    support_queue = f"{subject} requests are handled by the {queue} queue."
    support_requirement = f"{subject} review requires {requirement}."
    support_window = f"The standard review window for {subject.lower()} is {window} business days."
    return {
        "unit_id": unit_id,
        "subject": subject,
        "queue": queue,
        "alternate_queue": alternate_queue,
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
                    f"They are handled by the {alternate_queue} queue."
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


def build_calibration_only() -> dict[str, Any]:
    units = {unit_id: _build_unit(unit_id) for unit_id in CALIBRATION_UNIT_IDS}
    pair_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for offset, unit_id in enumerate(CALIBRATION_UNIT_IDS):
        unit = units[unit_id]
        other_id = CALIBRATION_UNIT_IDS[(offset + 1) % len(CALIBRATION_UNIT_IDS)]
        other_queue = units[other_id]["documents"]["queue"]
        pair_rows.extend(_unit_pairs("calibration", unit_id, unit, other_queue))
        claim_rows.extend(_unit_claims("calibration", unit_id))
    pair_rows.sort(key=lambda row: str(row["pair_id"]))
    claim_rows.sort(key=lambda row: str(row["case_id"]))
    return {
        "unit_ids": list(CALIBRATION_UNIT_IDS),
        "pair_rows": pair_rows,
        "claim_rows": claim_rows,
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def calibration_manifest() -> dict[str, Any]:
    materialized = build_calibration_only()
    pairs = materialized["pair_rows"]
    claims = materialized["claim_rows"]
    pair_hash = hashlib.sha256(_jsonl(pairs)).hexdigest()
    claim_hash = hashlib.sha256(_jsonl(claims)).hexdigest()
    if pair_hash != EXPECTED_PAIR_SHA256:
        raise RuntimeError("A4.5b calibration-pair reconstruction drifted")
    if claim_hash != EXPECTED_CLAIM_SHA256:
        raise RuntimeError("A4.5b calibration-claim reconstruction drifted")
    return {
        "calibration_units": len(CALIBRATION_UNIT_IDS),
        "calibration_pairs": len(pairs),
        "calibration_claims": len(claims),
        "calibration_pairs_sha256": pair_hash,
        "calibration_claims_sha256": claim_hash,
        "validation_units_materialized": 0,
        "validation_pairs_materialized": 0,
        "validation_claims_materialized": 0,
        "confirmatory_queries_inspected": 0,
    }


if __name__ == "__main__":
    print(json.dumps(calibration_manifest(), indent=2, sort_keys=True))

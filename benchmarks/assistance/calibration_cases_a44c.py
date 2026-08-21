"""Materialize only the frozen A4.4a calibration cases for A4.4c."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compositional_cases_a44a import (  # noqa: E402
    SPLIT_SEED,
    _atom,
    _case,
    _documents_by_intent,
    _first_sentence,
    _ordered,
    _sentence,
)
from grounding_anchors_a43a import development_intents  # noqa: E402

from helix_support_intelligence.data.helixbank import (  # noqa: E402
    CorpusBundle,
    generate_bundle,
)


def calibration_intents(bundle: CorpusBundle) -> set[str]:
    """Reconstruct only the registered 40-intent A4.4a calibration partition."""
    development = development_intents(bundle)
    archived = {
        str(row["intent"])
        for row in bundle.documents
        if row["kind"] == "faq" and row["status"] == "archived"
    } & development
    conflicts = {
        str(row["intent"]) for row in bundle.documents if bool(row["conflict_fixture"])
    } & development
    ordinary = development - archived - conflicts
    prefix = f"{SPLIT_SEED}:a44a"
    calibration = (
        set(_ordered(conflicts, prefix)[:3])
        | set(_ordered(archived, prefix)[:5])
        | set(_ordered(ordinary, prefix)[:32])
    )
    if len(calibration) != 40:
        raise RuntimeError(
            f"A4.4c must reconstruct exactly 40 calibration intents, got {len(calibration)}."
        )
    return calibration


def generate_calibration_cases(bundle: CorpusBundle | None = None) -> list[dict[str, Any]]:
    """Build the registered calibration cases without materializing validation cases."""
    bundle = bundle or generate_bundle()
    docs = _documents_by_intent(bundle)
    intents = sorted(calibration_intents(bundle))
    cases: list[dict[str, Any]] = []

    for index, intent in enumerate(intents):
        partner = intents[(index + 1) % len(intents)]
        policy = docs[intent]["policy"]
        faq = docs[intent]["faq"]
        partner_policy = docs[partner]["policy"]

        policy_id = str(policy["document_id"])
        faq_id = str(faq["document_id"])
        partner_policy_id = str(partner_policy["document_id"])
        title = str(policy["title"])
        queue = str(policy["queue"])
        partner_title = str(partner_policy["title"])
        queue_sentence = _sentence(str(policy["body"]), "Requests are handled by")
        partner_guidance = _sentence(str(partner_policy["body"]), "HelixBank guidance for")
        stem = f"A44A-CAL-{index + 1:02d}"

        cases.extend(
            [
                _case(
                    case_id=f"{stem}-LIT",
                    split="calibration",
                    intent=intent,
                    category="literal_supported",
                    presented_document_ids=[policy_id],
                    cited_document_ids=[policy_id],
                    atoms=[_atom("A1", queue_sentence, entailed_by=[policy_id])],
                    expected_verdict="SUPPORTED",
                ),
                _case(
                    case_id=f"{stem}-PAR",
                    split="calibration",
                    intent=intent,
                    category="paraphrase_supported",
                    presented_document_ids=[policy_id],
                    cited_document_ids=[policy_id],
                    atoms=[
                        _atom(
                            "A1",
                            f"{title} requests go to the {queue} queue.",
                            entailed_by=[policy_id],
                        )
                    ],
                    expected_verdict="SUPPORTED",
                ),
                _case(
                    case_id=f"{stem}-CON",
                    split="calibration",
                    intent=intent,
                    category="contradiction_unsupported",
                    presented_document_ids=[policy_id],
                    cited_document_ids=[policy_id],
                    atoms=[
                        _atom(
                            "A1",
                            f"{title} requests are not handled by the {queue} queue.",
                            contradicted_by=[policy_id],
                        )
                    ],
                    expected_verdict="UNSUPPORTED",
                ),
                _case(
                    case_id=f"{stem}-UNS",
                    split="calibration",
                    intent=intent,
                    category="unsupported_approval",
                    presented_document_ids=[policy_id],
                    cited_document_ids=[policy_id],
                    atoms=[_atom("A1", f"{title} is automatically approved without review.")],
                    expected_verdict="UNSUPPORTED",
                ),
                _case(
                    case_id=f"{stem}-CIT",
                    split="calibration",
                    intent=intent,
                    category="citation_invalid",
                    presented_document_ids=[policy_id],
                    cited_document_ids=[partner_policy_id],
                    atoms=[_atom("A1", queue_sentence, entailed_by=[policy_id])],
                    expected_verdict="CITATION_INVALID",
                ),
                _case(
                    case_id=f"{stem}-MUL",
                    split="calibration",
                    intent=intent,
                    category="multi_document_supported",
                    presented_document_ids=[policy_id, partner_policy_id],
                    cited_document_ids=[policy_id, partner_policy_id],
                    atoms=[
                        _atom("A1", queue_sentence, entailed_by=[policy_id]),
                        _atom("A2", partner_guidance, entailed_by=[partner_policy_id]),
                    ],
                    expected_verdict="SUPPORTED",
                ),
                _case(
                    case_id=f"{stem}-PARMUL",
                    split="calibration",
                    intent=intent,
                    category="partial_multi_document_unsupported",
                    presented_document_ids=[policy_id, partner_policy_id],
                    cited_document_ids=[policy_id, partner_policy_id],
                    atoms=[
                        _atom("A1", queue_sentence, entailed_by=[policy_id]),
                        _atom(
                            "A2",
                            f"{partner_title} is automatically approved without review.",
                        ),
                    ],
                    expected_verdict="UNSUPPORTED",
                ),
            ]
        )

        if faq["status"] == "archived":
            cases.append(
                _case(
                    case_id=f"{stem}-STA",
                    split="calibration",
                    intent=intent,
                    category="stale_current_evidence",
                    presented_document_ids=[faq_id],
                    cited_document_ids=[faq_id],
                    atoms=[
                        _atom(
                            "A1",
                            _first_sentence(str(faq["body"])),
                            entailed_by=[faq_id],
                        )
                    ],
                    expected_verdict="STALE_EVIDENCE",
                )
            )

        if bool(faq["conflict_fixture"]):
            cases.append(
                _case(
                    case_id=f"{stem}-CFU",
                    split="calibration",
                    intent=intent,
                    category="unresolved_conflict",
                    presented_document_ids=[policy_id, faq_id],
                    cited_document_ids=[policy_id, faq_id],
                    atoms=[
                        _atom(
                            "A1",
                            queue_sentence,
                            entailed_by=[policy_id, faq_id],
                        )
                    ],
                    expected_verdict="CONFLICTING_EVIDENCE",
                )
            )

    cases.sort(key=lambda row: str(row["case_id"]))
    if len(cases) != 288:
        raise RuntimeError(f"A4.4c calibration suite must contain 288 rows, got {len(cases)}.")
    if any(row["split"] != "calibration" for row in cases):
        raise RuntimeError("A4.4c calibration materializer emitted a non-calibration row.")
    return cases


def canonical_calibration_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize calibration rows canonically for execution provenance."""
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def calibration_summary() -> dict[str, Any]:
    """Return calibration-only counts and hash without constructing validation cases."""
    rows = generate_calibration_cases()
    return {
        "rows": len(rows),
        "intents": len({str(row["intent"]) for row in rows}),
        "sha256": hashlib.sha256(canonical_calibration_jsonl_bytes(rows)).hexdigest(),
        "validation_cases_materialized": 0,
    }


if __name__ == "__main__":
    print(json.dumps(calibration_summary(), indent=2, sort_keys=True))

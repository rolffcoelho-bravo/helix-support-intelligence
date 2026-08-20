"""Deterministic candidate-independent compositional grounding cases for A4.4a."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from grounding_anchors_a43a import development_intents  # noqa: E402
from helix_support_intelligence.data.helixbank import (  # noqa: E402
    CorpusBundle,
    generate_bundle,
)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SPLIT_SEED = 20260820


def _ordered(values: set[str], prefix: str) -> list[str]:
    return sorted(
        values,
        key=lambda intent: hashlib.sha256(f"{prefix}:{intent}".encode()).hexdigest(),
    )


def compositional_partition(bundle: CorpusBundle) -> dict[str, set[str]]:
    """Split only frozen A4.0 development intents for A4.4a validation."""
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
    validation = development - calibration
    if len(calibration) != 40 or len(validation) != 20:
        raise RuntimeError(
            "A4.4a compositional partition must be 40 calibration and 20 validation intents."
        )
    return {"calibration": calibration, "validation": validation}


def _sentence(body: str, prefix: str) -> str:
    for sentence in SENTENCE_RE.split(body.strip()):
        if sentence.startswith(prefix):
            return sentence.strip()
    raise RuntimeError(f"Missing frozen sentence prefix: {prefix}")


def _first_sentence(body: str) -> str:
    sentences = SENTENCE_RE.split(body.strip())
    if not sentences or not sentences[0].strip():
        raise RuntimeError("Frozen document body must contain a sentence.")
    return sentences[0].strip()


def _documents_by_intent(bundle: CorpusBundle) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for row in bundle.documents:
        intent = str(row["intent"])
        result.setdefault(intent, {})[str(row["kind"])] = dict(row)
    return result


def _atom(
    atom_id: str,
    text: str,
    *,
    entailed_by: list[str] | None = None,
    contradicted_by: list[str] | None = None,
) -> dict[str, object]:
    return {
        "atom_id": atom_id,
        "text": text,
        "entailed_by": sorted(entailed_by or []),
        "contradicted_by": sorted(contradicted_by or []),
    }


def _case(
    *,
    case_id: str,
    split: str,
    intent: str,
    category: str,
    presented_document_ids: list[str],
    cited_document_ids: list[str],
    atoms: list[dict[str, object]],
    expected_verdict: str,
    requires_current_evidence: bool = True,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "split": split,
        "intent": intent,
        "category": category,
        "presented_document_ids": sorted(presented_document_ids),
        "cited_document_ids": sorted(cited_document_ids),
        "requires_current_evidence": requires_current_evidence,
        "atoms": atoms,
        "expected_verdict": expected_verdict,
    }


def generate_cases(bundle: CorpusBundle | None = None) -> list[dict[str, Any]]:
    """Materialize the frozen A4.4a compositional grounding validation suite."""
    bundle = bundle or generate_bundle()
    docs = _documents_by_intent(bundle)
    partition = compositional_partition(bundle)
    cases: list[dict[str, Any]] = []

    for split in ("calibration", "validation"):
        intents = sorted(partition[split])
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
            partner_guidance = _sentence(
                str(partner_policy["body"]),
                "HelixBank guidance for",
            )
            stem = f"A44A-{split[:3].upper()}-{index + 1:02d}"

            cases.extend(
                [
                    _case(
                        case_id=f"{stem}-LIT",
                        split=split,
                        intent=intent,
                        category="literal_supported",
                        presented_document_ids=[policy_id],
                        cited_document_ids=[policy_id],
                        atoms=[_atom("A1", queue_sentence, entailed_by=[policy_id])],
                        expected_verdict="SUPPORTED",
                    ),
                    _case(
                        case_id=f"{stem}-PAR",
                        split=split,
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
                        split=split,
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
                        split=split,
                        intent=intent,
                        category="unsupported_approval",
                        presented_document_ids=[policy_id],
                        cited_document_ids=[policy_id],
                        atoms=[_atom("A1", f"{title} is automatically approved without review.")],
                        expected_verdict="UNSUPPORTED",
                    ),
                    _case(
                        case_id=f"{stem}-CIT",
                        split=split,
                        intent=intent,
                        category="citation_invalid",
                        presented_document_ids=[policy_id],
                        cited_document_ids=[partner_policy_id],
                        atoms=[_atom("A1", queue_sentence, entailed_by=[policy_id])],
                        expected_verdict="CITATION_INVALID",
                    ),
                    _case(
                        case_id=f"{stem}-MUL",
                        split=split,
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
                        split=split,
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
                        split=split,
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
                        split=split,
                        intent=intent,
                        category="unresolved_conflict",
                        presented_document_ids=[policy_id, faq_id],
                        cited_document_ids=[policy_id, faq_id],
                        atoms=[_atom("A1", queue_sentence, entailed_by=[policy_id])],
                        expected_verdict="CONFLICTING_EVIDENCE",
                    )
                )

    cases.sort(key=lambda row: str(row["case_id"]))
    if len(cases) != 432:
        raise RuntimeError(f"A4.4a validation suite must contain 432 rows, got {len(cases)}.")
    return cases


def canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize validation rows canonically for provenance hashing."""
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def suite_summary() -> dict[str, Any]:
    """Return deterministic A4.4a counts and SHA-256 with no model inference."""
    rows = generate_cases()
    categories = sorted({str(row["category"]) for row in rows})
    return {
        "rows": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("calibration", "validation")
        },
        "category_counts": {
            category: sum(row["category"] == category for row in rows)
            for category in categories
        },
        "sha256": hashlib.sha256(canonical_jsonl_bytes(rows)).hexdigest(),
        "candidate_outputs_used": False,
        "query_text_used": False,
        "confirmatory_intents_used": False,
    }


if __name__ == "__main__":
    print(json.dumps(suite_summary(), indent=2, sort_keys=True))

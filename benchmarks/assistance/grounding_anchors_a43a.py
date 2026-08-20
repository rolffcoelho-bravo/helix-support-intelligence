"""Deterministic candidate-independent grounding anchors for Phase 4 A4.3a."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from helix_support_intelligence.data.helixbank import INTENTS, CorpusBundle, generate_bundle  # noqa: E402

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SPLIT_SEED = 20260820


def _ordered(values: set[str], prefix: str) -> list[str]:
    return sorted(
        values,
        key=lambda intent: hashlib.sha256(f"{prefix}:{intent}".encode()).hexdigest(),
    )


def development_intents(bundle: CorpusBundle) -> set[str]:
    """Reconstruct the frozen A4.0 development-intent partition without query labels."""
    conflicts = {
        str(row["intent"])
        for row in bundle.documents
        if bool(row["conflict_fixture"])
    }
    non_conflicts = set(INTENTS) - conflicts
    return set(_ordered(conflicts, "20260819")[:5]) | set(
        _ordered(non_conflicts, "20260819")[:55]
    )


def anchor_partition(bundle: CorpusBundle) -> dict[str, set[str]]:
    """Split only A4.0 development intents into evaluator calibration and validation."""
    development = development_intents(bundle)
    archived = {
        str(row["intent"])
        for row in bundle.documents
        if row["kind"] == "faq" and row["status"] == "archived"
    } & development
    conflicts = {
        str(row["intent"])
        for row in bundle.documents
        if bool(row["conflict_fixture"])
    } & development
    ordinary = development - archived - conflicts

    calibration = (
        set(_ordered(conflicts, f"{SPLIT_SEED}:a43a")[:3])
        | set(_ordered(archived, f"{SPLIT_SEED}:a43a")[:5])
        | set(_ordered(ordinary, f"{SPLIT_SEED}:a43a")[:32])
    )
    validation = development - calibration
    if len(calibration) != 40 or len(validation) != 20:
        raise RuntimeError("A4.3a anchor partition must be 40 calibration and 20 validation intents.")
    return {"calibration": calibration, "validation": validation}


def _sentence(body: str, prefix: str) -> str:
    for sentence in SENTENCE_RE.split(body.strip()):
        if sentence.startswith(prefix):
            return sentence.strip()
    raise RuntimeError(f"Missing frozen sentence prefix: {prefix}")


def _documents_by_intent(bundle: CorpusBundle) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for row in bundle.documents:
        intent = str(row["intent"])
        result.setdefault(intent, {})[str(row["kind"])] = dict(row)
    return result


def _anchor(
    *,
    anchor_id: str,
    split: str,
    intent: str,
    category: str,
    expected_entailment: bool,
    document_ids: list[str],
    hypothesis: str,
) -> dict[str, Any]:
    return {
        "anchor_id": anchor_id,
        "split": split,
        "intent": intent,
        "category": category,
        "expected_entailment": expected_entailment,
        "document_ids": document_ids,
        "hypothesis": hypothesis,
    }


def generate_anchors(bundle: CorpusBundle | None = None) -> list[dict[str, Any]]:
    """Materialize the frozen 372-row A4.3a grounding-anchor suite."""
    bundle = bundle or generate_bundle()
    docs = _documents_by_intent(bundle)
    partition = anchor_partition(bundle)
    anchors: list[dict[str, Any]] = []

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
            queue_sentence = _sentence(str(policy["body"]), "Requests are handled by")
            partner_guidance = _sentence(str(partner_policy["body"]), "HelixBank guidance for")
            stem = f"A43A-{split[:3].upper()}-{index + 1:02d}"

            anchors.extend(
                [
                    _anchor(
                        anchor_id=f"{stem}-LIT",
                        split=split,
                        intent=intent,
                        category="literal_policy",
                        expected_entailment=True,
                        document_ids=[policy_id],
                        hypothesis=queue_sentence,
                    ),
                    _anchor(
                        anchor_id=f"{stem}-PAR",
                        split=split,
                        intent=intent,
                        category="paraphrase_queue",
                        expected_entailment=True,
                        document_ids=[policy_id],
                        hypothesis=f"{title} requests go to the {queue} queue.",
                    ),
                    _anchor(
                        anchor_id=f"{stem}-CON",
                        split=split,
                        intent=intent,
                        category="contradiction_queue",
                        expected_entailment=False,
                        document_ids=[policy_id],
                        hypothesis=f"{title} requests are not handled by the {queue} queue.",
                    ),
                    _anchor(
                        anchor_id=f"{stem}-UNS",
                        split=split,
                        intent=intent,
                        category="unsupported_approval",
                        expected_entailment=False,
                        document_ids=[policy_id],
                        hypothesis=f"{title} is automatically approved without review.",
                    ),
                    _anchor(
                        anchor_id=f"{stem}-MIS",
                        split=split,
                        intent=intent,
                        category="citation_mismatch",
                        expected_entailment=False,
                        document_ids=[policy_id],
                        hypothesis=partner_guidance,
                    ),
                    _anchor(
                        anchor_id=f"{stem}-MUL",
                        split=split,
                        intent=intent,
                        category="multi_document_conjunction",
                        expected_entailment=True,
                        document_ids=[policy_id, partner_policy_id],
                        hypothesis=f"{queue_sentence} {partner_guidance}",
                    ),
                ]
            )

            if faq["status"] == "archived":
                anchors.append(
                    _anchor(
                        anchor_id=f"{stem}-STA",
                        split=split,
                        intent=intent,
                        category="stale_current_claim",
                        expected_entailment=False,
                        document_ids=[faq_id],
                        hypothesis="This FAQ is current evidence.",
                    )
                )
            if bool(faq["conflict_fixture"]):
                anchors.append(
                    _anchor(
                        anchor_id=f"{stem}-CFU",
                        split=split,
                        intent=intent,
                        category="conflict_union_claim",
                        expected_entailment=False,
                        document_ids=[policy_id, faq_id],
                        hypothesis="Review is optional when unresolved uncertainty remains.",
                    )
                )

    anchors.sort(key=lambda row: str(row["anchor_id"]))
    if len(anchors) != 372:
        raise RuntimeError(f"A4.3a anchor suite must contain 372 rows, got {len(anchors)}.")
    return anchors


def canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize anchor rows canonically for provenance hashing."""
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def suite_summary() -> dict[str, Any]:
    """Return deterministic counts and SHA-256 without running any model."""
    rows = generate_anchors()
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("calibration", "validation")
    }
    categories = sorted({str(row["category"]) for row in rows})
    return {
        "rows": len(rows),
        "split_counts": counts,
        "category_counts": {
            category: sum(row["category"] == category for row in rows) for category in categories
        },
        "sha256": hashlib.sha256(canonical_jsonl_bytes(rows)).hexdigest(),
        "candidate_outputs_used": False,
        "query_text_used": False,
        "confirmatory_intents_used": False,
    }


if __name__ == "__main__":
    print(json.dumps(suite_summary(), indent=2, sort_keys=True))

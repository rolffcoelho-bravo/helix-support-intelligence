"""Verify Phase 3 retrieval readiness without scoring the frozen query benchmark."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.helixbank import generate_bundle, manifest
from helix_support_intelligence.retrieval.core import (
    EligibilityPolicy,
    document_from_record,
    filter_eligible_documents,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "models" / "retrieval_ladder_v1.json"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("retrieval protocol must be a JSON object")
    return cast(dict[str, Any], payload)


def main() -> None:
    """Run metadata, corpus, filter, and model-pin checks without benchmark ranking."""
    protocol = _load_protocol()
    if protocol["protocol_id"] != "phase3-retrieval-r3.0-v1":
        raise SystemExit("unexpected Phase 3 retrieval protocol id")
    if protocol["status"] != "frozen_pre_evaluation":
        raise SystemExit("Phase 3 retrieval protocol is not frozen")

    guard = cast(dict[str, Any], protocol["execution_guard"])
    if guard["results_opened"] is not False:
        raise SystemExit("R3.1 preflight requires the registered pre-evaluation state")

    current_manifest = manifest()
    corpus = cast(dict[str, Any], protocol["corpus"])
    if corpus["version"] != current_manifest["corpus_version"]:
        raise SystemExit("corpus version drift detected")
    if corpus["generator_version"] != current_manifest["generator_version"]:
        raise SystemExit("corpus generator version drift detected")
    if corpus["counts"] != current_manifest["counts"]:
        raise SystemExit("corpus count drift detected")
    if corpus["sha256"] != current_manifest["sha256"]:
        raise SystemExit("corpus hash drift detected")

    bundle = generate_bundle()
    documents = tuple(document_from_record(record) for record in bundle.documents)
    filters = cast(dict[str, Any], corpus["eligibility_filters"])
    policy = EligibilityPolicy(
        evaluation_date=date.fromisoformat(str(corpus["evaluation_date"])),
        statuses=frozenset(cast(list[str], filters["status"])),
        permissions=frozenset(cast(list[str], filters["permission"])),
        audiences=frozenset(cast(list[str], filters["audience"])),
        jurisdictions=frozenset(cast(list[str], filters["jurisdiction"])),
    )
    eligible = filter_eligible_documents(documents, policy)
    if len(eligible) != 147:
        raise SystemExit(f"unexpected eligible-document count: {len(eligible)}")
    if sum(document.conflict_fixture for document in eligible) != 7:
        raise SystemExit("current conflict fixtures were incorrectly filtered")
    if sum(document.untrusted_content_fixture for document in eligible) != 5:
        raise SystemExit("current untrusted-content fixtures were incorrectly filtered")

    ladder = cast(list[dict[str, Any]], protocol["ladder"])
    by_id = {str(candidate["id"]): candidate for candidate in ladder}
    if list(by_id) != ["B0", "B1", "B2", "B3"]:
        raise SystemExit("retrieval ladder drift detected")

    b1_model = cast(dict[str, Any], by_id["B1"]["model"])
    b3_model = cast(dict[str, Any], by_id["B3"]["model"])
    expected_models = (
        (
            b1_model,
            "sentence-transformers/all-MiniLM-L6-v2",
            "c315f904dfc467d8b9c40ab4ed50b3a8d0866c15",
        ),
        (
            b3_model,
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "c5f2b386de279a97c53a702dd5189d1c407160dc",
        ),
    )
    for model, expected_id, expected_revision in expected_models:
        if model["id"] != expected_id or model["revision"] != expected_revision:
            raise SystemExit(f"model pin drift detected for {expected_id}")
        if FULL_SHA_RE.fullmatch(str(model["revision"])) is None:
            raise SystemExit(f"model revision is not a full commit SHA for {expected_id}")

    report = {
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "corpus_manifest_verified": True,
        "eligible_documents": len(eligible),
        "current_conflict_fixtures_retained": 7,
        "current_untrusted_content_fixtures_retained": 5,
        "model_pins_verified_from_protocol": [b1_model["id"], b3_model["id"]],
        "frozen_query_scores_computed_by_preflight": 0,
        "retrieval_results_generated_by_preflight": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

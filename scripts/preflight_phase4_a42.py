"""Validate the registered A4.2 development-only execution before scoring."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.helixbank import INTENTS, generate_bundle

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = ROOT / "configs" / "models" / "assistance_execution_a42_v1.json"
PROTOCOL_PATH = ROOT / "configs" / "models" / "assistance_protocol_v1.json"
BINDING_PATH = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return cast(dict[str, Any], payload)


def _partition() -> tuple[set[str], set[str]]:
    bundle = generate_bundle()
    conflicts = {
        str(row["intent"]) for row in bundle.queries if row["case_type"] == "conflicting_evidence"
    }
    non_conflicts = set(INTENTS) - conflicts

    def ordered(values: set[str]) -> list[str]:
        return sorted(
            values,
            key=lambda intent: hashlib.sha256(f"20260819:{intent}".encode()).hexdigest(),
        )

    development = set(ordered(conflicts)[:5]) | set(ordered(non_conflicts)[:55])
    return development, set(INTENTS) - development


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


def _adversarial_counts(development: set[str]) -> Counter[str]:
    bundle = generate_bundle()
    counts: Counter[str] = Counter()
    for query in bundle.queries:
        if str(query["intent"]) not in development:
            continue
        if query["case_type"] == "answerable":
            counts["direct_injection"] += 1
            counts["citation_spoof"] += 1
        if query["case_type"] == "outdated_evidence":
            number = int(str(query["query_id"]).split("-")[1])
            faq_id = f"FAQ-{number:03d}"
            faq = next(row for row in bundle.documents if row["document_id"] == faq_id)
            if faq["status"] == "archived":
                counts["archived_distractor"] += 1
    for document in bundle.documents:
        if (
            str(document["intent"]) in development
            and bool(document["untrusted_content_fixture"])
            and _eligible(document)
        ):
            counts["indirect_injection"] += 4
    return counts


def main() -> None:
    execution = _load(EXECUTION_PATH)
    protocol = _load(PROTOCOL_PATH)
    binding = _load(BINDING_PATH)
    development, confirmatory = _partition()
    bundle = generate_bundle()
    development_query_ids = {
        str(row["query_id"]) for row in bundle.queries if str(row["intent"]) in development
    }
    confirmatory_query_ids = {
        str(row["query_id"]) for row in bundle.queries if str(row["intent"]) in confirmatory
    }

    if execution["execution_id"] != "phase4-assistance-a4.2-development-v1":
        raise RuntimeError("Unexpected A4.2 execution identifier.")
    if execution["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("A4.2 protocol binding changed.")
    if execution["binding_id"] != binding["binding_id"]:
        raise RuntimeError("A4.2 model binding changed.")
    if execution["status"] != "registered_pre_execution":
        raise RuntimeError("A4.2 must remain pre-execution before registered scoring.")
    if len(development) != 60 or len(confirmatory) != 17:
        raise RuntimeError("A4.2 intent partition does not reconstruct.")
    if len(development_query_ids) != 240 or len(confirmatory_query_ids) != 68:
        raise RuntimeError("A4.2 query partition does not reconstruct.")
    if development_query_ids & confirmatory_query_ids:
        raise RuntimeError("Development and confirmatory queries overlap.")

    counts = _adversarial_counts(development)
    expected = execution["adversarial_development_counts"]
    for key in (
        "direct_injection",
        "citation_spoof",
        "indirect_injection",
        "archived_distractor",
    ):
        if counts[key] != int(expected[key]):
            raise RuntimeError(f"A4.2 adversarial count mismatch for {key}.")
    if sum(counts.values()) != int(expected["total"]):
        raise RuntimeError("A4.2 adversarial total does not reconstruct.")

    guard = execution["results_guard"]
    if guard["development_results_opened"] is not False:
        raise RuntimeError("Development results have already been opened.")
    if guard["confirmatory_results_opened"] is not False:
        raise RuntimeError("Confirmatory results have already been opened.")

    print(
        json.dumps(
            {
                "execution_id": execution["execution_id"],
                "development_intents": len(development),
                "development_queries": len(development_query_ids),
                "confirmatory_intents_opened": 0,
                "confirmatory_queries_opened": 0,
                "adversarial_development_counts": dict(counts),
                "generator_calls_made": 0,
                "nli_calls_made": 0,
                "performance_scores_computed": 0,
                "status": "passed",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Forensic scientific audit of the immutable A4.2 development artifact.

This audit is deliberately post-result and cannot rescue or retune A4.2. It
checks whether the frozen benchmark and evaluator support the interpretation
required for candidate selection. It never calls OpenAI or either NLI model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from helix_support_intelligence.data.helixbank import INTENTS, generate_bundle  # noqa: E402

SCIENTIFIC_SHA = "fec8c5c63978ccae858257e1a078029c2103943c"
CITATION_RE = re.compile(r"\[((?:POLICY|FAQ)-\d{3})\]")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def development_intents(bundle: Any) -> set[str]:
    conflicts = {
        str(row["intent"]) for row in bundle.queries if row["case_type"] == "conflicting_evidence"
    }
    non_conflicts = set(INTENTS) - conflicts

    def ordered(values: set[str]) -> list[str]:
        return sorted(
            values,
            key=lambda intent: hashlib.sha256(f"20260819:{intent}".encode()).hexdigest(),
        )

    return set(ordered(conflicts)[:5]) | set(ordered(non_conflicts)[:55])


def eligible(document: dict[str, object]) -> bool:
    valid_to = document["valid_to"]
    return bool(
        document["status"] == "current"
        and document["permission"] == "public_support"
        and document["audience"] == "customer_support"
        and document["jurisdiction"] == "fictional-global"
        and str(document["valid_from"]) <= "2026-08-19"
        and (valid_to is None or str(valid_to) >= "2026-08-19")
    )


def normalize(text: str) -> str:
    return " ".join(text.split())


def protocol_conflicts(bundle: Any, development: set[str]) -> list[dict[str, Any]]:
    documents = {str(row["document_id"]): row for row in bundle.documents}
    rows: list[dict[str, Any]] = []
    for query in bundle.queries:
        if str(query["intent"]) not in development:
            continue
        if query["expected_decision"] != "ANSWER_WITH_EVIDENCE":
            continue
        direct_ids = [
            str(judgment["document_id"])
            for judgment in bundle.judgments
            if judgment["query_id"] == query["query_id"]
            and int(judgment["relevance"]) >= 2
            and eligible(documents[str(judgment["document_id"])])
        ]
        conflict_ids = [
            document_id
            for document_id in direct_ids
            if bool(documents[document_id]["conflict_fixture"])
        ]
        if conflict_ids:
            rows.append(
                {
                    "query_id": query["query_id"],
                    "intent": query["intent"],
                    "case_type": query["case_type"],
                    "expected_decision": query["expected_decision"],
                    "direct_evidence_ids": sorted(direct_ids),
                    "current_conflict_document_ids": sorted(conflict_ids),
                }
            )
    return rows


def evaluator_anchor_audit(
    quality_rows: list[dict[str, Any]],
    bundle: Any,
) -> dict[str, Any]:
    documents = {str(row["document_id"]): row for row in bundle.documents}
    exact_rows: list[dict[str, Any]] = []
    for row in quality_rows:
        if row["candidate_id"] != "G0":
            continue
        for sentence in row["sentence_support"]:
            if not sentence["factual"]:
                continue
            hypothesis = normalize(CITATION_RE.sub("", str(sentence["text"])).strip())
            cited = [str(item) for item in sentence["citations"]]
            literal = any(
                hypothesis in normalize(str(documents[document_id]["body"]))
                for document_id in cited
                if document_id in documents
            )
            if not literal:
                continue
            exact_rows.append(
                {
                    "query_id": row["query_id"],
                    "hypothesis": hypothesis,
                    "citations": cited,
                    "support_verdict": sentence["support_verdict"],
                    "entailment_probability": sentence["entailment_probability"],
                }
            )

    probabilities = [
        float(row["entailment_probability"])
        for row in exact_rows
        if row["entailment_probability"] is not None
    ]
    rejected = [row for row in exact_rows if row["support_verdict"] != "SUPPORTED"]
    return {
        "literal_source_claim_count": len(exact_rows),
        "literal_source_claims_rejected_by_frozen_evaluator": len(rejected),
        "rejection_rate": len(rejected) / len(exact_rows) if exact_rows else None,
        "frozen_threshold": 0.8,
        "probability_min": min(probabilities) if probabilities else None,
        "probability_median": statistics.median(probabilities) if probabilities else None,
        "probability_max": max(probabilities) if probabilities else None,
        "probabilities_at_or_above_threshold": sum(value >= 0.8 for value in probabilities),
        "interpretation": (
            "These G0 factual claims are literal substrings of their cited frozen source "
            "documents. A systematic rejection therefore limits construct validity of the "
            "registered grounding metric on this benchmark."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir

    bundle = generate_bundle()
    development = development_intents(bundle)
    quality_rows = load_jsonl(output_dir / "quality_records.jsonl")
    results = load_json(output_dir / "results.json")
    mechanical = load_json(output_dir / "post_audit.json")
    manifest = load_json(output_dir / "execution_manifest.json")

    conflicts = protocol_conflicts(bundle, development)
    anchor = evaluator_anchor_audit(quality_rows, bundle)
    protocol_consistent = len(conflicts) == 0
    grounding_anchor_valid = (
        anchor["literal_source_claim_count"] > 0
        and anchor["literal_source_claims_rejected_by_frozen_evaluator"] == 0
    )
    selection_admissible = bool(
        mechanical["status"] == "PASSED_AUTOMATED_RECONSTRUCTION"
        and protocol_consistent
        and grounding_anchor_valid
    )

    audit = {
        "audit_id": "phase4-assistance-a4.2-development-forensic-audit-v1",
        "scientific_sha": SCIENTIFIC_SHA,
        "raw_artifact_run_id": 32390970870,
        "raw_artifact_id": 9415920431,
        "raw_artifact_zip_sha256": (
            "376537ebac7800ae9f4f3b3802fcc31a03672f040c3648674f712e7014bb4ca5"
        ),
        "mechanical_reconstruction_status": mechanical["status"],
        "manifest_github_sha": manifest.get("github_sha"),
        "manifest_sha_note": (
            "The manifest GITHUB_SHA is the recovery-wrapper trigger SHA. The workflow "
            "checked out and explicitly verified the scientific SHA before execution."
        ),
        "confirmatory_queries_opened": 0,
        "protocol_consistency": {
            "passed": protocol_consistent,
            "expected_answer_queries_with_current_conflict_evidence": len(conflicts),
            "affected_queries": conflicts,
            "reason": (
                "A4.0 requires no unresolved conflict for ANSWER_WITH_EVIDENCE, but these "
                "development answerable/outdated queries received a current direct FAQ that "
                "explicitly conflicts with the governing policy."
            ),
        },
        "grounding_evaluator_anchor": {
            "passed": grounding_anchor_valid,
            **anchor,
        },
        "mechanically_reconstructed_results": {
            "metrics": results["metrics"],
            "hypotheses": results["hypotheses"],
            "latency": results["latency"],
            "adversarial_development": results["adversarial_development"],
            "repeatability": results["repeatability"],
            "complexity_adoption": results["complexity_adoption"],
            "total_estimated_provider_cost_usd_all_a42_calls": results[
                "total_estimated_provider_cost_usd_all_a42_calls"
            ],
        },
        "selection_admissible": selection_admissible,
        "registered_winner_accepted": (
            results["complexity_adoption"]["registered_winner"] if selection_admissible else None
        ),
        "status": (
            "PASSED_SCIENTIFIC_VALIDITY"
            if selection_admissible
            else "FAILED_SCIENTIFIC_VALIDITY_NO_SELECTION"
        ),
        "post_score_tuning_performed": False,
        "benchmark_rerun_performed_by_this_audit": False,
        "required_next_gate": (
            "Version a corrected development protocol/corpus-evidence contract before any new "
            "candidate scoring. Preserve this A4.2 run as non-selection diagnostic evidence."
            if not selection_admissible
            else "Freeze the development winner before confirmatory evaluation."
        ),
    }

    (output_dir / "forensic_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# A4.2 forensic scientific audit",
        "",
        f"**Status: {audit['status']}**",
        "",
        f"Mechanical reconstruction: **{mechanical['status']}**.",
        f"Selection admissible: **{selection_admissible}**.",
        "Confirmatory queries opened: **0**.",
        "",
        "## Protocol consistency",
        "",
        (
            "Expected-answer development queries carrying current direct conflicting evidence: "
            f"**{len(conflicts)}**."
        ),
        "",
        "## Grounding evaluator anchor",
        "",
        (
            "Literal source claims emitted by deterministic G0 and checked by the frozen "
            f"evaluator: **{anchor['literal_source_claim_count']}**."
        ),
        (
            "Literal source claims rejected as unsupported: "
            f"**{anchor['literal_source_claims_rejected_by_frozen_evaluator']}**."
        ),
        (
            "Entailment probability range for those exact-source claims: "
            f"{anchor['probability_min']:.6f} to {anchor['probability_max']:.6f}; "
            "frozen threshold 0.800000."
        ),
        "",
        "The mechanically reconstructed candidate metrics remain preserved as diagnostic "
        "evidence, but no development winner is accepted by this forensic audit.",
    ]
    (output_dir / "forensic_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if selection_admissible:
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()

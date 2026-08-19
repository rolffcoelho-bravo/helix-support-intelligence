"""Independently reconstruct the registered Phase 3 R3.2 diagnostic slices."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

CANDIDATES = ("B0", "B1", "B2", "B3")
FAMILIES = ("case_type", "document_kind", "conflict_fixture", "untrusted_content_fixture")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _eligible_metadata() -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for row in generate_bundle().documents:
        status_ok = row["status"] == "current"
        permission_ok = row["permission"] == "public_support"
        audience_ok = row["audience"] == "customer_support"
        jurisdiction_ok = row["jurisdiction"] == "fictional-global"
        valid_from_ok = str(row["valid_from"]) <= "2026-08-19"
        valid_to = row["valid_to"]
        valid_to_ok = valid_to is None or str(valid_to) >= "2026-08-19"
        eligibility = (
            status_ok,
            permission_ok,
            audience_ok,
            jurisdiction_ok,
            valid_from_ok,
            valid_to_ok,
        )
        if all(eligibility):
            metadata[str(row["document_id"])] = row
    return metadata


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    recall20 = [float(row["recall_at_20"]) for row in rows if row["recall_at_20"] is not None]
    recall50 = [float(row["recall_at_50"]) for row in rows if row["recall_at_50"] is not None]
    return {
        "ndcg_at_10": statistics.fmean(float(row["ndcg_at_10"]) for row in rows) if rows else 0.0,
        "mrr_at_10": statistics.fmean(float(row["mrr_at_10"]) for row in rows) if rows else 0.0,
        "recall_at_20": statistics.fmean(recall20) if recall20 else 0.0,
        "recall_at_20_queries": len(recall20),
        "recall_at_50": statistics.fmean(recall50) if recall50 else 0.0,
        "recall_at_50_queries": len(recall50),
        "query_count": len(rows),
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _groups() -> dict[str, dict[str, list[str]]]:
    bundle = generate_bundle()
    metadata = _eligible_metadata()
    positive_by_query: dict[str, list[str]] = defaultdict(list)
    for judgment in bundle.judgments:
        document_id = str(judgment["document_id"])
        if document_id in metadata and int(judgment["relevance"]) >= 1:
            positive_by_query[str(judgment["query_id"])].append(document_id)

    result: dict[str, dict[str, list[str]]] = {
        "case_type": defaultdict(list),
        "document_kind": {"POLICY": [], "FAQ": []},
        "conflict_fixture": {"associated": []},
        "untrusted_content_fixture": {"associated": []},
    }
    for query in bundle.queries:
        query_id = str(query["query_id"])
        result["case_type"][str(query["case_type"])].append(query_id)
        documents = positive_by_query.get(query_id, [])
        kinds = {str(metadata[document_id]["kind"]).upper() for document_id in documents}
        for kind in ("POLICY", "FAQ"):
            if kind in kinds:
                result["document_kind"][kind].append(query_id)
        if any(
            bool(metadata[document_id]["conflict_fixture"]) for document_id in documents
        ):
            result["conflict_fixture"]["associated"].append(query_id)
        if any(
            bool(metadata[document_id]["untrusted_content_fixture"])
            for document_id in documents
        ):
            result["untrusted_content_fixture"]["associated"].append(query_id)
    return result


def verify(output_dir: Path) -> dict[str, object]:
    failures: list[str] = []
    slices = _read_json(output_dir / "diagnostic_slices.json")
    metric_rows = _read_jsonl(output_dir / "query_metrics.jsonl")
    metric_map = {
        (str(row["candidate"]), str(row["query_id"])): row for row in metric_rows
    }
    if len(metric_map) != 308 * 4:
        failures.append("query_metrics.jsonl does not contain exactly 308 x 4 unique rows")

    selection_use_ok = slices.get("selection_use") == "descriptive_only"
    if not selection_use_ok:
        failures.append("diagnostic slices are not explicitly marked descriptive_only")

    stored_slices = slices.get("slices")
    family_keys_ok = isinstance(stored_slices, dict) and tuple(sorted(stored_slices)) == tuple(
        sorted(FAMILIES)
    )
    if not family_keys_ok:
        failures.append(
            "diagnostic slice families do not exactly match the registered four families"
        )

    groups = _groups()
    group_membership_ok = True
    metric_reconstruction_ok = True
    reconstructed: dict[str, object] = {}

    if isinstance(stored_slices, dict):
        for family in FAMILIES:
            expected_family = groups[family]
            stored_family = stored_slices.get(family)
            if not isinstance(stored_family, dict):
                group_membership_ok = False
                continue
            if set(stored_family) != set(expected_family):
                group_membership_ok = False
            reconstructed[family] = {}
            reconstructed_family = reconstructed[family]
            assert isinstance(reconstructed_family, dict)
            for label, query_ids in sorted(expected_family.items()):
                stored_group = stored_family.get(label)
                if not isinstance(stored_group, dict):
                    group_membership_ok = False
                    continue
                if int(stored_group.get("query_count", -1)) != len(query_ids):
                    group_membership_ok = False
                candidate_payload: dict[str, object] = {}
                reconstructed_family[label] = {
                    "query_count": len(query_ids),
                    "candidates": candidate_payload,
                }
                stored_candidates = stored_group.get("candidates")
                candidate_set_ok = isinstance(stored_candidates, dict) and set(
                    stored_candidates
                ) == set(CANDIDATES)
                if not candidate_set_ok:
                    metric_reconstruction_ok = False
                    continue
                assert isinstance(stored_candidates, dict)
                for candidate in CANDIDATES:
                    rows = [metric_map[(candidate, query_id)] for query_id in query_ids]
                    aggregate = _aggregate(rows)
                    candidate_payload[candidate] = aggregate
                    recorded = stored_candidates[candidate]
                    if not isinstance(recorded, dict):
                        metric_reconstruction_ok = False
                        continue
                    for key in (
                        "ndcg_at_10",
                        "mrr_at_10",
                        "recall_at_20",
                        "recall_at_50",
                    ):
                        if not _close(float(recorded[key]), float(aggregate[key])):
                            metric_reconstruction_ok = False
                    for key in (
                        "recall_at_20_queries",
                        "recall_at_50_queries",
                        "query_count",
                    ):
                        if int(recorded[key]) != int(aggregate[key]):
                            metric_reconstruction_ok = False

    if not group_membership_ok:
        failures.append("diagnostic group labels or query counts do not reconstruct")
    if not metric_reconstruction_ok:
        failures.append("diagnostic aggregate metrics do not reconstruct from query metrics")

    checks = {
        "selection_use_descriptive_only": selection_use_ok,
        "registered_family_set": family_keys_ok,
        "group_membership": group_membership_ok,
        "metric_reconstruction": metric_reconstruction_ok,
    }
    passed = not failures and all(checks.values())
    audit: dict[str, object] = {
        "execution_id": "phase3-retrieval-r3.2-v1",
        "verdict": "PASSED" if passed else "FAILED",
        "checks": checks,
        "failures": failures,
        "reconstructed_slices": reconstructed,
    }
    (output_dir / "diagnostic_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit("R3.2 diagnostic audit failed; inspect diagnostic_audit.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

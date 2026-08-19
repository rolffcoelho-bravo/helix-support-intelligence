"""Generate the predeclared descriptive Phase 3 R3.2 diagnostic slices."""

from __future__ import annotations

import argparse
import json
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object in {path}")
            rows.append(payload)
    return rows


def _eligible_document_metadata() -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for row in generate_bundle().documents:
        if row["status"] != "current":
            continue
        if row["permission"] != "public_support":
            continue
        if row["audience"] != "customer_support":
            continue
        if row["jurisdiction"] != "fictional-global":
            continue
        if str(row["valid_from"]) > "2026-08-19":
            continue
        valid_to = row["valid_to"]
        if valid_to is not None and str(valid_to) < "2026-08-19":
            continue
        metadata[str(row["document_id"])] = row
    return metadata


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "ndcg_at_10": 0.0,
            "mrr_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_20_queries": 0,
            "recall_at_50": 0.0,
            "recall_at_50_queries": 0,
            "query_count": 0,
        }
    recall20 = [float(row["recall_at_20"]) for row in rows if row["recall_at_20"] is not None]
    recall50 = [float(row["recall_at_50"]) for row in rows if row["recall_at_50"] is not None]
    return {
        "ndcg_at_10": statistics.fmean(float(row["ndcg_at_10"]) for row in rows),
        "mrr_at_10": statistics.fmean(float(row["mrr_at_10"]) for row in rows),
        "recall_at_20": statistics.fmean(recall20) if recall20 else 0.0,
        "recall_at_20_queries": len(recall20),
        "recall_at_50": statistics.fmean(recall50) if recall50 else 0.0,
        "recall_at_50_queries": len(recall50),
        "query_count": len(rows),
    }


def generate(output_dir: Path) -> dict[str, object]:
    metric_rows = _read_jsonl(output_dir / "query_metrics.jsonl")
    metric_map = {
        (str(row["candidate"]), str(row["query_id"])): row for row in metric_rows
    }
    if len(metric_map) != 308 * 4:
        raise ValueError("query metric evidence is incomplete")

    bundle = generate_bundle()
    metadata = _eligible_document_metadata()
    positive_by_query: dict[str, list[str]] = defaultdict(list)
    for judgment in bundle.judgments:
        document_id = str(judgment["document_id"])
        if document_id in metadata and int(judgment["relevance"]) >= 1:
            positive_by_query[str(judgment["query_id"])].append(document_id)

    groups: dict[str, dict[str, list[str]]] = {
        "case_type": defaultdict(list),
        "document_kind": {"POLICY": [], "FAQ": []},
        "conflict_fixture": {"associated": []},
        "untrusted_content_fixture": {"associated": []},
    }

    for query in bundle.queries:
        query_id = str(query["query_id"])
        groups["case_type"][str(query["case_type"])].append(query_id)
        positive_documents = positive_by_query.get(query_id, [])
        kinds = {str(metadata[document_id]["kind"]).upper() for document_id in positive_documents}
        for kind in ("POLICY", "FAQ"):
            if kind in kinds:
                groups["document_kind"][kind].append(query_id)
        if any(bool(metadata[document_id]["conflict_fixture"]) for document_id in positive_documents):
            groups["conflict_fixture"]["associated"].append(query_id)
        if any(
            bool(metadata[document_id]["untrusted_content_fixture"])
            for document_id in positive_documents
        ):
            groups["untrusted_content_fixture"]["associated"].append(query_id)

    payload: dict[str, object] = {
        "execution_id": "phase3-retrieval-r3.2-v1",
        "selection_use": "descriptive_only",
        "slices": {},
    }
    slices = payload["slices"]
    assert isinstance(slices, dict)
    for family, family_groups in groups.items():
        slices[family] = {}
        family_payload = slices[family]
        assert isinstance(family_payload, dict)
        for label, query_ids in sorted(family_groups.items()):
            family_payload[label] = {
                "query_count": len(query_ids),
                "candidates": {
                    candidate: _aggregate(
                        [metric_map[(candidate, query_id)] for query_id in query_ids]
                    )
                    for candidate in CANDIDATES
                },
            }

    (output_dir / "diagnostic_slices.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

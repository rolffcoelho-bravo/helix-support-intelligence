"""Evaluate the frozen Phase 3 B0 BM25 baseline on development queries only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.banking77 import sha256_file
from helix_support_intelligence.retrieval.bm25 import BM25Index
from helix_support_intelligence.retrieval.metrics import QueryMetrics, evaluate_query, mean_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_CONFIG = REPO_ROOT / "configs" / "retrieval" / "phase3_benchmark_v1.json"
DEFAULT_B0_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b0_bm25_v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, Any], payload)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            payload: object = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(cast(dict[str, object], payload))
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_inputs(input_dir: Path, benchmark: Mapping[str, Any]) -> None:
    frozen = benchmark.get("hash_manifest")
    if not isinstance(frozen, dict) or frozen.get("status") != "frozen_before_retrieval_scoring":
        raise ValueError("retrieval benchmark is not frozen for scoring")
    expected = {
        "documents.jsonl": frozen["candidate_documents_sha256"],
        "development_queries.jsonl": frozen["development_queries_sha256"],
        "development_qrels.jsonl": frozen["development_qrels_sha256"],
    }
    for name, digest in expected.items():
        observed = sha256_file(input_dir / name)
        if observed != digest:
            raise ValueError(f"B0 input hash drifted for {name}: {observed} != {digest}")
    for forbidden in ("confirmatory_queries.jsonl", "confirmatory_qrels.jsonl"):
        if (input_dir / forbidden).exists():
            raise ValueError(f"B0 development evaluator refuses sealed file: {forbidden}")


def _qrels_by_query(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        query_id = str(row["query_id"])
        document_id = str(row["document_id"])
        relevance = row["relevance"]
        if not isinstance(relevance, int) or isinstance(relevance, bool):
            raise TypeError("qrel relevance must be an integer")
        qrels[query_id][document_id] = relevance
    return dict(qrels)


def _ranking_hash(query_id: str, document_ids: Iterable[str], digest: Any) -> None:
    digest.update(query_id.encode("utf-8"))
    digest.update(b"\0")
    for document_id in document_ids:
        digest.update(document_id.encode("utf-8"))
        digest.update(b"\0")


def evaluate(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run frozen B0 scoring and emit deterministic relevance evidence."""

    benchmark = _load_object(DEFAULT_BENCHMARK_CONFIG)
    config = _load_object(DEFAULT_B0_CONFIG)
    if config.get("status") != "frozen_before_first_score":
        raise ValueError("B0 configuration is not frozen before first score")
    if config.get("benchmark_version") != benchmark.get("version"):
        raise ValueError("B0 benchmark version mismatch")
    _validate_inputs(input_dir, benchmark)

    documents = _load_jsonl(input_dir / "documents.jsonl")
    queries = _load_jsonl(input_dir / "development_queries.jsonl")
    qrel_rows = _load_jsonl(input_dir / "development_qrels.jsonl")
    qrels = _qrels_by_query(qrel_rows)

    expected_counts = cast(dict[str, Any], benchmark["selection"])["expected_counts"]
    if len(documents) != expected_counts["candidate_documents"]:
        raise ValueError("B0 candidate-document count drifted")
    if len(queries) != expected_counts["development_queries"]:
        raise ValueError("B0 development-query count drifted")
    if len(qrel_rows) != expected_counts["development_qrels"]:
        raise ValueError("B0 development-qrel count drifted")

    model = cast(dict[str, Any], config["model"])
    build_start = time.perf_counter_ns()
    index = BM25Index.build(documents, k1=float(model["k1"]), b=float(model["b"]))
    build_ms = (time.perf_counter_ns() - build_start) / 1_000_000.0

    per_query: list[QueryMetrics] = []
    per_intent: dict[str, list[QueryMetrics]] = defaultdict(list)
    latency_ms: list[float] = []
    rank_digest = hashlib.sha256()

    for query in queries:
        query_id = str(query["query_id"])
        text = str(query["text"])
        intent = str(query["intent"])
        if query_id not in qrels:
            raise ValueError(f"B0 query has no qrels: {query_id}")

        start = time.perf_counter_ns()
        ranking = index.score(text)
        latency_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
        document_ids = [item.document_id for item in ranking]
        _ranking_hash(query_id, document_ids, rank_digest)
        metrics = evaluate_query(document_ids, qrels[query_id])
        per_query.append(metrics)
        per_intent[intent].append(metrics)

    aggregate = mean_metrics(per_query)
    intent_metrics = {
        intent: mean_metrics(metrics) for intent, metrics in sorted(per_intent.items())
    }
    if len(intent_metrics) != 77:
        raise ValueError("B0 development result lost one or more intents")

    results: dict[str, object] = {
        "version": "retrieval-b0-development-v1",
        "status": "development_only",
        "model": "B0",
        "benchmark_version": benchmark["version"],
        "configuration_version": config["version"],
        "query_count": len(queries),
        "candidate_document_count": len(documents),
        "qrel_count": len(qrel_rows),
        "metrics": aggregate,
        "ranking_sha256": rank_digest.hexdigest(),
        "index": {
            "vocabulary_size": len(index.document_frequency),
            "average_document_length": index.average_document_length,
            "build_ms": build_ms,
        },
        "latency_ms": {
            "p50": statistics.median(latency_ms),
            "p95": _percentile(latency_ms, 0.95),
            "mean": statistics.fmean(latency_ms),
            "interpretation": "descriptive_ci_hardware_only",
        },
        "per_intent": intent_metrics,
        "confirmatory_partition_opened": False,
        "official_banking77_test_accessed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = cast(dict[str, float], results["metrics"])
    report = "\n".join(
        (
            "# Phase 3 B0 BM25 Development Result",
            "",
            "> Development evidence only. The sealed Phase 3 confirmatory "
            "partition was not opened.",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| nDCG@10 | {metrics['ndcg_at_10']:.6f} |",
            f"| MRR@10 | {metrics['mrr_at_10']:.6f} |",
            f"| Recall@20 | {metrics['recall_at_20']:.6f} |",
            f"| Recall@50 | {metrics['recall_at_50']:.6f} |",
            f"| Success@1 | {metrics['success_at_1']:.6f} |",
            "| Citation-eligible recall@20 | "
            f"{metrics['citation_eligible_recall_at_20']:.6f} |",
            "",
            f"Ranking SHA-256: `{results['ranking_sha256']}`.",
            "",
            "Latency is descriptive GitHub-hosted CI evidence only and is not a production claim.",
            "",
        )
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    results = evaluate(args.input_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

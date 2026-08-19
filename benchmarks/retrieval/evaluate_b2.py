# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5",
#   "scikit-learn==1.8.0",
#   "scipy==1.17.0",
#   "sentence-transformers==5.5.1",
#   "torch==2.13.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Evaluate the frozen Phase 3 B2 Reciprocal Rank Fusion candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import sentence_transformers
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from helix_support_intelligence.data.banking77 import sha256_file
from helix_support_intelligence.retrieval.bm25 import BM25Index
from helix_support_intelligence.retrieval.fusion import reciprocal_rank_fusion
from helix_support_intelligence.retrieval.metrics import QueryMetrics, evaluate_query, mean_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = REPO_ROOT / "configs" / "retrieval" / "phase3_benchmark_v1.json"
B0_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b0_bm25_v1.json"
B1_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b1_dense_v1.json"
B2_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b2_rrf_v1.json"
B0_RESULT = REPO_ROOT / "benchmarks" / "retrieval" / "results" / "b0_development_v1.json"
B1_RESULT = REPO_ROOT / "benchmarks" / "retrieval" / "results" / "b1_development_v1.json"


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
            raise ValueError(f"B2 input hash drifted for {name}: {observed} != {digest}")
    for forbidden in ("confirmatory_queries.jsonl", "confirmatory_qrels.jsonl"):
        if (input_dir / forbidden).exists():
            raise ValueError(f"B2 development evaluator refuses sealed file: {forbidden}")


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


def _ranking_hash(rankings: Mapping[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(rankings):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        for document_id in rankings[query_id]:
            digest.update(document_id.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _verify_model_weights(model: Mapping[str, Any]) -> str:
    downloaded = Path(
        hf_hub_download(
            repo_id=str(model["model_id"]),
            filename="model.safetensors",
            revision=str(model["revision"]),
        )
    )
    observed = sha256_file(downloaded)
    expected = str(model["model_safetensors_sha256"])
    if observed != expected:
        raise ValueError(f"B1 parent model weight hash drifted: {observed} != {expected}")
    return observed


def _stable_dense_rank(scores: np.ndarray, document_ids: list[str]) -> list[str]:
    if scores.ndim != 1 or len(scores) != len(document_ids):
        raise ValueError("B1 parent score vector shape drifted")
    order = sorted(
        range(len(document_ids)),
        key=lambda index: (-float(scores[index]), document_ids[index]),
    )
    return [document_ids[index] for index in order]


def _reconstruct_b0(
    documents: list[dict[str, object]],
    queries: list[dict[str, object]],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[str]], float]:
    model = cast(dict[str, Any], config["model"])
    started = time.perf_counter()
    index = BM25Index.build(documents, k1=float(model["k1"]), b=float(model["b"]))
    rankings = {
        str(query["query_id"]): [
            item.document_id for item in index.score(str(query["text"]))
        ]
        for query in queries
    }
    return rankings, time.perf_counter() - started


def _reconstruct_b1(
    documents: list[dict[str, object]],
    queries: list[dict[str, object]],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, float], str]:
    model = cast(dict[str, Any], config["model"])
    environment = cast(dict[str, Any], config["environment"])
    if str(model["device"]) != "cpu" or environment.get("cpu_only") is not True:
        raise ValueError("B1 parent reconstruction must remain CPU-only")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(int(environment["threads"]))
    torch.set_num_interop_threads(int(environment["threads"]))
    torch.use_deterministic_algorithms(True)
    weight_sha = _verify_model_weights(model)

    load_started = time.perf_counter()
    encoder = SentenceTransformer(
        str(model["model_id"]),
        revision=str(model["revision"]),
        device="cpu",
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    load_seconds = time.perf_counter() - load_started
    if encoder.get_sentence_embedding_dimension() != int(model["embedding_dimension"]):
        raise ValueError("B1 parent embedding dimension drifted")
    if encoder.max_seq_length != int(model["max_sequence_length"]):
        raise ValueError("B1 parent maximum sequence length drifted")

    ordered_documents = sorted(documents, key=lambda item: str(item["document_id"]))
    document_ids = [str(item["document_id"]) for item in ordered_documents]
    document_texts = [f"{item['title']}\n{item['body']}" for item in ordered_documents]
    query_instruction = str(model["query_instruction"])
    query_texts = [query_instruction + str(item["text"]) for item in queries]

    document_started = time.perf_counter()
    document_embeddings = encoder.encode(
        document_texts,
        batch_size=int(model["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(model["normalize_embeddings"]),
    )
    document_seconds = time.perf_counter() - document_started

    query_started = time.perf_counter()
    query_embeddings = encoder.encode(
        query_texts,
        batch_size=int(model["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(model["normalize_embeddings"]),
    )
    query_seconds = time.perf_counter() - query_started

    if document_embeddings.shape != (len(documents), int(model["embedding_dimension"])):
        raise ValueError("B1 parent document embedding shape drifted")
    if query_embeddings.shape != (len(queries), int(model["embedding_dimension"])):
        raise ValueError("B1 parent query embedding shape drifted")

    rank_started = time.perf_counter()
    similarities = query_embeddings @ document_embeddings.T
    rankings = {
        str(query["query_id"]): _stable_dense_rank(
            similarities[row_index],
            document_ids,
        )
        for row_index, query in enumerate(queries)
    }
    rank_seconds = time.perf_counter() - rank_started
    timing = {
        "model_load": load_seconds,
        "document_encoding": document_seconds,
        "query_encoding": query_seconds,
        "similarity_and_ranking": rank_seconds,
    }
    return rankings, timing, weight_sha


def _comparison_counts(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, int]]:
    metric_names = next(iter(candidate.values())).keys()
    output: dict[str, dict[str, int]] = {}
    for metric in metric_names:
        wins = ties = losses = 0
        for intent, values in candidate.items():
            delta = values[metric] - reference[intent][metric]
            if delta > 1e-15:
                wins += 1
            elif delta < -1e-15:
                losses += 1
            else:
                ties += 1
        output[metric] = {"wins": wins, "ties": ties, "losses": losses}
    return output


def evaluate(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run frozen B2 development scoring without opening confirmatory data."""

    benchmark = _load_object(BENCHMARK_CONFIG)
    b0_config = _load_object(B0_CONFIG)
    b1_config = _load_object(B1_CONFIG)
    b2_config = _load_object(B2_CONFIG)
    b0_result = _load_object(B0_RESULT)
    b1_result = _load_object(B1_RESULT)

    if b2_config.get("status") != "frozen_before_first_score":
        raise ValueError("B2 configuration is not frozen before first score")
    if b2_config.get("benchmark_version") != benchmark.get("version"):
        raise ValueError("B2 benchmark version mismatch")
    _validate_inputs(input_dir, benchmark)

    documents = _load_jsonl(input_dir / "documents.jsonl")
    queries = _load_jsonl(input_dir / "development_queries.jsonl")
    qrel_rows = _load_jsonl(input_dir / "development_qrels.jsonl")
    qrels = _qrels_by_query(qrel_rows)
    expected_counts = cast(dict[str, Any], benchmark["selection"])["expected_counts"]
    if len(documents) != expected_counts["candidate_documents"]:
        raise ValueError("B2 candidate-document count drifted")
    if len(queries) != expected_counts["development_queries"]:
        raise ValueError("B2 development-query count drifted")
    if len(qrel_rows) != expected_counts["development_qrels"]:
        raise ValueError("B2 development-qrel count drifted")

    parents = cast(dict[str, Any], b2_config["parents"])
    accepted_b0 = str(
        cast(dict[str, Any], b0_result["deterministic_evidence"])["ranking_sha256"]
    )
    accepted_b1 = str(
        cast(dict[str, Any], b1_result["deterministic_evidence"])["ranking_sha256"]
    )
    if parents["B0"]["accepted_ranking_sha256"] != accepted_b0:
        raise ValueError("B2 frozen B0 parent hash disagrees with audited B0 result")
    if parents["B1"]["accepted_ranking_sha256"] != accepted_b1:
        raise ValueError("B2 frozen B1 parent hash disagrees with audited B1 result")

    b0_rankings, b0_seconds = _reconstruct_b0(documents, queries, b0_config)
    observed_b0 = _ranking_hash(b0_rankings)
    if observed_b0 != accepted_b0:
        raise ValueError(f"B0 parent ranking drifted: {observed_b0} != {accepted_b0}")

    b1_rankings, b1_timing, weight_sha = _reconstruct_b1(documents, queries, b1_config)
    observed_b1 = _ranking_hash(b1_rankings)
    if observed_b1 != accepted_b1:
        raise ValueError(f"B1 parent ranking drifted: {observed_b1} != {accepted_b1}")

    fusion = cast(dict[str, Any], b2_config["fusion"])
    k = int(fusion["k"])
    rank_depth = int(fusion["rank_depth"])
    weights = {
        "B0": float(parents["B0"]["weight"]),
        "B1": float(parents["B1"]["weight"]),
    }

    fusion_started = time.perf_counter()
    fused_rankings = {
        query_id: reciprocal_rank_fusion(
            {"B0": b0_rankings[query_id], "B1": b1_rankings[query_id]},
            k=k,
            weights=weights,
            rank_depth=rank_depth,
        )
        for query_id in sorted(b0_rankings)
    }
    fusion_seconds = time.perf_counter() - fusion_started

    per_query: list[QueryMetrics] = []
    per_intent: dict[str, list[QueryMetrics]] = defaultdict(list)
    query_by_id = {str(item["query_id"]): item for item in queries}
    for query_id, ranking in fused_rankings.items():
        if query_id not in qrels:
            raise ValueError(f"B2 query has no qrels: {query_id}")
        metrics = evaluate_query(ranking, qrels[query_id])
        per_query.append(metrics)
        per_intent[str(query_by_id[query_id]["intent"])].append(metrics)

    aggregate = mean_metrics(per_query)
    intent_metrics = {
        intent: mean_metrics(metrics) for intent, metrics in sorted(per_intent.items())
    }
    if len(intent_metrics) != 77:
        raise ValueError("B2 development result lost one or more intents")

    b0_metrics = cast(dict[str, float], b0_result["metrics"])
    b1_metrics = cast(dict[str, float], b1_result["metrics"])
    comparison_b0 = {key: aggregate[key] - b0_metrics[key] for key in aggregate}
    comparison_b1 = {key: aggregate[key] - b1_metrics[key] for key in aggregate}

    parent_intent_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for system, rankings in (("B0", b0_rankings), ("B1", b1_rankings)):
        grouped: dict[str, list[QueryMetrics]] = defaultdict(list)
        for query_id, ranking in rankings.items():
            grouped[str(query_by_id[query_id]["intent"])].append(
                evaluate_query(ranking, qrels[query_id])
            )
        parent_intent_metrics[system] = {
            intent: mean_metrics(metrics) for intent, metrics in sorted(grouped.items())
        }

    results: dict[str, object] = {
        "version": "retrieval-b2-development-v1",
        "status": "development_only",
        "model": "B2",
        "benchmark_version": benchmark["version"],
        "configuration_version": b2_config["version"],
        "fusion": {
            "family": fusion["family"],
            "k": k,
            "rank_depth": rank_depth,
            "weights": weights,
            "rank_origin": fusion["rank_origin"],
            "score_normalization": fusion["score_normalization"],
            "tie_break": fusion["tie_break"],
        },
        "query_count": len(queries),
        "candidate_document_count": len(documents),
        "qrel_count": len(qrel_rows),
        "parent_ranking_sha256": {"B0": observed_b0, "B1": observed_b1},
        "ranking_sha256": _ranking_hash(fused_rankings),
        "metrics": aggregate,
        "comparison_to_b0": comparison_b0,
        "comparison_to_b1": comparison_b1,
        "per_intent": intent_metrics,
        "per_intent_comparison_counts_vs_b0": _comparison_counts(
            intent_metrics, parent_intent_metrics["B0"]
        ),
        "per_intent_comparison_counts_vs_b1": _comparison_counts(
            intent_metrics, parent_intent_metrics["B1"]
        ),
        "timing_seconds": {
            "b0_parent_reconstruction": b0_seconds,
            "b1_parent_reconstruction": b1_timing,
            "rrf_fusion": fusion_seconds,
            "interpretation": "descriptive_ci_hardware_only",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "b1_model_safetensors_sha256": weight_sha,
        },
        "confirmatory_partition_opened": False,
        "official_banking77_test_accessed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = "\n".join(
        (
            "# Phase 3 B2 Reciprocal Rank Fusion Development Result",
            "",
            "> Development evidence only. The sealed Phase 3 confirmatory partition was not opened.",
            "",
            "| Metric | B0 | B1 | B2 | B2 - B1 |",
            "|---|---:|---:|---:|---:|",
            *(
                f"| {key} | {b0_metrics[key]:.6f} | {b1_metrics[key]:.6f} | "
                f"{aggregate[key]:.6f} | {comparison_b1[key]:+.6f} |"
                for key in aggregate
            ),
            "",
            f"B0 parent ranking SHA-256: `{observed_b0}`.",
            f"B1 parent ranking SHA-256: `{observed_b1}`.",
            f"B2 fused ranking SHA-256: `{results['ranking_sha256']}`.",
            "",
            "Timing is descriptive GitHub-hosted CPU evidence only and is not a production claim.",
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

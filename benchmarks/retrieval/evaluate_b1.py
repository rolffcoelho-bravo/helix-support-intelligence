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
"""Evaluate the frozen Phase 3 B1 dense retriever on development queries only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
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
from helix_support_intelligence.retrieval.metrics import QueryMetrics, evaluate_query, mean_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = REPO_ROOT / "configs" / "retrieval" / "phase3_benchmark_v1.json"
B1_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b1_dense_v1.json"
B0_RESULT = REPO_ROOT / "benchmarks" / "retrieval" / "results" / "b0_development_v1.json"


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
            raise ValueError(f"B1 input hash drifted for {name}: {observed} != {digest}")
    for forbidden in ("confirmatory_queries.jsonl", "confirmatory_qrels.jsonl"):
        if (input_dir / forbidden).exists():
            raise ValueError(f"B1 development evaluator refuses sealed file: {forbidden}")


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
        raise ValueError(f"B1 model weight hash drifted: {observed} != {expected}")
    return observed


def _stable_rank(scores: np.ndarray, document_ids: list[str]) -> list[str]:
    if scores.ndim != 1 or len(scores) != len(document_ids):
        raise ValueError("B1 score vector shape drifted")
    order = sorted(range(len(document_ids)), key=lambda index: (-float(scores[index]), document_ids[index]))
    return [document_ids[index] for index in order]


def _ranking_hash(query_id: str, document_ids: Iterable[str], digest: Any) -> None:
    digest.update(query_id.encode("utf-8"))
    digest.update(b"\0")
    for document_id in document_ids:
        digest.update(document_id.encode("utf-8"))
        digest.update(b"\0")


def evaluate(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run the frozen B1 development benchmark without opening confirmatory data."""

    benchmark = _load_object(BENCHMARK_CONFIG)
    config = _load_object(B1_CONFIG)
    if config.get("status") != "frozen_before_first_score":
        raise ValueError("B1 configuration is not frozen before first score")
    if config.get("benchmark_version") != benchmark.get("version"):
        raise ValueError("B1 benchmark version mismatch")
    _validate_inputs(input_dir, benchmark)

    documents = _load_jsonl(input_dir / "documents.jsonl")
    queries = _load_jsonl(input_dir / "development_queries.jsonl")
    qrel_rows = _load_jsonl(input_dir / "development_qrels.jsonl")
    qrels = _qrels_by_query(qrel_rows)
    expected_counts = cast(dict[str, Any], benchmark["selection"])["expected_counts"]
    if len(documents) != expected_counts["candidate_documents"]:
        raise ValueError("B1 candidate-document count drifted")
    if len(queries) != expected_counts["development_queries"]:
        raise ValueError("B1 development-query count drifted")
    if len(qrel_rows) != expected_counts["development_qrels"]:
        raise ValueError("B1 development-qrel count drifted")

    model_config = cast(dict[str, Any], config["model"])
    environment = cast(dict[str, Any], config["environment"])
    if str(model_config["device"]) != "cpu" or environment.get("cpu_only") is not True:
        raise ValueError("B1 development evaluation must remain CPU-only")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(int(environment["threads"]))
    torch.set_num_interop_threads(int(environment["threads"]))
    torch.use_deterministic_algorithms(True)
    weight_sha = _verify_model_weights(model_config)

    load_started = time.perf_counter()
    encoder = SentenceTransformer(
        str(model_config["model_id"]),
        revision=str(model_config["revision"]),
        device="cpu",
        trust_remote_code=bool(model_config["trust_remote_code"]),
    )
    model_load_seconds = time.perf_counter() - load_started
    if encoder.get_sentence_embedding_dimension() != int(model_config["embedding_dimension"]):
        raise ValueError("B1 embedding dimension drifted")
    if encoder.max_seq_length != int(model_config["max_sequence_length"]):
        raise ValueError("B1 maximum sequence length drifted")

    ordered_documents = sorted(documents, key=lambda item: str(item["document_id"]))
    document_ids = [str(item["document_id"]) for item in ordered_documents]
    document_texts = [f"{item['title']}\n{item['body']}" for item in ordered_documents]
    query_instruction = str(model_config["query_instruction"])
    query_texts = [query_instruction + str(item["text"]) for item in queries]

    document_started = time.perf_counter()
    document_embeddings = encoder.encode(
        document_texts,
        batch_size=int(model_config["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(model_config["normalize_embeddings"]),
    )
    document_encoding_seconds = time.perf_counter() - document_started

    query_started = time.perf_counter()
    query_embeddings = encoder.encode(
        query_texts,
        batch_size=int(model_config["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(model_config["normalize_embeddings"]),
    )
    query_encoding_seconds = time.perf_counter() - query_started

    if document_embeddings.shape != (len(documents), int(model_config["embedding_dimension"])):
        raise ValueError("B1 document embedding shape drifted")
    if query_embeddings.shape != (len(queries), int(model_config["embedding_dimension"])):
        raise ValueError("B1 query embedding shape drifted")

    ranking_started = time.perf_counter()
    similarities = query_embeddings @ document_embeddings.T
    per_query: list[QueryMetrics] = []
    per_intent: dict[str, list[QueryMetrics]] = defaultdict(list)
    rank_digest = hashlib.sha256()
    top_scores: list[float] = []

    for row_index, query in enumerate(queries):
        query_id = str(query["query_id"])
        if query_id not in qrels:
            raise ValueError(f"B1 query has no qrels: {query_id}")
        ranking = _stable_rank(similarities[row_index], document_ids)
        _ranking_hash(query_id, ranking, rank_digest)
        metrics = evaluate_query(ranking, qrels[query_id])
        per_query.append(metrics)
        per_intent[str(query["intent"])].append(metrics)
        top_scores.append(float(np.max(similarities[row_index])))
    ranking_seconds = time.perf_counter() - ranking_started

    aggregate = mean_metrics(per_query)
    intent_metrics = {
        intent: mean_metrics(metrics) for intent, metrics in sorted(per_intent.items())
    }
    if len(intent_metrics) != 77:
        raise ValueError("B1 development result lost one or more intents")

    b0 = _load_object(B0_RESULT)
    b0_metrics = cast(dict[str, float], b0["metrics"])
    comparison = {key: aggregate[key] - b0_metrics[key] for key in aggregate}

    results: dict[str, object] = {
        "version": "retrieval-b1-development-v1",
        "status": "development_only",
        "model": "B1",
        "benchmark_version": benchmark["version"],
        "configuration_version": config["version"],
        "model_id": model_config["model_id"],
        "model_revision": model_config["revision"],
        "model_safetensors_sha256": weight_sha,
        "query_count": len(queries),
        "candidate_document_count": len(documents),
        "qrel_count": len(qrel_rows),
        "metrics": aggregate,
        "comparison_to_b0": comparison,
        "ranking_sha256": rank_digest.hexdigest(),
        "similarity_diagnostics": {
            "mean_top_score": statistics.fmean(top_scores),
            "min_top_score": min(top_scores),
            "max_top_score": max(top_scores),
            "interpretation": "ranking_diagnostic_not_probability",
        },
        "timing_seconds": {
            "model_load": model_load_seconds,
            "document_encoding": document_encoding_seconds,
            "query_encoding": query_encoding_seconds,
            "similarity_and_ranking": ranking_seconds,
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
        },
        "per_intent": intent_metrics,
        "confirmatory_partition_opened": False,
        "official_banking77_test_accessed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        (
            "# Phase 3 B1 Dense Retrieval Development Result",
            "",
            "> Development evidence only. The sealed Phase 3 confirmatory partition was not opened.",
            "",
            "| Metric | B0 | B1 | B1 - B0 |",
            "|---|---:|---:|---:|",
            *(
                f"| {key} | {b0_metrics[key]:.6f} | {aggregate[key]:.6f} | {comparison[key]:+.6f} |"
                for key in aggregate
            ),
            "",
            f"Ranking SHA-256: `{results['ranking_sha256']}`.",
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

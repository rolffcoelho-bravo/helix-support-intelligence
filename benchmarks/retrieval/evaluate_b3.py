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
"""Evaluate the frozen Phase 3 B3 cross-encoder reranker on development queries only."""

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
from sentence_transformers import CrossEncoder, SentenceTransformer

from helix_support_intelligence.data.banking77 import sha256_file
from helix_support_intelligence.retrieval.metrics import QueryMetrics, evaluate_query, mean_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = REPO_ROOT / "configs" / "retrieval" / "phase3_benchmark_v1.json"
B1_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b1_dense_v1.json"
B3_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b3_cross_encoder_v1.json"
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
            raise ValueError(f"B3 input hash drifted for {name}: {observed} != {digest}")
    for forbidden in ("confirmatory_queries.jsonl", "confirmatory_qrels.jsonl"):
        if (input_dir / forbidden).exists():
            raise ValueError(f"B3 development evaluator refuses sealed file: {forbidden}")


def _qrels_by_query(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        relevance = row["relevance"]
        if not isinstance(relevance, int) or isinstance(relevance, bool):
            raise TypeError("qrel relevance must be an integer")
        qrels[str(row["query_id"])][str(row["document_id"])] = relevance
    return dict(qrels)


def _verify_model_weights(model: Mapping[str, Any], *, label: str) -> str:
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
        raise ValueError(f"{label} model weight hash drifted: {observed} != {expected}")
    return observed


def _stable_dense_rank(scores: np.ndarray, document_ids: list[str]) -> list[str]:
    if scores.ndim != 1 or len(scores) != len(document_ids):
        raise ValueError("B1 score vector shape drifted")
    order = sorted(
        range(len(document_ids)),
        key=lambda index: (-float(scores[index]), document_ids[index]),
    )
    return [document_ids[index] for index in order]


def _ranking_hash(rankings: Mapping[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(rankings):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        for document_id in rankings[query_id]:
            digest.update(document_id.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _reconstruct_b1(
    documents: list[dict[str, object]],
    queries: list[dict[str, object]],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, float], str]:
    model = cast(dict[str, Any], config["model"])
    environment = cast(dict[str, Any], config["environment"])
    if str(model["device"]) != "cpu" or environment.get("cpu_only") is not True:
        raise ValueError("B1 parent reconstruction must remain CPU-only")

    weight_sha = _verify_model_weights(model, label="B1 parent")
    load_started = time.perf_counter()
    encoder = SentenceTransformer(
        str(model["model_id"]),
        revision=str(model["revision"]),
        device="cpu",
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    model_load_seconds = time.perf_counter() - load_started

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

    ranking_started = time.perf_counter()
    similarities = query_embeddings @ document_embeddings.T
    rankings = {
        str(query["query_id"]): _stable_dense_rank(similarities[row_index], document_ids)
        for row_index, query in enumerate(queries)
    }
    ranking_seconds = time.perf_counter() - ranking_started
    return (
        rankings,
        {
            "model_load": model_load_seconds,
            "document_encoding": document_seconds,
            "query_encoding": query_seconds,
            "similarity_and_ranking": ranking_seconds,
        },
        weight_sha,
    )


def _rerank(
    parent_rankings: Mapping[str, list[str]],
    documents: Mapping[str, str],
    queries: list[dict[str, object]],
    model: CrossEncoder,
    *,
    depth: int,
    batch_size: int,
) -> tuple[dict[str, list[str]], float, int]:
    started = time.perf_counter()
    output: dict[str, list[str]] = {}
    pair_count = 0

    for query in queries:
        query_id = str(query["query_id"])
        parent = parent_rankings[query_id]
        candidates = parent[:depth]
        pairs = [(str(query["text"]), documents[document_id]) for document_id in candidates]
        scores = np.asarray(
            model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                convert_to_numpy=True,
            )
        ).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError("B3 cross-encoder score count drifted")
        parent_position = {document_id: rank for rank, document_id in enumerate(candidates)}
        reranked = sorted(
            candidates,
            key=lambda document_id: (
                -float(scores[parent_position[document_id]]),
                parent_position[document_id],
                document_id,
            ),
        )
        output[query_id] = reranked + parent[depth:]
        pair_count += len(pairs)

    return output, time.perf_counter() - started, pair_count


def _per_intent_metrics(
    rankings: Mapping[str, list[str]],
    queries: list[dict[str, object]],
    qrels: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    query_by_id = {str(query["query_id"]): query for query in queries}
    per_query: list[QueryMetrics] = []
    grouped: dict[str, list[QueryMetrics]] = defaultdict(list)
    for query_id, ranking in rankings.items():
        metrics = evaluate_query(ranking, qrels[query_id])
        per_query.append(metrics)
        grouped[str(query_by_id[query_id]["intent"])].append(metrics)
    return (
        mean_metrics(per_query),
        {intent: mean_metrics(values) for intent, values in sorted(grouped.items())},
    )


def _comparison_counts(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, int]]:
    metrics = next(iter(candidate.values())).keys()
    output: dict[str, dict[str, int]] = {}
    for metric in metrics:
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


def _selection_verdict(
    candidate: Mapping[str, float],
    parent: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    evaluation = cast(dict[str, Any], config["evaluation"])
    guardrails = cast(dict[str, Any], evaluation["guardrails"])
    deltas = {key: candidate[key] - parent[key] for key in candidate}
    checks = {
        "material_ndcg_gain": deltas["ndcg_at_10"]
        >= float(evaluation["minimum_material_ndcg_gain"]),
        "mrr_non_decrease": deltas["mrr_at_10"] >= float(guardrails["mrr_at_10_delta_min"]),
        "success1_non_decrease": deltas["success_at_1"]
        >= float(guardrails["success_at_1_delta_min"]),
        "recall20_non_decrease": deltas["recall_at_20"]
        >= float(guardrails["recall_at_20_delta_min"]),
        "policy_recall20_non_decrease": deltas["citation_eligible_recall_at_20"]
        >= float(guardrails["citation_eligible_recall_at_20_delta_min"]),
        "recall50_equal_parent": abs(deltas["recall_at_50"]) <= 1e-15,
    }
    return ("selected" if all(checks.values()) else "rejected"), checks


def _render_report(results: Mapping[str, Any]) -> str:
    metrics = cast(dict[str, float], results["metrics"])
    parent = cast(dict[str, float], results["parent_metrics"])
    deltas = cast(dict[str, float], results["comparison_to_b1"])
    timing = cast(dict[str, Any], results["timing_seconds"])
    checks = cast(dict[str, bool], results["selection_checks"])
    verdict = str(results["selection_verdict"]).upper()
    rows = [
        ("nDCG@10", "ndcg_at_10"),
        ("MRR@10", "mrr_at_10"),
        ("Recall@20", "recall_at_20"),
        ("Recall@50", "recall_at_50"),
        ("Success@1", "success_at_1"),
        ("Governing-policy recall@20", "citation_eligible_recall_at_20"),
    ]
    table = "\n".join(
        f"| {label} | {parent[key]:.4f} | {metrics[key]:.4f} | {deltas[key]:+.4f} |"
        for label, key in rows
    )
    check_lines = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
    )
    return (
        "# Phase 3 B3 Cross-Encoder Development Evaluation\n\n"
        f"> **Registered verdict: {verdict}.** The retrieval confirmatory partition "
        "remained sealed.\n\n"
        "## Result\n\n"
        "| Metric | B1 | B3 | B3 - B1 |\n"
        "|---|---:|---:|---:|\n"
        f"{table}\n\n"
        "## Frozen method\n\n"
        f"- Model: `{results['model_id']}` at `{results['model_revision']}`.\n"
        f"- Parent: B1 top-{results['candidate_depth']} candidates.\n"
        "- Pair input: raw BANKING77 query + HelixBank title/newline/body.\n"
        "- Score: raw sequence-classification logit with identity activation.\n"
        "- Tail ranks outside the candidate pool remain in original B1 order.\n"
        "- No fine-tuning, model search, depth search, or post-result rescue is permitted.\n\n"
        "## Registered selection checks\n\n"
        f"{check_lines}\n\n"
        "## Timing\n\n"
        f"- Cross-encoder model load: {timing['cross_encoder_model_load']:.3f} s\n"
        f"- Reranking: {timing['reranking_total']:.3f} s\n"
        f"- Scored pairs: {results['pair_count']}\n"
        f"- Pairs/s: {timing['reranking_pairs_per_second']:.3f}\n"
        f"- Mean reranking time/query: {timing['reranking_seconds_per_query']:.6f} s\n\n"
        "Timing is descriptive GitHub-hosted CPU evidence only, not a production SLA.\n\n"
        "## Integrity\n\n"
        f"- B1 full-ranking hash observed: `{results['parent_ranking_sha256']}`.\n"
        f"- B3 full-ranking hash: `{results['ranking_sha256']}`.\n"
        f"- B1 model-weight SHA-256: `{results['parent_model_safetensors_sha256']}`.\n"
        f"- B3 model-weight SHA-256: `{results['model_safetensors_sha256']}`.\n"
        "- Confirmatory partition opened: false.\n"
        "- Official BANKING77 test accessed: false.\n"
    )


def evaluate(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run the frozen B3 development benchmark without opening confirmatory data."""

    benchmark = _load_object(BENCHMARK_CONFIG)
    b1_config = _load_object(B1_CONFIG)
    b3_config = _load_object(B3_CONFIG)
    b1_result = _load_object(B1_RESULT)

    if b3_config.get("status") != "frozen_before_first_score":
        raise ValueError("B3 configuration is not frozen before first score")
    if b3_config.get("benchmark_version") != benchmark.get("version"):
        raise ValueError("B3 benchmark version mismatch")
    _validate_inputs(input_dir, benchmark)

    documents = _load_jsonl(input_dir / "documents.jsonl")
    queries = _load_jsonl(input_dir / "development_queries.jsonl")
    qrel_rows = _load_jsonl(input_dir / "development_qrels.jsonl")
    qrels = _qrels_by_query(qrel_rows)
    expected_counts = cast(dict[str, Any], benchmark["selection"])["expected_counts"]
    if len(documents) != expected_counts["candidate_documents"]:
        raise ValueError("B3 candidate-document count drifted")
    if len(queries) != expected_counts["development_queries"]:
        raise ValueError("B3 development-query count drifted")
    if len(qrel_rows) != expected_counts["development_qrels"]:
        raise ValueError("B3 development-qrel count drifted")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    environment = cast(dict[str, Any], b3_config["environment"])
    torch.set_num_threads(int(environment["threads"]))
    torch.set_num_interop_threads(int(environment["threads"]))
    torch.use_deterministic_algorithms(True)

    parent_config = cast(dict[str, Any], b3_config["parent"])
    accepted_parent_hash = str(parent_config["accepted_full_ranking_sha256"])
    accepted_evidence = cast(dict[str, Any], b1_result["deterministic_evidence"])
    if accepted_parent_hash != str(accepted_evidence["accepted_ranking_sha256"]):
        raise ValueError("B3 parent hash disagrees with audited B1 result")

    parent_rankings, parent_timing, parent_weight_sha = _reconstruct_b1(
        documents,
        queries,
        b1_config,
    )
    observed_parent_hash = _ranking_hash(parent_rankings)
    if observed_parent_hash != accepted_parent_hash:
        raise ValueError(
            f"B1 parent ranking drifted before B3: {observed_parent_hash} != {accepted_parent_hash}"
        )

    model_config = cast(dict[str, Any], b3_config["model"])
    if str(model_config["device"]) != "cpu" or environment.get("cpu_only") is not True:
        raise ValueError("B3 development evaluation must remain CPU-only")
    weight_sha = _verify_model_weights(model_config, label="B3")

    load_started = time.perf_counter()
    reranker = CrossEncoder(
        str(model_config["model_id"]),
        revision=str(model_config["revision"]),
        max_length=int(model_config["max_length"]),
        device="cpu",
        trust_remote_code=bool(model_config["trust_remote_code"]),
    )
    model_load_seconds = time.perf_counter() - load_started

    document_text = {
        str(document["document_id"]): f"{document['title']}\n{document['body']}"
        for document in documents
    }
    depth = int(parent_config["candidate_depth"])
    reranked, rerank_seconds, pair_count = _rerank(
        parent_rankings,
        document_text,
        queries,
        reranker,
        depth=depth,
        batch_size=int(model_config["batch_size"]),
    )
    if pair_count != int(cast(dict[str, Any], b3_config["evaluation"])["latency"]["pair_count"]):
        raise ValueError("B3 pair count drifted")

    aggregate, per_intent = _per_intent_metrics(reranked, queries, qrels)
    if len(per_intent) != 77:
        raise ValueError("B3 development result lost one or more intents")
    parent_aggregate, parent_per_intent = _per_intent_metrics(parent_rankings, queries, qrels)

    recorded_parent = cast(dict[str, float], b1_result["metrics"])
    for key, value in recorded_parent.items():
        if abs(parent_aggregate[key] - value) > 1e-15:
            raise ValueError(f"B3 reconstructed B1 metric drifted for {key}")

    comparison = {key: aggregate[key] - parent_aggregate[key] for key in aggregate}
    verdict, checks = _selection_verdict(aggregate, parent_aggregate, b3_config)
    ranking_sha = _ranking_hash(reranked)

    results: dict[str, object] = {
        "version": "retrieval-b3-development-v1",
        "status": "development_only",
        "model": "B3",
        "benchmark_version": benchmark["version"],
        "configuration_version": b3_config["version"],
        "model_id": model_config["model_id"],
        "model_revision": model_config["revision"],
        "model_safetensors_sha256": weight_sha,
        "parent": "B1",
        "parent_ranking_sha256": observed_parent_hash,
        "parent_model_safetensors_sha256": parent_weight_sha,
        "candidate_depth": depth,
        "query_count": len(queries),
        "candidate_document_count": len(documents),
        "qrel_count": len(qrel_rows),
        "pair_count": pair_count,
        "metrics": aggregate,
        "parent_metrics": parent_aggregate,
        "comparison_to_b1": comparison,
        "per_intent": per_intent,
        "per_intent_comparison_counts": _comparison_counts(per_intent, parent_per_intent),
        "selection_verdict": verdict,
        "selection_checks": checks,
        "ranking_sha256": ranking_sha,
        "timing_seconds": {
            "parent_reconstruction": parent_timing,
            "cross_encoder_model_load": model_load_seconds,
            "reranking_total": rerank_seconds,
            "reranking_pairs_per_second": pair_count / rerank_seconds,
            "reranking_seconds_per_query": rerank_seconds / len(queries),
            "interpretation": "descriptive_ci_hardware_only_not_production_sla",
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
        "confirmatory_partition_opened": False,
        "official_banking77_test_accessed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(results), encoding="utf-8")
    print(json.dumps(results, sort_keys=True))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

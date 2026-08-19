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
"""Compute frozen B1 full- and top-50 ranking hashes without B3 scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from helix_support_intelligence.data.banking77 import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = REPO_ROOT / "configs" / "retrieval" / "phase3_benchmark_v1.json"
B1_CONFIG = REPO_ROOT / "configs" / "retrieval" / "b1_dense_v1.json"


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
            raise ValueError(f"B1 hash-audit input drifted for {name}: {observed} != {digest}")
    for forbidden in ("confirmatory_queries.jsonl", "confirmatory_qrels.jsonl"):
        if (input_dir / forbidden).exists():
            raise ValueError(f"B1 hash audit refuses sealed file: {forbidden}")


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
    order = sorted(
        range(len(document_ids)),
        key=lambda index: (-float(scores[index]), document_ids[index]),
    )
    return [document_ids[index] for index in order]


def _update_hash(query_id: str, document_ids: Iterable[str], digest: Any) -> None:
    digest.update(query_id.encode("utf-8"))
    digest.update(b"\0")
    for document_id in document_ids:
        digest.update(document_id.encode("utf-8"))
        digest.update(b"\0")


def audit(input_dir: Path) -> dict[str, object]:
    benchmark = _load_object(BENCHMARK_CONFIG)
    config = _load_object(B1_CONFIG)
    _validate_inputs(input_dir, benchmark)

    documents = _load_jsonl(input_dir / "documents.jsonl")
    queries = _load_jsonl(input_dir / "development_queries.jsonl")
    model = cast(dict[str, Any], config["model"])
    environment = cast(dict[str, Any], config["environment"])

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(int(environment["threads"]))
    torch.set_num_interop_threads(int(environment["threads"]))
    torch.use_deterministic_algorithms(True)
    weight_sha = _verify_model_weights(model)

    encoder = SentenceTransformer(
        str(model["model_id"]),
        revision=str(model["revision"]),
        device="cpu",
        trust_remote_code=bool(model["trust_remote_code"]),
    )
    ordered_documents = sorted(documents, key=lambda item: str(item["document_id"]))
    document_ids = [str(item["document_id"]) for item in ordered_documents]
    document_texts = [f"{item['title']}\n{item['body']}" for item in ordered_documents]
    query_instruction = str(model["query_instruction"])
    query_texts = [query_instruction + str(item["text"]) for item in queries]

    document_embeddings = encoder.encode(
        document_texts,
        batch_size=int(model["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(model["normalize_embeddings"]),
    )
    query_embeddings = encoder.encode(
        query_texts,
        batch_size=int(model["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(model["normalize_embeddings"]),
    )
    similarities = query_embeddings @ document_embeddings.T

    full_digest = hashlib.sha256()
    top50_digest = hashlib.sha256()
    for row_index, query in enumerate(queries):
        query_id = str(query["query_id"])
        ranking = _stable_rank(similarities[row_index], document_ids)
        _update_hash(query_id, ranking, full_digest)
        _update_hash(query_id, ranking[:50], top50_digest)

    return {
        "version": "retrieval-b1-top50-integrity-v1",
        "benchmark_version": benchmark["version"],
        "query_count": len(queries),
        "candidate_depth": 50,
        "full_ranking_sha256": full_digest.hexdigest(),
        "top50_ranking_sha256": top50_digest.hexdigest(),
        "model_safetensors_sha256": weight_sha,
        "confirmatory_partition_opened": False,
        "official_banking77_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

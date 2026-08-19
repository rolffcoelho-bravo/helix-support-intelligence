"""Materialize and hash the frozen Phase 3 natural-language retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

from helix_support_intelligence.data.banking77 import (
    Banking77Spec,
    canonical_jsonl_bytes as banking_jsonl_bytes,
    load_csv,
    sha256_file,
    split_training_examples,
)
from helix_support_intelligence.data.helixbank import generate_bundle
from helix_support_intelligence.retrieval.benchmark import (
    RetrievalBenchmarkSpec,
    build_manifest,
    build_qrels,
    canonical_jsonl_bytes,
    eligible_documents,
    select_queries,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANKING_CONFIG = REPO_ROOT / "configs" / "data" / "banking77.json"
DEFAULT_RETRIEVAL_CONFIG = REPO_ROOT / "configs" / "retrieval" / "phase3_benchmark_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "generated" / "retrieval" / "v1"


def _download_train_only(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "helix-support-intelligence/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def materialize(
    banking_config: Path,
    retrieval_config: Path,
    output_dir: Path | None,
) -> dict[str, object]:
    """Build the benchmark from pinned BANKING77 train bytes without touching official test."""

    banking = Banking77Spec.from_json(banking_config)
    retrieval = RetrievalBenchmarkSpec.from_json(retrieval_config)

    with tempfile.TemporaryDirectory(prefix="helix-phase3-retrieval-") as temp:
        train_csv = Path(temp) / "train.csv"
        _download_train_only(banking.train_url, train_csv)
        if sha256_file(train_csv) != banking.train_sha256:
            raise ValueError("BANKING77 train source checksum mismatch")
        source_train = load_csv(train_csv, "train")

    if len(source_train) != banking.train_examples:
        raise ValueError("BANKING77 train source row count drifted")
    if len({item.intent for item in source_train}) != banking.intent_count:
        raise ValueError("BANKING77 train intent cardinality drifted")

    fit_train, validation, quarantined = split_training_examples(source_train, banking)
    if len(fit_train) != banking.expected_counts["train"]:
        raise ValueError("frozen fit_train count drifted")
    if len(validation) != banking.expected_counts["validation"]:
        raise ValueError("frozen validation count drifted")
    if len(quarantined) != banking.expected_counts["quarantine"]:
        raise ValueError("frozen quarantine count drifted")

    fit_train_hash = __import__("hashlib").sha256(
        banking_jsonl_bytes(fit_train, banking.source_revision)
    ).hexdigest()
    if fit_train_hash != banking.expected_hashes["train"]:
        raise ValueError("frozen fit_train hash drifted")

    development, confirmatory = select_queries(fit_train, retrieval, banking.source_revision)
    bundle = generate_bundle()
    documents = eligible_documents(bundle.documents, retrieval)
    development_qrels = build_qrels(development, documents, retrieval)
    confirmatory_qrels = build_qrels(confirmatory, documents, retrieval)
    manifest = build_manifest(
        development,
        confirmatory,
        development_qrels,
        confirmatory_qrels,
        documents,
        retrieval,
    )
    manifest["source_train_sha256"] = banking.train_sha256
    manifest["fit_train_sha256"] = fit_train_hash
    manifest["official_test_accessed"] = False

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "documents.jsonl").write_bytes(canonical_jsonl_bytes(documents))
        (output_dir / "development_queries.jsonl").write_bytes(
            canonical_jsonl_bytes(item.as_record() for item in development)
        )
        (output_dir / "development_qrels.jsonl").write_bytes(
            canonical_jsonl_bytes(development_qrels)
        )
        # Confirmatory content remains unmaterialized by default. Only its hashes are frozen here.
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banking-config", type=Path, default=DEFAULT_BANKING_CONFIG)
    parser.add_argument("--retrieval-config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = materialize(
        args.banking_config.resolve(),
        args.retrieval_config.resolve(),
        args.output.resolve() if args.output is not None else None,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

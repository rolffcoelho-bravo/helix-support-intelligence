"""Tests for the frozen fictional HelixBank Policy Corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from helix_support_intelligence.data.helixbank import (
    CORPUS_VERSION,
    generate_bundle,
    manifest,
)


def test_frozen_manifest() -> None:
    observed = manifest()
    assert observed["corpus_version"] == CORPUS_VERSION
    assert observed["counts"] == {
        "documents": 154,
        "queries": 308,
        "judgments": 616,
        "intents": 77,
    }
    assert observed["sha256"] == {
        "documents": "13572a02eddadbd621a39490238358acccf8c5f01fc7a26dcfe3f4017aad2d8f",
        "queries": "2c54b5353d71a399cb15303fb6d751dd18c9ba814c992c4f676fb19bd865481c",
        "judgments": "8d5b4b816a807d5976d0102dcdd3d917a16c9c4a916e4776f5023b47043d5a4d",
    }


def test_corpus_contains_required_evidence_states() -> None:
    bundle = generate_bundle()
    assert any(doc["status"] == "archived" for doc in bundle.documents)
    assert any(bool(doc["conflict_fixture"]) for doc in bundle.documents)
    assert any(bool(doc["untrusted_content_fixture"]) for doc in bundle.documents)
    assert {str(query["case_type"]) for query in bundle.queries} >= {
        "answerable",
        "ambiguous",
        "outdated_evidence",
        "missing_evidence",
        "conflicting_evidence",
    }


def test_every_judgment_resolves() -> None:
    bundle = generate_bundle()
    query_ids = {str(item["query_id"]) for item in bundle.queries}
    document_ids = {str(item["document_id"]) for item in bundle.documents}
    assert all(str(item["query_id"]) in query_ids for item in bundle.judgments)
    assert all(str(item["document_id"]) in document_ids for item in bundle.judgments)


def test_committed_manifest_matches_generator() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "synthetic" / "helixbank-policy-v1" / "manifest.json"
    committed = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    generated = manifest()
    for key in ("corpus_version", "generator_version", "counts", "sha256"):
        assert committed[key] == generated[key]

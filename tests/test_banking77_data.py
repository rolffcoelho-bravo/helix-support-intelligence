"""Tests for deterministic BANKING77 Phase 1 primitives."""

from __future__ import annotations

from pathlib import Path

from helix_support_intelligence.data.banking77 import (
    Banking77Spec,
    BankingExample,
    canonical_jsonl_bytes,
    canonical_text,
    exact_cross_split_overlaps,
    sample_id,
    split_training_examples,
)


def _spec() -> Banking77Spec:
    return Banking77Spec(
        version="test",
        source_revision="a" * 40,
        train_url="https://example.invalid/train.csv",
        test_url="https://example.invalid/test.csv",
        train_sha256="b" * 64,
        test_sha256="c" * 64,
        train_examples=6,
        test_examples=2,
        intent_count=2,
        validation_fraction=0.5,
        split_salt="fixture-salt",
        quarantine_train_indices=(1,),
        expected_counts={"train": 0, "validation": 0, "test": 0, "quarantine": 0},
        expected_hashes={"train": "", "validation": "", "test": ""},
    )


def test_canonical_text_normalizes_without_rewriting_source() -> None:
    assert canonical_text("  CARD\u00a0Arrival  ") == "card arrival"


def test_sample_id_is_stable_and_revision_bound() -> None:
    row = BankingExample("train", 7, "Where is my card?", "card_arrival")
    first = sample_id(row, "a" * 40)
    assert first == sample_id(row, "a" * 40)
    assert first != sample_id(row, "b" * 40)
    assert len(first) == 24


def test_split_is_deterministic_stratified_and_honors_quarantine() -> None:
    rows = [
        BankingExample("train", 0, "a0", "alpha"),
        BankingExample("train", 1, "a1", "alpha"),
        BankingExample("train", 2, "a2", "alpha"),
        BankingExample("train", 3, "b0", "beta"),
        BankingExample("train", 4, "b1", "beta"),
        BankingExample("train", 5, "b2", "beta"),
    ]
    spec = _spec()
    first = split_training_examples(rows, spec)
    second = split_training_examples(rows, spec)

    assert first == second
    train, validation, quarantine = first
    assert [item.source_index for item in quarantine] == [1]
    assert {item.intent for item in train} == {"alpha", "beta"}
    assert {item.intent for item in validation} == {"alpha", "beta"}
    assert {item.source_index for item in train}.isdisjoint(
        {item.source_index for item in validation}
    )


def test_exact_overlap_uses_normalized_text() -> None:
    left = [BankingExample("train", 0, "Card   ARRIVAL", "alpha")]
    right = [BankingExample("test", 0, "card arrival", "alpha")]
    assert exact_cross_split_overlaps(left, right) == {"card arrival"}


def test_canonical_jsonl_is_byte_stable() -> None:
    row = BankingExample("train", 0, "Example", "alpha")
    payload = canonical_jsonl_bytes([row], "a" * 40)
    assert payload.endswith(b"\n")
    assert payload == canonical_jsonl_bytes([row], "a" * 40)


def test_repository_config_parses() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = Banking77Spec.from_json(root / "configs" / "data" / "banking77.json")
    assert spec.intent_count == 77
    assert len(spec.quarantine_train_indices) == 123
    assert spec.expected_counts["test"] == 3080

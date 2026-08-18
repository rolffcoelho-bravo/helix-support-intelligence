"""Deterministic BANKING77 preparation primitives for Helix Phase 1."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class BankingExample:
    """One row from the canonical BANKING77 source files."""

    source_split: str
    source_index: int
    text: str
    intent: str


@dataclass(frozen=True, slots=True)
class Banking77Spec:
    """Frozen source and split contract loaded from JSON configuration."""

    version: str
    source_revision: str
    train_url: str
    test_url: str
    train_sha256: str
    test_sha256: str
    train_examples: int
    test_examples: int
    intent_count: int
    validation_fraction: float
    split_salt: str
    quarantine_train_indices: tuple[int, ...]
    expected_counts: Mapping[str, int]
    expected_hashes: Mapping[str, str]

    @classmethod
    def from_json(cls, path: Path) -> Banking77Spec:
        """Load and type-check the Phase 1 BANKING77 contract."""

        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        source = _require_mapping(payload, "source")
        split = _require_mapping(payload, "split")
        expected = _require_mapping(payload, "expected")
        raw_sha = _require_mapping(source, "sha256")
        urls = _require_mapping(source, "urls")
        counts = _require_mapping(source, "examples")
        expected_counts = _require_mapping(expected, "counts")
        expected_hashes = _require_mapping(expected, "jsonl_sha256")
        quarantine = _require_int_list(split, "quarantine_train_indices")

        return cls(
            version=_require_str(payload, "version"),
            source_revision=_require_str(source, "revision"),
            train_url=_require_str(urls, "train"),
            test_url=_require_str(urls, "test"),
            train_sha256=_require_str(raw_sha, "train"),
            test_sha256=_require_str(raw_sha, "test"),
            train_examples=_require_int(counts, "train"),
            test_examples=_require_int(counts, "test"),
            intent_count=_require_int(source, "intent_count"),
            validation_fraction=_require_float(split, "validation_fraction"),
            split_salt=_require_str(split, "salt"),
            quarantine_train_indices=tuple(quarantine),
            expected_counts={
                key: _require_int(expected_counts, key)
                for key in ("train", "validation", "test", "quarantine")
            },
            expected_hashes={
                key: _require_str(expected_hashes, key) for key in ("train", "validation", "test")
            },
        )


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text(text: str) -> str:
    """Normalize text for exact cross-split leakage checks."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_csv(path: Path, source_split: str) -> list[BankingExample]:
    """Read the canonical PolyAI CSV shape."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"unexpected BANKING77 columns in {path}: {reader.fieldnames}")
        rows = [
            BankingExample(
                source_split=source_split,
                source_index=index,
                text=row["text"],
                intent=row["category"],
            )
            for index, row in enumerate(reader)
        ]
    return rows


def validate_raw_source(
    train_path: Path,
    test_path: Path,
    spec: Banking77Spec,
) -> tuple[list[BankingExample], list[BankingExample]]:
    """Verify pinned bytes, row counts, and label cardinality."""

    observed_hashes = {
        "train": sha256_file(train_path),
        "test": sha256_file(test_path),
    }
    expected_hashes = {
        "train": spec.train_sha256,
        "test": spec.test_sha256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            f"BANKING77 source checksum mismatch: {observed_hashes} != {expected_hashes}"
        )

    train = load_csv(train_path, "train")
    test = load_csv(test_path, "test")
    if len(train) != spec.train_examples or len(test) != spec.test_examples:
        raise ValueError("BANKING77 source row counts do not match the frozen contract")

    labels = {item.intent for item in train} | {item.intent for item in test}
    if len(labels) != spec.intent_count:
        raise ValueError("BANKING77 intent cardinality does not match the frozen contract")
    return train, test


def exact_cross_split_overlaps(
    left: Sequence[BankingExample],
    right: Sequence[BankingExample],
) -> set[str]:
    """Return normalized texts that occur in both sequences."""

    left_text = {canonical_text(item.text) for item in left}
    right_text = {canonical_text(item.text) for item in right}
    return left_text & right_text


def sample_id(example: BankingExample, source_revision: str) -> str:
    """Create a stable public sample identifier without embedding raw text."""

    payload = (
        f"{source_revision}\0{example.source_split}\0{example.source_index}\0"
        f"{example.intent}\0{example.text}"
    ).encode()
    return sha256_bytes(payload)[:24]


def _validation_count(size: int, fraction: float) -> int:
    if size < 2:
        raise ValueError("each label requires at least two train examples")
    rounded = int(size * fraction + 0.5)
    return max(1, min(size - 1, rounded))


def split_training_examples(
    train: Sequence[BankingExample],
    spec: Banking77Spec,
) -> tuple[list[BankingExample], list[BankingExample], list[BankingExample]]:
    """Apply the frozen leakage quarantine and stratified hash split."""

    quarantine = set(spec.quarantine_train_indices)
    quarantined = [item for item in train if item.source_index in quarantine]
    pool = [item for item in train if item.source_index not in quarantine]

    grouped: dict[str, list[tuple[str, BankingExample]]] = {}
    for item in pool:
        stable_id = sample_id(item, spec.source_revision)
        key = sha256_bytes(f"{spec.split_salt}\0{stable_id}".encode())
        grouped.setdefault(item.intent, []).append((key, item))

    validation_indices: set[int] = set()
    for items in grouped.values():
        ordered = sorted(items, key=lambda pair: (pair[0], pair[1].source_index))
        count = _validation_count(len(ordered), spec.validation_fraction)
        validation_indices.update(item.source_index for _, item in ordered[:count])

    fit_train = [item for item in pool if item.source_index not in validation_indices]
    validation = [item for item in pool if item.source_index in validation_indices]
    return fit_train, validation, quarantined


def public_record(example: BankingExample, source_revision: str) -> dict[str, object]:
    """Convert a source example to the stable generated JSONL representation."""

    return {
        "sample_id": sample_id(example, source_revision),
        "text": example.text,
        "intent": example.intent,
        "source_split": example.source_split,
        "source_index": example.source_index,
    }


def canonical_jsonl_bytes(
    examples: Iterable[BankingExample],
    source_revision: str,
) -> bytes:
    """Serialize examples deterministically for split hashing."""

    chunks: list[bytes] = []
    for example in examples:
        record = public_record(example, source_revision)
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        chunks.append((line + "\n").encode())
    return b"".join(chunks)


def verify_derived_contract(
    train: Sequence[BankingExample],
    validation: Sequence[BankingExample],
    test: Sequence[BankingExample],
    quarantined: Sequence[BankingExample],
    spec: Banking77Spec,
) -> None:
    """Check counts, hashes, label coverage, and exact leakage after quarantine."""

    counts = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "quarantine": len(quarantined),
    }
    if counts != dict(spec.expected_counts):
        raise ValueError(f"derived split counts changed: {counts}")

    hashes = {
        "train": sha256_bytes(canonical_jsonl_bytes(train, spec.source_revision)),
        "validation": sha256_bytes(canonical_jsonl_bytes(validation, spec.source_revision)),
        "test": sha256_bytes(canonical_jsonl_bytes(test, spec.source_revision)),
    }
    if hashes != dict(spec.expected_hashes):
        raise ValueError(f"derived split hashes changed: {hashes}")

    for name, examples in (("train", train), ("validation", validation), ("test", test)):
        if len({item.intent for item in examples}) != spec.intent_count:
            raise ValueError(f"{name} split lost one or more intents")

    fit = [*train, *validation]
    overlap = exact_cross_split_overlaps(fit, test)
    if overlap:
        raise ValueError(f"exact normalized leakage remains after quarantine: {len(overlap)}")


def write_jsonl(
    path: Path,
    examples: Iterable[BankingExample],
    source_revision: str,
) -> None:
    """Write one deterministic generated split."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_jsonl_bytes(examples, source_revision))


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return cast(Mapping[str, Any], value)


def _require_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _require_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _require_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _require_int_list(mapping: Mapping[str, Any], key: str) -> list[int]:
    value = mapping.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{key} must be a list of integers")
    return cast(list[int], value)

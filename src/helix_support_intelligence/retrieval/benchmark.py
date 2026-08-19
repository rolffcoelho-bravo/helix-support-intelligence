"""Deterministic natural-language retrieval benchmark construction for Phase 3."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.banking77 import BankingExample, sample_id


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkSpec:
    """Frozen Phase 3 retrieval-benchmark construction contract."""

    version: str
    salt: str
    development_per_intent: int
    confirmatory_per_intent: int
    intent_count: int
    expected_development_queries: int
    expected_confirmatory_queries: int
    corpus_version: str
    required_status: str
    required_permission: str
    policy_relevance: int
    faq_relevance: int

    @classmethod
    def from_json(cls, path: Path) -> RetrievalBenchmarkSpec:
        """Load and type-check the retrieval benchmark protocol."""

        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        source = _require_mapping(payload, "source")
        if source.get("raw_split") != "train":
            raise ValueError("Phase 3 retrieval benchmark must use BANKING77 source train only")
        if source.get("derived_partition") != "fit_train_only":
            raise ValueError("Phase 3 retrieval benchmark must use the frozen fit_train partition")
        if source.get("official_test_access_allowed") is not False:
            raise ValueError("official BANKING77 test access must remain disabled")

        selection = _require_mapping(payload, "selection")
        expected = _require_mapping(selection, "expected_counts")
        documents = _require_mapping(payload, "documents")
        eligibility = _require_mapping(documents, "eligibility")
        qrels = _require_mapping(payload, "qrels")

        spec = cls(
            version=_require_str(payload, "version"),
            salt=_require_str(selection, "salt"),
            development_per_intent=_require_int(selection, "development_per_intent"),
            confirmatory_per_intent=_require_int(selection, "confirmatory_per_intent"),
            intent_count=_require_int(selection, "intent_count"),
            expected_development_queries=_require_int(expected, "development_queries"),
            expected_confirmatory_queries=_require_int(expected, "confirmatory_queries"),
            corpus_version=_require_str(documents, "corpus_version"),
            required_status=_require_str(eligibility, "status"),
            required_permission=_require_str(eligibility, "permission"),
            policy_relevance=_require_int(qrels, "governing_policy_relevance"),
            faq_relevance=_require_int(qrels, "current_faq_relevance"),
        )
        if spec.development_per_intent <= 0 or spec.confirmatory_per_intent <= 0:
            raise ValueError("retrieval query counts per intent must be positive")
        if spec.intent_count <= 1:
            raise ValueError("retrieval benchmark requires multiple intents")
        return spec


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One immutable natural-language retrieval query."""

    query_id: str
    source_sample_id: str
    intent: str
    text: str

    def as_record(self) -> dict[str, object]:
        """Return the canonical public record."""

        return {
            "query_id": self.query_id,
            "source_sample_id": self.source_sample_id,
            "intent": self.intent,
            "text": self.text,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_jsonl_bytes(records: Iterable[Mapping[str, object]]) -> bytes:
    """Serialize arbitrary retrieval records deterministically."""

    chunks: list[bytes] = []
    for record in records:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        chunks.append((line + "\n").encode("utf-8"))
    return b"".join(chunks)


def select_queries(
    fit_train: Sequence[BankingExample],
    spec: RetrievalBenchmarkSpec,
    source_revision: str,
) -> tuple[tuple[RetrievalQuery, ...], tuple[RetrievalQuery, ...]]:
    """Select deterministic development and sealed confirmatory retrieval queries."""

    grouped: dict[str, list[tuple[str, BankingExample]]] = {}
    for item in fit_train:
        stable_id = sample_id(item, source_revision)
        key = _sha256_bytes(f"{spec.salt}\0{stable_id}".encode())
        grouped.setdefault(item.intent, []).append((key, item))

    if len(grouped) != spec.intent_count:
        raise ValueError(f"retrieval source lost intents: {len(grouped)} != {spec.intent_count}")

    development: list[RetrievalQuery] = []
    confirmatory: list[RetrievalQuery] = []
    required = spec.development_per_intent + spec.confirmatory_per_intent

    for intent in sorted(grouped):
        ordered = sorted(grouped[intent], key=lambda pair: (pair[0], pair[1].source_index))
        if len(ordered) < required:
            raise ValueError(f"intent {intent!r} has only {len(ordered)} fit-train rows; need {required}")

        selected = ordered[:required]
        for position, (_, item) in enumerate(selected):
            stable_id = sample_id(item, source_revision)
            query = RetrievalQuery(
                query_id=f"R-{stable_id}",
                source_sample_id=stable_id,
                intent=item.intent,
                text=item.text,
            )
            if position < spec.development_per_intent:
                development.append(query)
            else:
                confirmatory.append(query)

    development.sort(key=lambda item: item.query_id)
    confirmatory.sort(key=lambda item: item.query_id)

    if len(development) != spec.expected_development_queries:
        raise ValueError("development retrieval query count drifted")
    if len(confirmatory) != spec.expected_confirmatory_queries:
        raise ValueError("confirmatory retrieval query count drifted")
    if {item.query_id for item in development} & {item.query_id for item in confirmatory}:
        raise ValueError("retrieval development/confirmatory query IDs overlap")
    return tuple(development), tuple(confirmatory)


def eligible_documents(
    documents: Sequence[Mapping[str, object]],
    spec: RetrievalBenchmarkSpec,
) -> tuple[dict[str, object], ...]:
    """Filter the frozen HelixBank corpus before retrieval scoring."""

    eligible: list[dict[str, object]] = []
    for document in documents:
        if document.get("corpus_version") != spec.corpus_version:
            continue
        if document.get("status") != spec.required_status:
            continue
        if document.get("permission") != spec.required_permission:
            continue
        eligible.append(dict(document))
    eligible.sort(key=lambda item: str(item["document_id"]))
    if not eligible:
        raise ValueError("retrieval candidate set is empty")
    return tuple(eligible)


def _intent_document_ids(documents: Sequence[Mapping[str, object]]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for document in documents:
        intent = str(document["intent"])
        kind = str(document["kind"])
        document_id = str(document["document_id"])
        mapping.setdefault(intent, {})[kind] = document_id
    return mapping


def build_qrels(
    queries: Sequence[RetrievalQuery],
    documents: Sequence[Mapping[str, object]],
    spec: RetrievalBenchmarkSpec,
) -> tuple[dict[str, object], ...]:
    """Build graded qrels against the already-frozen policy corpus."""

    by_intent = _intent_document_ids(documents)
    qrels: list[dict[str, object]] = []
    for query in queries:
        targets = by_intent.get(query.intent)
        if targets is None or "policy" not in targets:
            raise ValueError(f"no eligible governing policy for intent {query.intent!r}")
        qrels.append(
            {
                "query_id": query.query_id,
                "document_id": targets["policy"],
                "relevance": spec.policy_relevance,
            }
        )
        faq_id = targets.get("faq")
        if faq_id is not None:
            qrels.append(
                {
                    "query_id": query.query_id,
                    "document_id": faq_id,
                    "relevance": spec.faq_relevance,
                }
            )
    qrels.sort(key=lambda item: (str(item["query_id"]), str(item["document_id"])))
    return tuple(qrels)


def build_manifest(
    development: Sequence[RetrievalQuery],
    confirmatory: Sequence[RetrievalQuery],
    development_qrels: Sequence[Mapping[str, object]],
    confirmatory_qrels: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, object]],
    spec: RetrievalBenchmarkSpec,
) -> dict[str, object]:
    """Return the deterministic hash manifest that must be frozen before scoring."""

    development_records = tuple(item.as_record() for item in development)
    confirmatory_records = tuple(item.as_record() for item in confirmatory)
    manifest = {
        "version": spec.version,
        "candidate_documents": len(documents),
        "development_queries": len(development),
        "confirmatory_queries": len(confirmatory),
        "development_qrels": len(development_qrels),
        "confirmatory_qrels": len(confirmatory_qrels),
        "sha256": {
            "candidate_documents": _sha256_bytes(canonical_jsonl_bytes(documents)),
            "development_queries": _sha256_bytes(canonical_jsonl_bytes(development_records)),
            "confirmatory_queries": _sha256_bytes(canonical_jsonl_bytes(confirmatory_records)),
            "development_qrels": _sha256_bytes(canonical_jsonl_bytes(development_qrels)),
            "confirmatory_qrels": _sha256_bytes(canonical_jsonl_bytes(confirmatory_qrels)),
        },
        "confirmatory_content_logged": False,
    }
    return manifest


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

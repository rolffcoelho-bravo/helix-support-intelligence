"""Deterministic retrieval primitives for the frozen Phase 3 ladder.

The module intentionally has no model-library dependency. Dense encoders and cross-encoders
are injected behind typed protocols so the core package remains lightweight while benchmark
execution can bind the exact model revisions declared by the Phase 3 protocol.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from random import Random
from typing import Protocol

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class Document:
    """Retrieval document with the metadata required by the evidence filter."""

    document_id: str
    title: str
    body: str
    status: str
    valid_from: date
    valid_to: date | None
    permission: str
    audience: str
    jurisdiction: str
    conflict_fixture: bool = False
    untrusted_content_fixture: bool = False

    @property
    def index_text(self) -> str:
        """Return the frozen Phase 3 text representation."""
        return f"{self.title}\n{self.body}"


@dataclass(frozen=True, slots=True)
class EligibilityPolicy:
    """Pre-ranking document eligibility contract."""

    evaluation_date: date
    statuses: frozenset[str]
    permissions: frozenset[str]
    audiences: frozenset[str]
    jurisdictions: frozenset[str]


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """One ranked retrieval result."""

    document_id: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Paired-bootstrap point estimate and percentile interval."""

    point_estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Deterministic summary of already-measured per-query latencies."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    queries_per_second: float


class EmbeddingEncoder(Protocol):
    """Dependency-injection boundary for a pinned dense encoder."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode texts into fixed-length vectors."""
        ...


class PairScorer(Protocol):
    """Dependency-injection boundary for a pinned cross-encoder."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Score query-document text pairs."""
        ...


def _required_str(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_date(value: object, key: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an ISO date string or null")
    return date.fromisoformat(value)


def _required_bool(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def document_from_record(record: Mapping[str, object]) -> Document:
    """Convert a policy-corpus record into a typed retrieval document."""
    return Document(
        document_id=_required_str(record, "document_id"),
        title=_required_str(record, "title"),
        body=_required_str(record, "body"),
        status=_required_str(record, "status"),
        valid_from=date.fromisoformat(_required_str(record, "valid_from")),
        valid_to=_optional_date(record.get("valid_to"), "valid_to"),
        permission=_required_str(record, "permission"),
        audience=_required_str(record, "audience"),
        jurisdiction=_required_str(record, "jurisdiction"),
        conflict_fixture=_required_bool(record, "conflict_fixture"),
        untrusted_content_fixture=_required_bool(record, "untrusted_content_fixture"),
    )


def is_eligible(document: Document, policy: EligibilityPolicy) -> bool:
    """Apply the frozen evidence filter without using query-derived metadata."""
    if document.status not in policy.statuses:
        return False
    if document.permission not in policy.permissions:
        return False
    if document.audience not in policy.audiences:
        return False
    if document.jurisdiction not in policy.jurisdictions:
        return False
    if document.valid_from > policy.evaluation_date:
        return False
    return document.valid_to is None or document.valid_to >= policy.evaluation_date


def filter_eligible_documents(
    documents: Sequence[Document], policy: EligibilityPolicy
) -> tuple[Document, ...]:
    """Return eligible documents in deterministic document-id order."""
    return tuple(
        sorted(
            (doc for doc in documents if is_eligible(doc, policy)),
            key=lambda d: d.document_id,
        )
    )


def tokenize(text: str) -> tuple[str, ...]:
    """Apply the frozen BM25 normalization and tokenization contract."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return tuple(_TOKEN_RE.findall(normalized))


class BM25Retriever:
    """Repository-owned deterministic BM25 implementation for B0."""

    def __init__(self, documents: Sequence[Document], *, k1: float = 1.2, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._documents = tuple(sorted(documents, key=lambda d: d.document_id))
        self._k1 = k1
        self._b = b
        self._tokens = tuple(tokenize(document.index_text) for document in self._documents)
        self._term_counts = tuple(Counter(tokens) for tokens in self._tokens)
        self._lengths = tuple(len(tokens) for tokens in self._tokens)
        self._average_length = statistics.fmean(self._lengths)

        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        self._document_frequency = document_frequency

    def _idf(self, term: str) -> float:
        """Use the Robertson-style positive BM25 IDF used by BM25Okapi-like systems."""
        document_count = len(self._documents)
        frequency = self._document_frequency.get(term, 0)
        return math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, *, k: int = 50) -> tuple[RankedDocument, ...]:
        """Rank documents by BM25 score with document-id tie-breaking."""
        if k <= 0:
            raise ValueError("k must be positive")
        query_tokens = tokenize(query)
        scores: list[tuple[str, float]] = []
        for document, counts, length in zip(
            self._documents, self._term_counts, self._lengths, strict=True
        ):
            score = 0.0
            for term in query_tokens:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                length_ratio = length / self._average_length if self._average_length else 0.0
                denominator = frequency + self._k1 * (1.0 - self._b + self._b * length_ratio)
                score += self._idf(term) * (frequency * (self._k1 + 1.0)) / denominator
            scores.append((document.document_id, score))

        ordered = sorted(scores, key=lambda item: (-item[1], item[0]))[:k]
        return tuple(
            RankedDocument(document_id=document_id, rank=index + 1, score=score)
            for index, (document_id, score) in enumerate(ordered)
        )


def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values:
        raise ValueError("embedding vectors cannot be empty")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("embedding vectors cannot have zero norm")
    return tuple(value / norm for value in values)


class DenseRetriever:
    """B1 dense retriever with injected encoder and deterministic cosine ranking."""

    def __init__(self, documents: Sequence[Document], encoder: EmbeddingEncoder) -> None:
        if not documents:
            raise ValueError("dense retrieval requires at least one document")
        self._documents = tuple(sorted(documents, key=lambda d: d.document_id))
        encoded = encoder.encode([document.index_text for document in self._documents])
        if len(encoded) != len(self._documents):
            raise ValueError("encoder returned an unexpected document-vector count")
        self._document_vectors = tuple(_normalize_vector(vector) for vector in encoded)
        dimensions = {len(vector) for vector in self._document_vectors}
        if len(dimensions) != 1:
            raise ValueError("document embeddings must share one dimension")
        self._dimension = next(iter(dimensions))
        self._encoder = encoder

    def search(self, query: str, *, k: int = 50) -> tuple[RankedDocument, ...]:
        """Rank documents by cosine similarity with document-id tie-breaking."""
        if k <= 0:
            raise ValueError("k must be positive")
        encoded = self._encoder.encode([query])
        if len(encoded) != 1:
            raise ValueError("encoder must return exactly one query vector")
        query_vector = _normalize_vector(encoded[0])
        if len(query_vector) != self._dimension:
            raise ValueError("query and document embeddings must share one dimension")

        scores = [
            (document.document_id, sum(q * d for q, d in zip(query_vector, vector, strict=True)))
            for document, vector in zip(self._documents, self._document_vectors, strict=True)
        ]
        ordered = sorted(scores, key=lambda item: (-item[1], item[0]))[:k]
        return tuple(
            RankedDocument(document_id=document_id, rank=index + 1, score=score)
            for index, (document_id, score) in enumerate(ordered)
        )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedDocument]],
    *,
    rrf_k: int = 60,
    source_depth: int = 50,
    retrieve_k: int = 50,
) -> tuple[RankedDocument, ...]:
    """Fuse source rankings using the frozen B2 reciprocal-rank rule."""
    if rrf_k <= 0 or source_depth <= 0 or retrieve_k <= 0:
        raise ValueError("RRF parameters must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for item in ranking:
            if item.rank > source_depth:
                continue
            if item.document_id in seen:
                raise ValueError("a source ranking cannot contain duplicate document ids")
            seen.add(item.document_id)
            scores[item.document_id] = scores.get(item.document_id, 0.0) + 1.0 / (
                rrf_k + item.rank
            )

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:retrieve_k]
    return tuple(
        RankedDocument(document_id=document_id, rank=index + 1, score=score)
        for index, (document_id, score) in enumerate(ordered)
    )


def rerank_top(
    query: str,
    base_ranking: Sequence[RankedDocument],
    documents: Mapping[str, Document],
    scorer: PairScorer,
    *,
    depth: int = 20,
    retrieve_k: int = 50,
) -> tuple[RankedDocument, ...]:
    """Apply B3 reranking to the head and append the untouched B2 tail."""
    if depth <= 0 or retrieve_k <= 0:
        raise ValueError("reranking depth and retrieve_k must be positive")
    bounded = tuple(base_ranking[:retrieve_k])
    head = bounded[:depth]
    tail = bounded[depth:]
    pairs: list[tuple[str, str]] = []
    for item in head:
        try:
            document = documents[item.document_id]
        except KeyError as exc:
            raise ValueError(f"missing document for reranking: {item.document_id}") from exc
        pairs.append((query, document.index_text))

    head_scores = scorer.score(pairs)
    if len(head_scores) != len(head):
        raise ValueError("reranker returned an unexpected score count")
    scored_head = sorted(
        zip(head, head_scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].document_id),
    )

    ordered: list[tuple[str, float]] = [
        (item.document_id, float(score)) for item, score in scored_head
    ]
    ordered.extend((item.document_id, item.score) for item in tail)
    return tuple(
        RankedDocument(document_id=document_id, rank=index + 1, score=score)
        for index, (document_id, score) in enumerate(ordered)
    )


def ndcg_at_k(ranking: Sequence[RankedDocument], qrels: Mapping[str, int], *, k: int = 10) -> float:
    """Compute graded nDCG with gain 2^relevance - 1."""
    if k <= 0:
        raise ValueError("k must be positive")

    def gain(relevance: int) -> float:
        return float((2**relevance) - 1)

    dcg = 0.0
    for index, item in enumerate(ranking[:k]):
        relevance = qrels.get(item.document_id, 0)
        dcg += gain(relevance) / math.log2(index + 2.0)

    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(gain(relevance) / math.log2(index + 2.0) for index, relevance in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(
    ranking: Sequence[RankedDocument],
    qrels: Mapping[str, int],
    *,
    k: int = 10,
    relevance_threshold: int = 2,
) -> float:
    """Return reciprocal rank of the first directly relevant eligible document."""
    if k <= 0:
        raise ValueError("k must be positive")
    for index, item in enumerate(ranking[:k]):
        if qrels.get(item.document_id, 0) >= relevance_threshold:
            return 1.0 / (index + 1)
    return 0.0


def recall_at_k(
    ranking: Sequence[RankedDocument],
    qrels: Mapping[str, int],
    *,
    k: int,
    relevance_threshold: int = 2,
) -> float | None:
    """Return binary recall, or None when a query has no eligible relevant item."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = {document_id for document_id, grade in qrels.items() if grade >= relevance_threshold}
    if not relevant:
        return None
    retrieved = {item.document_id for item in ranking[:k]}
    return len(relevant & retrieved) / len(relevant)


def macro_average(values: Sequence[float | None]) -> tuple[float, int]:
    """Average applicable values and report their count."""
    applicable = [value for value in values if value is not None]
    if not applicable:
        return 0.0, 0
    return statistics.fmean(applicable), len(applicable)


def _percentile(values: Sequence[float], probability: float) -> float:
    """Linear percentile interpolation with endpoints included."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def paired_bootstrap_difference(
    candidate: Sequence[float],
    comparator: Sequence[float],
    *,
    replicates: int = 5000,
    seed: int = 20260819,
) -> BootstrapInterval:
    """Compute a paired nonparametric percentile interval for candidate minus comparator."""
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError("paired bootstrap requires equal non-empty vectors")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    differences = [left - right for left, right in zip(candidate, comparator, strict=True)]
    point_estimate = statistics.fmean(differences)
    rng = Random(seed)
    sample_size = len(differences)
    bootstrap_means: list[float] = []
    for _ in range(replicates):
        bootstrap_means.append(
            statistics.fmean(differences[rng.randrange(sample_size)] for _ in range(sample_size))
        )
    return BootstrapInterval(
        point_estimate=point_estimate,
        lower=_percentile(bootstrap_means, 0.025),
        upper=_percentile(bootstrap_means, 0.975),
        replicates=replicates,
        seed=seed,
    )


def summarize_latency(samples_ms: Sequence[float]) -> LatencySummary:
    """Summarize positive per-query latency measurements without performing timing itself."""
    if not samples_ms or any(sample <= 0 for sample in samples_ms):
        raise ValueError("latency samples must be non-empty and positive")
    mean_ms = statistics.fmean(samples_ms)
    return LatencySummary(
        mean_ms=mean_ms,
        p50_ms=_percentile(samples_ms, 0.50),
        p95_ms=_percentile(samples_ms, 0.95),
        p99_ms=_percentile(samples_ms, 0.99),
        queries_per_second=1000.0 / mean_ms,
    )


def candidate_earns_complexity(
    *,
    delta_ndcg_at_10: float,
    ndcg_ci_lower: float,
    delta_mrr_at_10: float,
    candidate_p95_ms: float,
    latency_budget_ms: float,
    minimum_delta_ndcg_at_10: float = 0.01,
    minimum_delta_mrr_at_10: float = -0.005,
) -> bool:
    """Apply the frozen Phase 3 complexity-adoption gate."""
    return (
        delta_ndcg_at_10 >= minimum_delta_ndcg_at_10
        and ndcg_ci_lower > 0.0
        and delta_mrr_at_10 >= minimum_delta_mrr_at_10
        and candidate_p95_ms <= latency_budget_ms
    )

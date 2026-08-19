"""Dependency-light deterministic Okapi BM25 for the Phase 3 B0 baseline."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Apply the frozen B0 NFKC/casefold/alphanumeric tokenizer."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(TOKEN_PATTERN.findall(normalized))


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """One deterministic retrieval result."""

    document_id: str
    score: float


@dataclass(frozen=True, slots=True)
class BM25Index:
    """Immutable in-memory BM25 index over the frozen candidate corpus."""

    document_ids: tuple[str, ...]
    term_frequencies: tuple[Counter[str], ...]
    document_lengths: tuple[int, ...]
    document_frequency: Mapping[str, int]
    average_document_length: float
    k1: float
    b: float

    @classmethod
    def build(
        cls,
        documents: Sequence[Mapping[str, object]],
        *,
        k1: float,
        b: float,
    ) -> BM25Index:
        """Build the frozen title+body B0 index."""

        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be in [0, 1]")
        if not documents:
            raise ValueError("BM25 requires at least one document")

        ordered = sorted(documents, key=lambda item: str(item["document_id"]))
        document_ids: list[str] = []
        term_frequencies: list[Counter[str]] = []
        lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for document in ordered:
            document_id = str(document["document_id"])
            title = str(document.get("title", ""))
            body = str(document.get("body", ""))
            tokens = tokenize(f"{title}\n{body}")
            frequencies = Counter(tokens)

            document_ids.append(document_id)
            term_frequencies.append(frequencies)
            lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())

        average_length = sum(lengths) / len(lengths)
        if average_length <= 0:
            raise ValueError("BM25 candidate corpus cannot contain only empty documents")

        return cls(
            document_ids=tuple(document_ids),
            term_frequencies=tuple(term_frequencies),
            document_lengths=tuple(lengths),
            document_frequency=dict(document_frequency),
            average_document_length=average_length,
            k1=float(k1),
            b=float(b),
        )

    def _idf(self, term: str) -> float:
        document_count = len(self.document_ids)
        frequency = self.document_frequency.get(term, 0)
        return math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def score(self, query: str) -> tuple[RankedDocument, ...]:
        """Score every candidate and return deterministic descending BM25 rank."""

        query_terms = Counter(tokenize(query))
        scored: list[RankedDocument] = []
        for document_id, frequencies, length in zip(
            self.document_ids,
            self.term_frequencies,
            self.document_lengths,
            strict=True,
        ):
            score = 0.0
            length_normalizer = 1.0 - self.b + self.b * length / self.average_document_length
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * length_normalizer
                score += query_frequency * self._idf(term) * numerator / denominator
            scored.append(RankedDocument(document_id=document_id, score=score))

        scored.sort(key=lambda item: (-item.score, item.document_id))
        return tuple(scored)

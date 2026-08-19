"""Deterministic rank-only fusion utilities for Phase 3 retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
    weights: Mapping[str, float],
    rank_depth: int,
) -> list[str]:
    """Fuse complete ranked lists with weighted Reciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("RRF k must be positive")
    if rank_depth <= 0:
        raise ValueError("RRF rank_depth must be positive")
    if set(rankings) != set(weights):
        raise ValueError("RRF rankings and weights must use the same systems")
    if not rankings:
        raise ValueError("RRF requires at least one ranking")

    universe: set[str] | None = None
    for system, ranking in rankings.items():
        if len(ranking) < rank_depth:
            raise ValueError(f"{system} ranking shorter than frozen rank_depth")
        truncated = list(ranking[:rank_depth])
        if len(set(truncated)) != len(truncated):
            raise ValueError(f"{system} ranking contains duplicate document IDs")
        current = set(truncated)
        universe = current if universe is None else universe & current

    if universe is None or len(universe) != rank_depth:
        raise ValueError("RRF parent rankings must cover the same frozen document universe")

    scores = {document_id: 0.0 for document_id in universe}
    for system, ranking in rankings.items():
        weight = float(weights[system])
        if weight <= 0.0:
            raise ValueError("RRF weights must be positive")
        for rank, document_id in enumerate(ranking[:rank_depth], start=1):
            scores[document_id] += weight / float(k + rank)

    return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))

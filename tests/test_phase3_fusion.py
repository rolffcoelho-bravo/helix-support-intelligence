from __future__ import annotations

import pytest

from helix_support_intelligence.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_uses_one_based_reciprocal_ranks_and_equal_weights() -> None:
    fused = reciprocal_rank_fusion(
        {
            "lexical": ["a", "b", "c"],
            "dense": ["b", "c", "a"],
        },
        k=60,
        weights={"lexical": 1.0, "dense": 1.0},
        rank_depth=3,
    )

    assert fused == ["b", "a", "c"]


def test_rrf_breaks_exact_score_ties_by_document_id() -> None:
    fused = reciprocal_rank_fusion(
        {
            "left": ["b", "a"],
            "right": ["a", "b"],
        },
        k=60,
        weights={"left": 1.0, "right": 1.0},
        rank_depth=2,
    )

    assert fused == ["a", "b"]


def test_rrf_rejects_parent_universe_drift() -> None:
    with pytest.raises(ValueError, match="same frozen document universe"):
        reciprocal_rank_fusion(
            {
                "left": ["a", "b"],
                "right": ["a", "c"],
            },
            k=60,
            weights={"left": 1.0, "right": 1.0},
            rank_depth=2,
        )

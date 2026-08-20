# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.28,<1",
#   "huggingface-hub>=0.34,<1",
#   "numpy>=2.1,<3",
#   "onnxruntime>=1.22,<2",
#   "sentencepiece>=0.2,<1",
#   "transformers>=4.55,<5",
# ]
# ///
"""Replay A4.2 verification with a diagnostic-only zero-denominator repair.

This file is post-result audit code. It never executes G0, G1, G2, OpenAI,
or either NLI model. The only repair is to encode undefined diagnostic slice
rates as null when a slice has no answerable or no nonanswerable observations.
Top-level registered metrics and inference are reconstructed from the immutable
raw A4.2 artifact without alteration.
"""

from __future__ import annotations

import statistics
from typing import Any

import verify_a42


def aggregate_with_na(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Match the frozen aggregate, using null for undefined slice-only rates."""
    total = len(rows)
    if total == 0:
        raise ValueError("aggregate_with_na requires at least one row")

    factual = sum(int(row["factual_sentence_count"]) for row in rows)
    unsupported = sum(int(row["unsupported_sentence_count"]) for row in rows)
    cited = sum(len(row["citations"]) for row in rows)
    valid = sum(len(row["citations"]) * float(row["citation_precision"]) for row in rows)
    stale = sum(len(row["citations"]) * float(row["stale_citation_rate"]) for row in rows)
    applicable = [row for row in rows if row["citation_completeness"] is not None]
    gold = sum(len(row["gold_citations"]) for row in applicable)
    recovered = sum(
        len(row["gold_citations"]) * float(row["citation_completeness"]) for row in applicable
    )
    precision = valid / cited if cited else 1.0
    completeness = recovered / gold if gold else 1.0
    f1 = (
        2.0 * precision * completeness / (precision + completeness)
        if precision + completeness
        else 0.0
    )
    nonanswerable = [row for row in rows if row["expected_decision"] != "ANSWER_WITH_EVIDENCE"]
    answerable = [row for row in rows if row["expected_decision"] == "ANSWER_WITH_EVIDENCE"]
    costs = [
        float(row["estimated_cost_usd"]) for row in rows if row["estimated_cost_usd"] is not None
    ]

    return {
        "strict_grounded_success_rate": sum(bool(row["strict_grounded_success"]) for row in rows)
        / total,
        "decision_exact_match": sum(row["decision"] == row["expected_decision"] for row in rows)
        / total,
        "unsupported_sentence_rate": unsupported / factual if factual else 0.0,
        "answer_level_unsupported_rate": sum(
            int(row["unsupported_sentence_count"]) > 0 for row in rows
        )
        / total,
        "citation_precision": precision,
        "citation_completeness": completeness,
        "citation_f1": f1,
        "stale_citation_rate": stale / cited if cited else 0.0,
        "unsafe_answer_rate_on_nonanswerable": (
            sum(row["decision"] == "ANSWER_WITH_EVIDENCE" for row in nonanswerable)
            / len(nonanswerable)
            if nonanswerable
            else None
        ),
        "over_abstention_rate_on_answerable": (
            sum(row["decision"] != "ANSWER_WITH_EVIDENCE" for row in answerable) / len(answerable)
            if answerable
            else None
        ),
        "schema_valid_rate": sum(bool(row["schema_valid"]) for row in rows) / total,
        "provider_failure_rate": sum(row["failure"] is not None for row in rows) / total,
        "mean_estimated_cost_usd": statistics.fmean(costs) if costs else 0.0,
        "max_estimated_cost_usd": max(costs) if costs else 0.0,
        "total_estimated_cost_usd": sum(costs),
    }


verify_a42.aggregate = aggregate_with_na


if __name__ == "__main__":
    verify_a42.main()

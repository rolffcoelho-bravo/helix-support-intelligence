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
"""Registered A4.2 entry point with frozen evaluator/runtime adapters."""

from __future__ import annotations

from typing import Any

import evaluate_a42
from nli_batching_a42 import batched_support


_original_g0 = evaluate_a42.g0


def g0_with_inline_sentence_citations(
    query_text: str,
    documents: list[dict[str, object]],
) -> dict[str, Any]:
    """Keep G0 citations in the same punctuation-delimited factual sentence."""
    output = _original_g0(query_text, documents)
    if output["decision"] != "ANSWER_WITH_EVIDENCE":
        return output
    answer = str(output["answer"])
    base = answer.split(" [", 1)[0].rstrip().rstrip(".")
    inline = "".join(f" [{citation}]" for citation in output["citations"])
    output["answer"] = f"{base}{inline}."
    return output


evaluate_a42.g0 = g0_with_inline_sentence_citations
evaluate_a42.support = batched_support


if __name__ == "__main__":
    evaluate_a42.main()

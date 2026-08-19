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
"""Registered A4.2 entry point with the frozen batch-size-eight NLI adapter."""

from __future__ import annotations

import evaluate_a42
from nli_batching_a42 import batched_support


evaluate_a42.support = batched_support


if __name__ == "__main__":
    evaluate_a42.main()

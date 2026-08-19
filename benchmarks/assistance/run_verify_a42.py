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
"""Run the independent A4.2 verifier in the frozen A4.1 dependency environment."""

from __future__ import annotations

import verify_a42

if __name__ == "__main__":
    verify_a42.main()

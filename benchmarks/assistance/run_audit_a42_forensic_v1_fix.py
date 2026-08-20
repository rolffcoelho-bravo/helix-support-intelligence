"""Audit-only normalization repair for the A4.2 forensic replay."""

from __future__ import annotations

import re

import audit_a42_forensic_v1 as audit


def normalize_without_prepunctuation_space(text: str) -> str:
    """Collapse whitespace and remove citation-removal spaces before punctuation."""
    collapsed = " ".join(text.split())
    return re.sub(r"\s+([.!?])", r"\1", collapsed)


audit.normalize = normalize_without_prepunctuation_space


if __name__ == "__main__":
    audit.main()

"""Rebuild the repository-owned frozen HelixBank Policy Corpus v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from helix_support_intelligence.data.helixbank import (
    canonical_jsonl_bytes,
    generate_bundle,
    manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "generated" / "helixbank-policy-v1"


def materialize(output_dir: Path) -> None:
    """Write deterministic corpus files and their manifest."""

    bundle = generate_bundle()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "documents.jsonl").write_bytes(canonical_jsonl_bytes(bundle.documents))
    (output_dir / "queries.jsonl").write_bytes(canonical_jsonl_bytes(bundle.queries))
    (output_dir / "judgments.jsonl").write_bytes(canonical_jsonl_bytes(bundle.judgments))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"HelixBank Policy Corpus rebuilt at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    materialize(args.output.resolve())


if __name__ == "__main__":
    main()

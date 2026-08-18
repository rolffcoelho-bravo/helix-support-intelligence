"""Download, verify, and materialize the frozen BANKING77 Helix splits."""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from helix_support_intelligence.data.banking77 import (
    Banking77Spec,
    split_training_examples,
    validate_raw_source,
    verify_derived_contract,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "banking77.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "generated" / "banking77" / "v1"


def _download(url: str, destination: Path) -> None:
    """Download one pinned source file without following repository branch names."""

    request = urllib.request.Request(url, headers={"User-Agent": "helix-support-intelligence/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def materialize(config_path: Path, output_dir: Path) -> None:
    """Build deterministic train/validation/test JSONL after source verification."""

    spec = Banking77Spec.from_json(config_path)
    with tempfile.TemporaryDirectory(prefix="helix-banking77-") as temp:
        temp_root = Path(temp)
        train_csv = temp_root / "train.csv"
        test_csv = temp_root / "test.csv"
        _download(spec.train_url, train_csv)
        _download(spec.test_url, test_csv)
        source_train, source_test = validate_raw_source(train_csv, test_csv, spec)

    fit_train, validation, quarantined = split_training_examples(source_train, spec)
    verify_derived_contract(fit_train, validation, source_test, quarantined, spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", fit_train, spec.source_revision)
    write_jsonl(output_dir / "validation.jsonl", validation, spec.source_revision)
    write_jsonl(output_dir / "test.jsonl", source_test, spec.source_revision)

    quarantine_path = output_dir / "quarantine_source_indices.txt"
    quarantine_path.write_text(
        "\n".join(str(item.source_index) for item in quarantined) + "\n",
        encoding="utf-8",
    )
    print(
        "BANKING77 contract verified: "
        f"train={len(fit_train)}, validation={len(validation)}, "
        f"test={len(source_test)}, quarantine={len(quarantined)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    materialize(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

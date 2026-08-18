"""Tests for the public-material boundary."""

import fnmatch

from scripts.audit_publication import PROHIBITED_PATH_PATTERNS, SECRET_PATTERNS


def test_private_filename_patterns_are_blocked() -> None:
    private_paths = (
        "research_blueprint.md",
        "notes.txt",
        "config.private.yaml",
        ".internal/strategy.md",
    )

    for path in private_paths:
        assert any(fnmatch.fnmatch(path.lower(), pattern) for pattern in PROHIBITED_PATH_PATTERNS)


def test_common_secret_shapes_are_detected_without_storing_a_secret() -> None:
    synthetic_key = "AK" + "IA" + "A" * 16
    synthetic_path = "E:" + "\\" + "Claude AI" + "\\" + "private"

    assert SECRET_PATTERNS["AWS access key"].search(synthetic_key)
    assert SECRET_PATTERNS["Windows workstation path"].search(synthetic_path)

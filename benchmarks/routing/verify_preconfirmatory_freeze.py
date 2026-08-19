"""Verify the frozen Phase 2 pre-confirmatory artifact manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "routing_preconfirmatory_manifest_v1.json"


def git_blob_sha1(path: Path) -> str:
    """Return the Git blob SHA-1 for the exact file bytes."""
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def verify() -> dict[str, object]:
    """Fail if any frozen artifact differs from its registered Git blob hash."""
    payload: object = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("pre-confirmatory manifest must be a JSON object")
    manifest = cast(dict[str, object], payload)
    if manifest.get("status") != "frozen_before_confirmatory_test_open":
        raise ValueError("pre-confirmatory manifest status drifted")
    if manifest.get("test_set_opened") is not False:
        raise ValueError("pre-confirmatory manifest no longer declares the test sealed")

    artifacts_raw = manifest.get("artifacts")
    if not isinstance(artifacts_raw, dict):
        raise TypeError("pre-confirmatory manifest artifacts must be an object")
    artifacts = cast(dict[str, object], artifacts_raw)

    checked = 0
    for relative_path, expected_raw in sorted(artifacts.items()):
        if not isinstance(expected_raw, str):
            raise TypeError(f"manifest hash for {relative_path} must be a string")
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact missing: {relative_path}")
        observed = git_blob_sha1(path)
        if observed != expected_raw:
            raise ValueError(
                f"frozen artifact drifted: {relative_path}: {observed} != {expected_raw}"
            )
        checked += 1

    return {
        "status": "preconfirmatory_freeze_verified",
        "test_set_opened": False,
        "manifest_version": manifest.get("version"),
        "frozen_parent_commit_sha": manifest.get("frozen_parent_commit_sha"),
        "artifacts_checked": checked,
    }


def main() -> None:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Confirmatory test opened: false")


if __name__ == "__main__":
    main()

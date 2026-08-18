"""Run offline Phase 1 data and contract invariants without network access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.banking77 import Banking77Spec
from helix_support_intelligence.data.helixbank import manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT / "data" / "contracts"
MANIFEST_PATH = REPO_ROOT / "data" / "synthetic" / "helixbank-policy-v1" / "manifest.json"
BANKING_CONFIG = REPO_ROOT / "configs" / "data" / "banking77.json"
HELIXBANK_CONFIG = REPO_ROOT / "configs" / "data" / "helixbank_policy_v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def validate_contract_files() -> None:
    """Require every public schema to be valid JSON Schema-shaped metadata."""

    schemas = sorted(CONTRACT_ROOT.glob("*.schema.json"))
    if len(schemas) < 8:
        raise ValueError("Phase 1 requires the complete public contract set")
    for schema_path in schemas:
        schema = _load_object(schema_path)
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError(f"unexpected schema dialect: {schema_path}")
        if schema.get("type") != "object":
            raise ValueError(f"top-level contract must be an object: {schema_path}")
        if not schema.get("required"):
            raise ValueError(f"contract requires an explicit required set: {schema_path}")


def validate_banking_contract() -> None:
    """Parse the frozen BANKING77 contract and reject incomplete provenance."""

    spec = Banking77Spec.from_json(BANKING_CONFIG)
    if len(spec.source_revision) != 40:
        raise ValueError("BANKING77 source revision must be a full Git commit SHA")
    if len(spec.quarantine_train_indices) != spec.expected_counts["quarantine"]:
        raise ValueError("BANKING77 quarantine count drifted")
    if spec.train_examples + spec.test_examples != 13_083:
        raise ValueError("BANKING77 frozen source size drifted")
    if spec.intent_count != 77:
        raise ValueError("BANKING77 intent cardinality drifted")


def validate_helixbank_contract() -> None:
    """Verify generated, configured, and committed corpus manifests agree exactly."""

    configured = _load_object(HELIXBANK_CONFIG)
    generated = manifest()
    committed = _load_object(MANIFEST_PATH)
    for key in ("corpus_version", "generator_version", "counts", "sha256"):
        if configured.get(key) != generated.get(key):
            raise ValueError(f"HelixBank configured {key} does not match generator")
        if committed.get(key) != generated.get(key):
            raise ValueError(f"HelixBank committed {key} does not match generator")


def main() -> None:
    validate_contract_files()
    validate_banking_contract()
    validate_helixbank_contract()
    print("Phase 1 offline data contracts passed.")


if __name__ == "__main__":
    main()

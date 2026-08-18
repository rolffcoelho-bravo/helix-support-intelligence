"""Public Phase 1 contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.domain.decisions import TerminalDecision

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "data" / "contracts"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _enum(schema: dict[str, Any], property_name: str) -> set[str]:
    properties = cast(dict[str, Any], schema["properties"])
    target = cast(dict[str, Any], properties[property_name])
    return set(cast(list[str], target["enum"]))


def test_contract_suite_is_complete_and_json_schema_2020_12() -> None:
    names = {path.name for path in CONTRACT_ROOT.glob("*.schema.json")}
    assert names == {
        "event.schema.json",
        "generation.schema.json",
        "policy_document.schema.json",
        "relevance_judgment.schema.json",
        "retrieval.schema.json",
        "retrieval_query.schema.json",
        "routing.schema.json",
        "safety.schema.json",
    }
    for path in CONTRACT_ROOT.glob("*.schema.json"):
        schema = _load(path)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["required"]


def test_decision_enums_match_domain_vocabulary() -> None:
    expected = {decision.value for decision in TerminalDecision}
    for name, prop in (
        ("event.schema.json", "terminal_decision"),
        ("generation.schema.json", "decision"),
        ("routing.schema.json", "decision"),
        ("safety.schema.json", "expected_decision"),
        ("retrieval_query.schema.json", "expected_decision"),
    ):
        assert _enum(_load(CONTRACT_ROOT / name), prop) == expected


def test_public_experiment_registry_contains_no_unpublished_runs() -> None:
    registry = (ROOT / "experiments" / "registry.yaml").read_text(encoding="utf-8")
    assert "experiments: []" in registry
    assert "private acceptance thresholds" in registry

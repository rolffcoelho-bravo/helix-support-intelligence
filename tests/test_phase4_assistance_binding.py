"""A4.1 assistance binding invariants before any development scoring."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.helixbank import INTENTS, generate_bundle
from helix_support_intelligence.domain.decisions import TerminalDecision

ROOT = Path(__file__).resolve().parents[1]
BINDING_PATH = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"
SUBSETS_PATH = ROOT / "configs" / "models" / "assistance_a41_subsets_v1.json"
OUTPUT_SCHEMA_PATH = (
    ROOT / "data" / "contracts" / "phase4" / "assistance_candidate_output.schema.json"
)
RUNTIME_PATH = ROOT / "benchmarks" / "assistance" / "runtime_a41.py"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_intents() -> set[str]:
    bundle = generate_bundle()
    conflicts = {
        str(row["intent"])
        for row in bundle.queries
        if row["case_type"] == "conflicting_evidence"
    }
    non_conflicts = set(INTENTS) - conflicts

    def ordered(values: set[str]) -> list[str]:
        return sorted(
            values,
            key=lambda intent: hashlib.sha256(f"20260819:{intent}".encode()).hexdigest(),
        )

    return set(ordered(conflicts)[:5]) | set(ordered(non_conflicts)[:55])


def _selected(quotas: dict[str, int]) -> list[str]:
    development = _development_intents()
    by_case: dict[str, list[str]] = {}
    for row in generate_bundle().queries:
        if str(row["intent"]) not in development:
            continue
        by_case.setdefault(str(row["case_type"]), []).append(str(row["query_id"]))

    result: list[str] = []
    for case_type, quota in quotas.items():
        ordered = sorted(
            by_case[case_type],
            key=lambda query_id: hashlib.sha256(
                f"A4.1-diagnostics-v1:{query_id}".encode()
            ).hexdigest(),
        )
        result.extend(ordered[:quota])
    return result


def test_a41_generator_prompt_and_schema_are_exactly_bound() -> None:
    binding = _load(BINDING_PATH)
    generator = cast(dict[str, Any], binding["generator"])
    prompts = cast(dict[str, dict[str, Any]], binding["prompts"])

    assert binding["status"] == "implemented_preflight_no_results"
    assert generator["model"] == "gpt-5.4-mini-2026-03-17"
    assert generator["reasoning_effort"] == "none"
    assert generator["temperature"] == 0.0
    assert generator["max_output_tokens"] == 512
    assert generator["tools"] == []
    assert generator["store"] is False
    assert generator["automatic_quality_call_retries"] == 0

    for name in ("system", "request_template"):
        prompt = prompts[name]
        path = ROOT / str(prompt["path"])
        assert _sha256(path) == prompt["sha256"]
        assert len(path.read_bytes()) == prompt["bytes"]

    structured = cast(dict[str, Any], generator["structured_output"])
    assert structured["post_parse_schema_sha256"] == _sha256(OUTPUT_SCHEMA_PATH)


def test_a41_verifiers_are_pinned_and_independent() -> None:
    binding = _load(BINDING_PATH)
    runtime = cast(dict[str, Any], binding["runtime_verifier"])
    evaluator = cast(dict[str, Any], binding["evaluation_verifier"])

    assert runtime["model_id"] == "cross-encoder/nli-deberta-v3-small"
    assert runtime["revision"] == "fa2804872c3b4bd748f38c0185cc85775361e735"
    assert runtime["architecture_family"] == "deberta-v3"
    assert evaluator["model_id"] == "cross-encoder/nli-MiniLM2-L6-H768"
    assert evaluator["revision"] == "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
    assert evaluator["architecture_family"] == "minilm2-roberta"
    assert runtime["architecture_family"] != evaluator["architecture_family"]

    for verifier in (runtime, evaluator):
        assert verifier["entailment_label"] == 1
        assert verifier["entailment_threshold"] == 0.8
        assert verifier["max_length"] == 512
        assert verifier["batch_size"] == 8
        assert verifier["device"] == "CPU"


def test_a41_diagnostic_subsets_are_deterministic_and_development_only() -> None:
    subsets = _load(SUBSETS_PATH)
    selection = cast(dict[str, dict[str, Any]], subsets["selection"])
    repeatability = selection["repeatability"]
    latency = selection["latency"]

    assert repeatability["query_ids"] == _selected(cast(dict[str, int], repeatability["quotas"]))
    assert latency["query_ids"] == _selected(cast(dict[str, int], latency["quotas"]))
    assert len(repeatability["query_ids"]) == 30
    assert len(latency["query_ids"]) == 60
    assert set(repeatability["query_ids"]) <= set(latency["query_ids"])

    development = _development_intents()
    query_intent = {
        str(row["query_id"]): str(row["intent"]) for row in generate_bundle().queries
    }
    assert all(query_intent[qid] in development for qid in latency["query_ids"])


def test_a41_budgets_pricing_and_result_guard_are_frozen() -> None:
    binding = _load(BINDING_PATH)
    budgets = cast(dict[str, Any], binding["budgets"])
    pricing = cast(dict[str, Any], binding["pricing_snapshot"])
    guard = cast(dict[str, Any], binding["results_guard"])

    assert budgets["p95_latency_ms"] == {"G0": 100, "G1": 6000, "G2": 8000}
    assert budgets["maximum_estimated_cost_usd_per_request"] == {
        "G0": 0.0,
        "G1": 0.005,
        "G2": 0.005,
    }
    assert pricing["snapshot_date"] == "2026-08-19"
    assert pricing["openai_standard_usd_per_1m_tokens"] == {
        "input": 0.75,
        "cached_input": 0.075,
        "output": 4.5,
    }
    assert guard["development_scores_computed"] == 0
    assert guard["confirmatory_scores_computed"] == 0
    assert guard["generator_calls_made_by_a41_preflight"] == 0
    assert guard["nli_calls_made_by_a41_preflight"] == 0
    assert guard["assistance_performance_results_generated"] is False


def test_a41_runtime_and_dependency_lock_match_binding_hashes() -> None:
    binding = _load(BINDING_PATH)
    runtime = cast(dict[str, Any], binding["benchmark_runtime"])

    assert _sha256(ROOT / str(runtime["script"])) == runtime["script_sha256"]
    assert _sha256(ROOT / str(runtime["lock"])) == runtime["lock_sha256"]
    assert runtime["heavy_dependencies_scoped_to_benchmark"] is True
    assert runtime["normal_package_dependencies_modified"] is False


def test_a41_candidate_schema_uses_existing_terminal_decisions() -> None:
    schema = _load(OUTPUT_SCHEMA_PATH)
    properties = cast(dict[str, Any], schema["properties"])
    decision = cast(dict[str, Any], properties["decision"])

    assert schema["additionalProperties"] is False
    assert set(decision["enum"]) == {item.value for item in TerminalDecision}


def test_a41_runtime_preflight_is_zero_call_and_zero_score() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNTIME_PATH), "--preflight"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "passed"
    assert payload["generator_calls_made"] == 0
    assert payload["nli_calls_made"] == 0
    assert payload["performance_scores_computed"] == 0

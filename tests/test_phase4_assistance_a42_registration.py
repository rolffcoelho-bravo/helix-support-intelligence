"""A4.2 registration guards before the development result surface is opened."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = ROOT / "configs" / "models" / "assistance_execution_a42_v1.json"
BINDING_PATH = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "phase4-assistance-a42.yml"
RUNNER_PATH = ROOT / "benchmarks" / "assistance" / "run_a42.py"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_a42_execution_contract_is_development_only_and_unopened() -> None:
    execution = _load(EXECUTION_PATH)

    assert execution["execution_id"] == "phase4-assistance-a4.2-development-v1"
    assert execution["status"] == "registered_pre_execution"
    assert execution["partition"] == "development"
    assert execution["development_intents"] == 60
    assert execution["development_queries"] == 240
    assert execution["confirmatory_intents_opened"] == 0
    assert execution["confirmatory_queries_opened"] == 0
    assert execution["candidate_order"] == ["G0", "G1", "G2"]
    assert execution["provider_failure_policy"]["automatic_retries"] == 0
    assert execution["results_guard"]["development_results_opened"] is False
    assert execution["results_guard"]["confirmatory_results_opened"] is False


def test_a42_operational_interpretations_are_frozen_before_scoring() -> None:
    execution = _load(EXECUTION_PATH)
    interpretation = execution["operational_interpretations_frozen_before_scoring"]

    assert "ANSWER_WITH_EVIDENCE" in interpretation["citation_completeness_applicability"]
    assert "up to eight" in interpretation["nli_batching"]
    assert "maximum estimated provider cost" in interpretation["cost_budget_check"]
    assert "development intents" in interpretation["adversarial_development_only"]
    assert execution["adversarial_development_counts"] == {
        "direct_injection": 60,
        "citation_spoof": 60,
        "indirect_injection": 16,
        "archived_distractor": 7,
        "total": 143,
    }


def test_a42_preserves_a41_generator_and_budget_binding() -> None:
    execution = _load(EXECUTION_PATH)
    binding = _load(BINDING_PATH)

    assert execution["binding_id"] == binding["binding_id"]
    assert binding["generator"]["model"] == "gpt-5.4-mini-2026-03-17"
    assert binding["generator"]["temperature"] == 0.0
    assert binding["generator"]["max_output_tokens"] == 512
    assert binding["runtime_verifier"]["batch_size"] == 8
    assert binding["evaluation_verifier"]["batch_size"] == 8
    assert binding["budgets"]["p95_latency_ms"] == {
        "G0": 100,
        "G1": 6000,
        "G2": 8000,
    }
    assert binding["budgets"]["maximum_estimated_cost_usd_per_request"] == {
        "G0": 0.0,
        "G1": 0.005,
        "G2": 0.005,
    }


def test_a42_runner_applies_batched_support_and_inline_g0_citations() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "evaluate_a42.support = batched_support" in source
    assert "g0_with_inline_sentence_citations" in source
    assert 'output["answer"] = f"{base}{inline}."' in source


def test_a42_workflow_is_one_shot_and_uses_frozen_a41_lock() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "branches: [main]" in workflow
    assert '".github/workflows/phase4-assistance-a42.yml"' in workflow
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in workflow
    assert "runtime_a41.py.lock benchmarks/assistance/run_a42.py.lock" in workflow
    assert "runtime_a41.py.lock benchmarks/assistance/run_verify_a42.py.lock" in workflow
    assert "scripts/preflight_phase4_a42.py" in workflow
    assert "benchmarks/assistance/run_verify_a42.py" in workflow
    assert "confirmatory" not in workflow.lower()


def test_a42_preflight_is_zero_call_and_zero_score() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preflight_phase4_a42.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "passed"
    assert payload["development_queries"] == 240
    assert payload["confirmatory_queries_opened"] == 0
    assert payload["generator_calls_made"] == 0
    assert payload["nli_calls_made"] == 0
    assert payload["performance_scores_computed"] == 0

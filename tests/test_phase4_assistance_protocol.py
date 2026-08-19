"""Phase 4 A4.0 assistance-protocol invariants before generation implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.helixbank import INTENTS, generate_bundle, manifest
from helix_support_intelligence.domain.decisions import TerminalDecision

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "models" / "assistance_protocol_v1.json"
EVAL_SCHEMA_PATH = (
    ROOT / "data" / "contracts" / "phase4" / "assistance_evaluation_record.schema.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _intent_partition() -> tuple[set[str], set[str]]:
    bundle = generate_bundle()
    conflict_intents = {
        str(row["intent"]) for row in bundle.queries if row["case_type"] == "conflicting_evidence"
    }
    non_conflict_intents = set(INTENTS) - conflict_intents

    def ordered(values: set[str]) -> list[str]:
        return sorted(
            values,
            key=lambda intent: hashlib.sha256(f"20260819:{intent}".encode()).hexdigest(),
        )

    development = set(ordered(conflict_intents)[:5]) | set(ordered(non_conflict_intents)[:55])
    confirmatory = set(INTENTS) - development
    return development, confirmatory


def test_a40_reuses_exact_frozen_helixbank_corpus() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    source = cast(dict[str, Any], protocol["source_state"])
    corpus = cast(dict[str, Any], source["corpus"])
    current = manifest()

    assert protocol["protocol_id"] == "phase4-assistance-a4.0-v1"
    assert protocol["status"] == "frozen_pre_implementation"
    assert protocol["phase"] == 4
    assert protocol["checkpoint"] == "A4.0"
    assert source["retrieval_version"] == "retrieval-selected-v1"
    assert source["retrieval_candidate"] == "B0"
    assert corpus["version"] == current["corpus_version"]
    assert corpus["generator_version"] == current["generator_version"]
    assert corpus["counts"] == current["counts"]
    assert corpus["sha256"] == current["sha256"]


def test_a40_case_counts_match_frozen_query_semantics() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    source = cast(dict[str, Any], protocol["source_state"])
    expected = cast(dict[str, int], source["case_counts"])
    actual: dict[str, int] = {}
    bundle = generate_bundle()
    for row in bundle.queries:
        key = str(row["case_type"])
        actual[key] = actual.get(key, 0) + 1

    assert (
        actual
        == expected
        == {
            "answerable": 77,
            "ambiguous": 77,
            "outdated_evidence": 77,
            "missing_evidence": 70,
            "conflicting_evidence": 7,
        }
    )
    current_untrusted = sum(
        bool(row["untrusted_content_fixture"]) and row["status"] == "current"
        for row in bundle.documents
    )
    assert current_untrusted == 5
    assert sum(row["status"] == "archived" for row in bundle.documents) == 7


def test_a40_partition_is_intent_clustered_and_exactly_frozen() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    partition = cast(dict[str, Any], protocol["partition"])
    counts = cast(dict[str, int], partition["counts"])
    development, confirmatory = _intent_partition()

    assert development.isdisjoint(confirmatory)
    assert development | confirmatory == set(INTENTS)
    assert len(development) == counts["development_intents"] == 60
    assert len(confirmatory) == counts["confirmatory_intents"] == 17
    assert counts["development_queries"] == 240
    assert counts["confirmatory_queries"] == 68

    bundle = generate_bundle()
    dev_queries = [row for row in bundle.queries if str(row["intent"]) in development]
    confirm_queries = [row for row in bundle.queries if str(row["intent"]) in confirmatory]
    assert len(dev_queries) == 240
    assert len(confirm_queries) == 68
    assert sum(row["case_type"] == "conflicting_evidence" for row in dev_queries) == 5
    assert sum(row["case_type"] == "conflicting_evidence" for row in confirm_queries) == 2


def test_generation_isolation_cannot_expose_gold_or_benchmark_labels() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    modes = cast(dict[str, Any], protocol["evaluation_modes"])
    isolation = cast(dict[str, Any], modes["generation_isolation"])
    forbidden = set(cast(list[str], isolation["candidate_forbidden_fields"]))

    assert isolation["role"] == "primary candidate-comparison surface"
    assert isolation["retrieval_dependency"] == "none during scoring"
    assert isolation["candidate_visible_query_fields"] == ["text"]
    assert {
        "query_id",
        "intent",
        "queue",
        "case_type",
        "expected_decision",
        "gold_citations",
        "allowed_resolution_types",
        "relevance",
        "relevance_judgments",
        "conflict_fixture",
        "untrusted_content_fixture",
    } == forbidden


def test_evidence_sufficiency_maps_existing_cases_to_terminal_decisions() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    rules = cast(dict[str, Any], protocol["evidence_sufficiency"])

    assert rules["answerable"]["expected_decision"] == "ANSWER_WITH_EVIDENCE"
    assert rules["outdated_evidence"]["expected_decision"] == "ANSWER_WITH_EVIDENCE"
    assert rules["ambiguous"]["expected_decision"] == "ASK_FOR_CLARIFICATION"
    assert rules["missing_evidence"]["expected_decision"] == "ESCALATE_LOW_CONFIDENCE"
    assert rules["conflicting_evidence"]["expected_decision"] == "ESCALATE_CONFLICTING_EVIDENCE"
    assert rules["safety_override"]["expected_decision"] == "ESCALATE_SAFETY_RISK"
    assert rules["system_failure"]["expected_decision"] == "ESCALATE_SYSTEM_FAILURE"
    assert rules["minimum_direct_relevance_grade"] == 2


def test_a40_candidate_ladder_is_small_and_verifier_is_independent() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    ladder = cast(list[dict[str, Any]], protocol["candidate_ladder"])
    by_id = {str(candidate["id"]): candidate for candidate in ladder}

    assert list(by_id) == ["G0", "G1", "G2"]
    assert by_id["G0"]["model"] is None
    assert by_id["G1"]["decoding"] == {"temperature": 0.0, "max_output_tokens": 512}
    assert by_id["G1"]["tools_allowed"] is False
    assert "same frozen generator" in str(by_id["G2"]["generator"])
    assert by_id["G2"]["runtime_verifier_must_differ_from_evaluation_verifier_family"] is True
    assert by_id["G2"]["repair_or_regeneration_after_verifier_failure"] is False


def test_a40_grounding_metrics_hypotheses_and_cluster_inference_are_registered() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    metrics = cast(dict[str, Any], protocol["metrics"])
    hypotheses = cast(dict[str, dict[str, Any]], protocol["registered_hypotheses"])
    inference = cast(dict[str, Any], protocol["inference"])
    grounding = cast(dict[str, Any], protocol["grounding_evaluation"])

    assert metrics["primary"]["name"] == "strict_grounded_success_rate"
    assert "unsupported_sentence_rate" in metrics["registered_secondary"]
    assert "citation_completeness" in metrics["registered_secondary"]
    assert hypotheses["H1"]["comparison"] == "G1 - G0"
    assert hypotheses["H1"]["endpoint"] == "strict_grounded_success_rate"
    assert hypotheses["H2"]["comparison"] == "G2 - G1"
    assert hypotheses["H2"]["endpoint"] == "unsupported_sentence_rate"
    assert inference["primary_resampling_unit"] == "intent cluster"
    assert inference["method"] == "paired nonparametric cluster bootstrap"
    assert inference["replicates"] == 5000
    assert inference["seed"] == 20260819
    independence = str(grounding["evaluation_verifier_independence"])
    assert independence.startswith("The evaluation verifier cannot be")


def test_a40_adversarial_suite_is_predeclared_and_bounded() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    suite = cast(dict[str, Any], protocol["adversarial_suite"])

    assert suite["direct_injection"]["count"] == 77
    assert suite["citation_spoof"]["count"] == 77
    assert suite["indirect_injection"]["count"] == 20
    assert suite["archived_distractor"]["count"] == 7
    assert "POLICY-999" in suite["direct_injection"]["query_suffix"]
    assert "POLICY-999" in suite["citation_spoof"]["query_suffix"]
    assert "POLICY-999" in suite["indirect_injection"]["document_body_suffix"]
    assert suite["archived_distractor"]["archived_citation_permitted"] is False
    assert suite["injection_success_rate_target"] == 0.0
    assert suite["report_exact_failure_counts"] is True


def test_a40_latency_cost_and_complexity_rules_require_pre_result_a41_binding() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    latency = cast(dict[str, Any], protocol["latency_and_cost"])
    adoption = cast(dict[str, Any], protocol["complexity_adoption"])
    rule = cast(dict[str, Any], adoption["candidate_rule"])
    guard = cast(dict[str, Any], protocol["execution_guard"])

    assert latency["repeatability_diagnostic"] == {
        "cases": 30,
        "selection": "deterministic stratified development subset chosen before A4.1 scoring",
        "repetitions": 3,
        "purpose": "measure output stability without replacing the registered quality pass",
    }
    assert latency["latency_sample"]["cases"] == 60
    assert latency["latency_sample"]["warmup_requests"] == 10
    assert latency["latency_sample"]["timed_passes"] == 3
    assert adoption["starting_winner"] == "G0"
    assert adoption["evaluation_order"] == ["G1", "G2"]
    assert rule["minimum_delta_strict_grounded_success_rate"] == 0.01
    assert rule["require_cluster_bootstrap_95_ci_lower_bound_above_zero"] is True
    assert rule["require_candidate_p95_within_A4_1_frozen_budget"] is True
    assert rule["require_candidate_cost_within_A4_1_frozen_budget"] is True
    assert guard["results_opened"] is False
    assert guard["candidate_implementation_started"] is False
    assert guard["scoring_allowed_only_after_A4_1_binding_merge"] is True
    assert "exact prompt version and prompt bytes" in guard["A4_1_may_bind_without_reopening_A4_0"]


def test_phase4_evaluation_schema_uses_existing_terminal_vocabulary() -> None:
    schema = _load_json(EVAL_SCHEMA_PATH)
    properties = cast(dict[str, Any], schema["properties"])
    decision = cast(dict[str, Any], properties["decision"])
    candidate = cast(dict[str, Any], properties["candidate_id"])

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(decision["enum"]) == {item.value for item in TerminalDecision}
    assert candidate["enum"] == ["G0", "G1", "G2"]
    assert properties["corpus_version"] == {"const": "helixbank-policy-v1.0.0"}

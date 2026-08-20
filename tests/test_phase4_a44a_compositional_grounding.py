from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from compositional_cases_a44a import (  # type: ignore[import-not-found]  # noqa: E402
    compositional_partition,
    generate_cases,
    suite_summary,
)
from grounding_anchors_a43a import (  # type: ignore[import-not-found]  # noqa: E402
    development_intents,
)
from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"


def test_a44a_suite_is_development_only_and_frozen_size() -> None:
    bundle = generate_bundle()
    development = development_intents(bundle)
    partition = compositional_partition(bundle)
    rows = generate_cases(bundle)
    summary = suite_summary()

    assert len(development) == 60
    assert len(partition["calibration"]) == 40
    assert len(partition["validation"]) == 20
    assert not partition["calibration"] & partition["validation"]
    assert partition["calibration"] | partition["validation"] == development
    assert len(rows) == 432
    assert summary["split_counts"] == {"calibration": 288, "validation": 144}
    assert summary["category_counts"] == {
        "citation_invalid": 60,
        "contradiction_unsupported": 60,
        "literal_supported": 60,
        "multi_document_supported": 60,
        "paraphrase_supported": 60,
        "partial_multi_document_unsupported": 60,
        "stale_current_evidence": 7,
        "unresolved_conflict": 5,
        "unsupported_approval": 60,
    }


def test_a44a_cases_are_candidate_and_query_independent() -> None:
    required = {
        "case_id",
        "split",
        "intent",
        "category",
        "presented_document_ids",
        "cited_document_ids",
        "requires_current_evidence",
        "atoms",
        "expected_verdict",
    }
    for row in generate_cases():
        assert set(row) == required
        assert "query_id" not in row
        assert "query_text" not in row
        assert "candidate_id" not in row
        assert "candidate_output" not in row


def test_a44a_semantic_verifier_is_unbound_and_no_execution_is_authorized() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    semantic = config["component_boundaries"]["atomic_semantic_relation"]
    guards = config["execution_guards"]

    assert semantic["binding_status"] == "UNBOUND"
    assert semantic["model_family"] is None
    assert semantic["model_revision"] is None
    assert semantic["thresholds"] is None
    assert semantic["separate_future_approval_required"] is True

    assert guards["generator_calls"] == 0
    assert guards["openai_calls"] == 0
    assert guards["candidate_calls"] == 0
    assert guards["candidate_scoring"] == 0
    assert guards["semantic_verifier_calls"] == 0
    assert guards["model_family_searches"] == 0
    assert guards["confirmatory_query_scoring"] == 0
    assert guards["confirmatory_query_inspection"] == 0
    assert guards["a44b_not_authorized_by_this_gate"] is True


def test_a44a_deterministic_veto_categories_are_structurally_separable() -> None:
    bundle = generate_bundle()
    documents = {str(row["document_id"]): dict(row) for row in bundle.documents}

    for row in generate_cases(bundle):
        presented = {str(value) for value in row["presented_document_ids"]}
        cited = {str(value) for value in row["cited_document_ids"]}
        category = str(row["category"])

        if category == "citation_invalid":
            assert not cited <= presented
            assert row["expected_verdict"] == "CITATION_INVALID"
        else:
            assert cited <= presented

        if category == "stale_current_evidence":
            assert row["expected_verdict"] == "STALE_EVIDENCE"
            assert all(documents[doc_id]["status"] == "archived" for doc_id in presented)

        if category == "unresolved_conflict":
            assert row["expected_verdict"] == "CONFLICTING_EVIDENCE"
            assert any(bool(documents[doc_id]["conflict_fixture"]) for doc_id in presented)


def test_a44a_future_binding_cannot_use_validation_or_prior_candidate_results() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rules = config["future_binding_rules"]

    assert rules["semantic_verifier_model_search_allowed_in_a44a"] is False
    assert rules["semantic_verifier_inference_allowed_in_a44a"] is False
    assert rules["a42_candidate_results_may_influence_binding"] is False
    assert rules["a43a_validation_results_may_select_replacement_family"] is False
    assert rules["a44a_validation_split_may_influence_binding_or_thresholds"] is False
    assert rules["calibration_split_only_for_future_parameter_binding"] is True
    assert rules["validation_split_remains_untouched_until_future_binding_is_frozen"] is True
    assert rules["replacement_family_requires_separately_versioned_approved_gate"] is True

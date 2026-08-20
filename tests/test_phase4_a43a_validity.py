from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from grounding_anchors_a43a import (  # noqa: E402
    anchor_partition,
    development_intents,
    generate_anchors,
    suite_summary,
)
from helix_support_intelligence.data.helixbank import generate_bundle

CONFIG_PATH = ROOT / "configs" / "models" / "assistance_validity_a43a_v1.json"


def test_a43a_anchor_suite_is_development_only_and_frozen_size() -> None:
    bundle = generate_bundle()
    development = development_intents(bundle)
    partition = anchor_partition(bundle)
    rows = generate_anchors(bundle)
    summary = suite_summary()

    assert len(development) == 60
    assert len(partition["calibration"]) == 40
    assert len(partition["validation"]) == 20
    assert not partition["calibration"] & partition["validation"]
    assert partition["calibration"] | partition["validation"] == development
    assert len(rows) == 372
    assert summary["split_counts"] == {"calibration": 248, "validation": 124}
    assert summary["category_counts"] == {
        "citation_mismatch": 60,
        "conflict_union_claim": 5,
        "contradiction_queue": 60,
        "literal_policy": 60,
        "multi_document_conjunction": 60,
        "paraphrase_queue": 60,
        "stale_current_claim": 7,
        "unsupported_approval": 60,
    }


def test_a43a_anchors_are_candidate_and_query_independent() -> None:
    for row in generate_anchors():
        assert set(row) == {
            "anchor_id",
            "split",
            "intent",
            "category",
            "expected_entailment",
            "document_ids",
            "hypothesis",
        }
        assert "query_id" not in row
        assert "query_text" not in row
        assert "candidate_id" not in row
        assert "candidate_output" not in row


def test_a43a_protocol_forbids_candidate_and_confirmatory_scoring() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    guards = config["execution_guards"]
    assert guards["generator_calls"] == 0
    assert guards["openai_calls"] == 0
    assert guards["candidate_calls"] == 0
    assert guards["candidate_scoring"] == 0
    assert guards["confirmatory_query_scoring"] == 0
    assert guards["a43b_not_authorized_by_this_gate"] is True


def test_a43a_threshold_calibration_cannot_use_validation_split() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = config["evaluator_validation"]["threshold_calibration"]
    assert calibration["always_run"] is True
    assert calibration["calibration_split_only"] is True
    assert calibration["validation_split_may_not_influence_threshold"] is True
    assert calibration["grid_min"] == 0.05
    assert calibration["grid_max"] == 0.95
    assert calibration["grid_step"] == 0.01

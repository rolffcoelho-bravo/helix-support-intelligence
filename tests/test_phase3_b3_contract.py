from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "retrieval" / "b3_cross_encoder_v1.json"
SCRIPT = ROOT / "benchmarks" / "retrieval" / "evaluate_b3.py"
B1_RESULT = ROOT / "benchmarks" / "retrieval" / "results" / "b1_development_v1.json"


def _load(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def test_b3_configuration_is_frozen_before_first_score() -> None:
    config = _load(CONFIG)
    parent = cast(dict[str, object], config["parent"])
    model = cast(dict[str, object], config["model"])
    evaluation = cast(dict[str, object], config["evaluation"])
    guardrails = cast(dict[str, object], evaluation["guardrails"])
    latency = cast(dict[str, object], evaluation["latency"])

    assert config["status"] == "frozen_before_first_score"
    assert config["benchmark_version"] == "retrieval-benchmark-v1"
    assert parent["system"] == "B1"
    assert parent["candidate_depth"] == 50
    assert model["model_id"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert model["revision"] == "fbf9045f293a58fa68636213c5e0cb8a2de5d45e"
    assert model["model_safetensors_sha256"] == (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    )
    assert model["max_length"] == 512
    assert model["activation"] == "identity"
    assert model["batch_size"] == 32
    assert model["fine_tuning"] is False
    assert evaluation["minimum_material_ndcg_gain"] == 0.02
    assert guardrails["mrr_at_10_delta_min"] == 0.0
    assert guardrails["success_at_1_delta_min"] == 0.0
    assert guardrails["recall_at_20_delta_min"] == 0.0
    assert guardrails["citation_eligible_recall_at_20_delta_min"] == 0.0
    assert guardrails["recall_at_50_must_equal_parent"] is True
    assert latency["pair_count"] == 69300


def test_b3_parent_hash_matches_audited_b1_result() -> None:
    config = _load(CONFIG)
    b1 = _load(B1_RESULT)
    parent = cast(dict[str, object], config["parent"])
    evidence = cast(dict[str, object], b1["deterministic_evidence"])

    assert parent["accepted_full_ranking_sha256"] == evidence["accepted_ranking_sha256"]


def test_b3_development_script_fails_closed_and_keeps_test_sealed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "B1 parent ranking drifted before B3" in source
    assert '"confirmatory_queries.jsonl"' in source
    assert '"confirmatory_qrels.jsonl"' in source
    assert '"confirmatory_partition_opened": False' in source
    assert '"official_banking77_test_accessed": False' in source
    assert "banking.test_url" not in source
    assert "test.csv" not in source
    assert "minimum_material_ndcg_gain" in source
    assert "activation_fn=torch.nn.Identity()" in source

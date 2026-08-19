from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "retrieval" / "b2_rrf_v1.json"
SCRIPT = ROOT / "benchmarks" / "retrieval" / "evaluate_b2.py"
B0_RESULT = ROOT / "benchmarks" / "retrieval" / "results" / "b0_development_v1.json"
B1_RESULT = ROOT / "benchmarks" / "retrieval" / "results" / "b1_development_v1.json"


def _load(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def test_b2_configuration_is_frozen_before_first_score() -> None:
    config = _load(CONFIG)
    parents = config["parents"]
    fusion = config["fusion"]
    assert isinstance(parents, dict)
    assert isinstance(fusion, dict)

    assert config["status"] == "frozen_before_first_score"
    assert config["benchmark_version"] == "retrieval-benchmark-v1"
    assert fusion["family"] == "reciprocal_rank_fusion"
    assert fusion["k"] == 60
    assert fusion["rank_origin"] == 1
    assert fusion["rank_depth"] == 147
    assert fusion["score_normalization"] == "none_rank_only"
    assert fusion["learned_weights"] is False
    assert fusion["tie_break"] == "document_id_ascending"
    assert parents["B0"]["weight"] == 1.0
    assert parents["B1"]["weight"] == 1.0


def test_b2_parent_hashes_match_audited_results() -> None:
    config = _load(CONFIG)
    b0 = _load(B0_RESULT)
    b1 = _load(B1_RESULT)
    parents = cast(dict[str, dict[str, object]], config["parents"])
    b0_evidence = cast(dict[str, object], b0["deterministic_evidence"])
    b1_evidence = cast(dict[str, object], b1["deterministic_evidence"])

    assert parents["B0"]["accepted_ranking_sha256"] == b0_evidence["ranking_sha256"]
    assert parents["B1"]["accepted_ranking_sha256"] == b1_evidence["ranking_sha256"]


def test_b2_development_script_fails_closed_on_parent_drift_and_sealed_inputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "B0 parent ranking drifted" in source
    assert "B1 parent ranking drifted" in source
    assert '"confirmatory_queries.jsonl"' in source
    assert '"confirmatory_qrels.jsonl"' in source
    assert '"confirmatory_partition_opened": False' in source
    assert '"official_banking77_test_accessed": False' in source
    assert "banking.test_url" not in source
    assert "test.csv" not in source

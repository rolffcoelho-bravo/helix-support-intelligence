"""Fail-closed preflight for A4.4c calibration-only execution."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from calibration_cases_a44c import (  # type: ignore[import-not-found]  # noqa: E402
    canonical_calibration_jsonl_bytes,
    generate_calibration_cases,
)

A44A = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"
A44B = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"
A44C = ROOT / "configs" / "models" / "assistance_grounding_a44c_v1.json"
FROZEN_A44A_SUITE_SHA256 = "0ad07e9d08678dbc5fa8b625870d2c3140eef83b0dddb013a4ae479c56bdd90c"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _relation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"ENTAILED": 0, "CONTRADICTED": 0, "UNKNOWN": 0}
    for row in rows:
        presented = {str(value) for value in row["presented_document_ids"]}
        cited = {str(value) for value in row["cited_document_ids"]}
        if not cited or not cited.issubset(presented):
            continue
        for atom in row["atoms"]:
            entailed = {str(value) for value in atom["entailed_by"]}
            contradicted = {str(value) for value in atom["contradicted_by"]}
            for document_id in cited:
                if document_id in entailed:
                    counts["ENTAILED"] += 1
                elif document_id in contradicted:
                    counts["CONTRADICTED"] += 1
                else:
                    counts["UNKNOWN"] += 1
    return counts


def main() -> None:
    a44a = _load(A44A)
    a44b = _load(A44B)
    a44c = _load(A44C)
    assert a44c["execution_id"] == "phase4-assistance-a4.4c-calibration-only-v1"
    assert a44c["source_main_sha"] == "d94b834f61c4208028907c602b54792c643ac074"
    assert a44c["protocol_id"] == a44a["protocol_id"]
    assert a44c["binding_id"] == a44b["binding_id"]
    assert a44a["validation_suite"]["sha256"] == FROZEN_A44A_SUITE_SHA256
    assert a44c["scope"]["calibration_case_rows"] == 288
    assert a44c["scope"]["calibration_semantic_pair_rows"] == 491
    assert a44c["scope"]["validation_case_rows_materialized_authorized"] == 0
    assert a44c["scope"]["validation_case_rows_authorized"] == 0
    assert a44c["scope"]["validation_semantic_pair_rows_authorized"] == 0
    assert a44c["scope"]["validation_metrics_authorized"] == 0
    assert a44c["scope"]["confirmatory_query_rows_authorized"] == 0
    assert a44c["artifact_contract"]["validation_cases_must_not_be_materialized"] is True
    assert a44c["calibration"]["grid_points"] == 376
    assert a44c["calibration"]["grid_start"] == 0.25
    assert a44c["calibration"]["grid_stop"] == 4.0
    assert a44c["calibration"]["grid_step"] == 0.01
    assert a44c["calibration"]["tie_break"] == "smallest_temperature"
    assert a44c["calibration"]["may_change_raw_argmax_class"] is False
    assert a44c["calibration"]["may_change_final_grounding_verdict"] is False
    assert a44c["forbidden"]["validation_inference"] is True
    assert a44c["forbidden"]["candidate_inference"] is True
    assert a44c["forbidden"]["confirmatory_inference"] is True
    assert a44c["forbidden"]["model_substitution"] is True

    verifier = a44b["semantic_verifier"]
    assert verifier["model_id"] == "FacebookAI/roberta-large-mnli"
    assert verifier["revision"] == "2a8f12d27941090092df78e4ba6f0928eb5eac98"
    assert verifier["weights_sha256"] == (
        "f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5"
    )
    assert verifier["class_decision"]["rule"] == "argmax_raw_logits"

    rows = generate_calibration_cases()
    assert len(rows) == 288
    assert len({str(row["intent"]) for row in rows}) == 40
    assert all(row["split"] == "calibration" for row in rows)
    relation_counts = _relation_counts(rows)
    expected_relation_counts = {
        str(key): int(value)
        for key, value in a44c["scope"]["calibration_gold_relation_counts"].items()
    }
    assert relation_counts == expected_relation_counts
    assert sum(relation_counts.values()) == 491
    calibration_hash = hashlib.sha256(canonical_calibration_jsonl_bytes(rows)).hexdigest()
    assert calibration_hash == a44c["scope"]["calibration_case_sha256"]

    print(
        json.dumps(
            {
                "status": "PASSED_PRE_EXECUTION_CALIBRATION_ONLY_NO_RESULTS",
                "calibration_cases_authorized": 288,
                "calibration_semantic_pairs": 491,
                "calibration_case_sha256": calibration_hash,
                "validation_cases_materialized": 0,
                "validation_cases_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "semantic_inference_performed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Fail-closed preflight for A4.4c calibration-only execution."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from compositional_cases_a44a import canonical_jsonl_bytes, generate_cases  # noqa: E402

A44A = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"
A44B = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"
A44C = ROOT / "configs" / "models" / "assistance_grounding_a44c_v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def main() -> None:
    a44a = _load(A44A)
    a44b = _load(A44B)
    a44c = _load(A44C)
    assert a44c["execution_id"] == "phase4-assistance-a4.4c-calibration-only-v1"
    assert a44c["source_main_sha"] == "d94b834f61c4208028907c602b54792c643ac074"
    assert a44c["protocol_id"] == a44a["protocol_id"]
    assert a44c["binding_id"] == a44b["binding_id"]
    assert a44c["scope"]["calibration_case_rows"] == 288
    assert a44c["scope"]["calibration_semantic_pair_rows"] == 491
    assert a44c["scope"]["validation_case_rows_authorized"] == 0
    assert a44c["scope"]["confirmatory_query_rows_authorized"] == 0
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

    rows = generate_cases()
    assert len(rows) == 432
    assert sum(row["split"] == "calibration" for row in rows) == 288
    assert sum(row["split"] == "validation" for row in rows) == 144
    suite_hash = hashlib.sha256(canonical_jsonl_bytes(rows)).hexdigest()
    assert suite_hash == "0ad07e9d08678dbc5fa8b625870d2c3140eef83b0dddb013a4ae479c56bdd90c"
    assert suite_hash == a44a["validation_suite"]["sha256"]

    print(
        json.dumps(
            {
                "status": "PASSED_PRE_EXECUTION_CALIBRATION_ONLY_NO_RESULTS",
                "calibration_cases_authorized": 288,
                "validation_cases_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "semantic_inference_performed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

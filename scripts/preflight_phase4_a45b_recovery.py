"""Fail-closed preregistration audit for deterministic A4.5b recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "benchmarks" / "assistance" / "results" / "a45b_partial_attempt1_v1.json"
FINALIZER = ROOT / "benchmarks" / "assistance" / "finalize_calibration_a45b_recovery.py"
VERIFIER = ROOT / "benchmarks" / "assistance" / "verify_calibration_a45b_recovery.py"
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"
SOURCE_MAIN_SHA = "92c5b073fd91454a6b3f1b11ab13f453e1644a6d"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _load_finalizer() -> Any:
    spec = importlib.util.spec_from_file_location("finalize_calibration_a45b_recovery", FINALIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load A4.5b recovery finalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    provenance = _json(PROVENANCE)
    config = _json(CONFIG)
    finalizer = _load_finalizer()

    if provenance["scientific_execution_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b recovery must use the exact partial-inference SHA")
    if provenance["workflow_run_id"] != 32581433921:
        raise RuntimeError("A4.5b recovery source run drifted")
    if provenance["artifact"]["id"] != 9477913279:
        raise RuntimeError("A4.5b recovery artifact ID drifted")
    if provenance["artifact"]["zip_sha256"] != (
        "0bb5bd16c5a582f69e6421d0017bfa3e33e60cb5d4b31a005f6929d6cfc2c633"
    ):
        raise RuntimeError("A4.5b recovery artifact ZIP digest drifted")
    if provenance["artifact"]["files"]["calibration_pair_scores.jsonl"] != (
        "fe6b22278945e4df094fe8d0314706281ae6fd7915a2f613606b29f63c5d32b6"
    ):
        raise RuntimeError("A4.5b recovery raw score digest drifted")
    if provenance["score_rows"] != 360 or provenance["unique_pair_ids"] != 360:
        raise RuntimeError("A4.5b recovery raw score cardinality drifted")
    if provenance["score_splits"] != ["calibration"]:
        raise RuntimeError("A4.5b recovery must consume calibration scores only")
    if provenance["threshold_selected"] is not False:
        raise RuntimeError("A4.5b recovery cannot follow a prior threshold selection")
    if provenance["scientific_pass_fail_computed"] is not False:
        raise RuntimeError("A4.5b recovery cannot follow a prior scientific verdict")

    if finalizer.registered_metric_name("sufficiency_macro_f1_min_on_relevant_pairs") != (
        "sufficiency_macro_f1_on_relevant_pairs",
        "min",
    ):
        raise RuntimeError("A4.5b sufficiency requirement dispatch is incorrect")
    if finalizer.registered_metric_name("polarity_macro_f1_min_on_relevant_sufficient_pairs") != (
        "polarity_macro_f1_on_relevant_sufficient_pairs",
        "min",
    ):
        raise RuntimeError("A4.5b polarity requirement dispatch is incorrect")

    for path in (FINALIZER, VERIFIER):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("from transformers", "import transformers", "hf_hub_download"):
            if forbidden in source:
                raise RuntimeError(f"A4.5b recovery may not load models: {path.name}")

    sealed = config["sealed_partitions"]
    if sealed["validation_scoring_authorized"] != 0:
        raise RuntimeError("Fresh validation remains unauthorized")
    if sealed["confirmatory_query_records_inspected"] != 0:
        raise RuntimeError("Confirmatory queries must remain unopened")
    if sealed["confirmatory_scoring_authorized"] != 0:
        raise RuntimeError("Confirmatory scoring remains unauthorized")
    if config["next_checkpoint"]["authorized_by_a45b"] is not False:
        raise RuntimeError("A4.5c requires separate approval")

    print(
        json.dumps(
            {
                "status": "PASSED_A45B_DETERMINISTIC_RECOVERY_PRE_EXECUTION",
                "source_artifact_id": provenance["artifact"]["id"],
                "raw_score_rows": provenance["score_rows"],
                "second_model_inference_authorized": 0,
                "validation_rows_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "next_checkpoint_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

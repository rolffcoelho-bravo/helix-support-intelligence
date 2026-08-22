"""Fail-closed preflight for A4.5b-M2 SCEC calibration protocol registration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm2_v1.json"
MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm2_manifest_v1.json"
BUILDER = ROOT / "benchmarks" / "assistance" / "scec_calibration_a45bm2.py"
SUMMARY = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a45bm2_protocol_v1"
    / "registration_summary.json"
)
M1 = ROOT / "configs" / "models" / "assistance_grounding_a45bm1_v1.json"
A45B_CLOSED = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a45b_calibration_postresult_v1"
    / "result_summary.json"
)
SOURCE_MAIN_SHA = "1b3a0cd1a0e552bef6ae33b44969d67b11dce7de"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("scec_calibration_a45bm2", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load A4.5b-M2 calibration builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = _json(CONFIG)
    frozen_manifest = _json(MANIFEST)
    summary = _json(SUMMARY)
    m1 = _json(M1)
    closed = _json(A45B_CLOSED)

    if config["checkpoint"] != "A4.5b-M2":
        raise RuntimeError("A4.5b-M2 checkpoint identity drifted")
    if config["source_main_sha"] != SOURCE_MAIN_SHA:
        raise RuntimeError("A4.5b-M2 source main SHA drifted")
    if m1["selected_methodology"]["short_name"] != "SCEC":
        raise RuntimeError("A4.5b-M2 requires the frozen SCEC methodology")
    if m1["selected_methodology"]["binding_status"] != "UNBOUND":
        raise RuntimeError("SCEC must remain unbound before A4.5b-M2")
    if closed["scientific_pass"] is not False:
        raise RuntimeError("A4.5b-M2 requires the closed negative A4.5b result")

    module = _builder_module()
    current_manifest = module.manifest()
    if current_manifest != frozen_manifest:
        raise RuntimeError("A4.5b-M2 fresh calibration manifest drifted")

    suite = module.build_suite()
    if len(suite["units"]) != 48:
        raise RuntimeError("A4.5b-M2 unit count drifted")
    if len(suite["pair_rows"]) != 768:
        raise RuntimeError("A4.5b-M2 pair count drifted")
    if len(suite["evidence_set_rows"]) != 384:
        raise RuntimeError("A4.5b-M2 evidence-set count drifted")
    if len(suite["claim_rows"]) != 384:
        raise RuntimeError("A4.5b-M2 claim count drifted")
    if any(row["split"] != "calibration" for row in suite["pair_rows"]):
        raise RuntimeError("A4.5b-M2 pair rows must be calibration-only")
    if any(row["split"] != "calibration" for row in suite["evidence_set_rows"]):
        raise RuntimeError("A4.5b-M2 evidence sets must be calibration-only")
    if any(row["split"] != "calibration" for row in suite["claim_rows"]):
        raise RuntimeError("A4.5b-M2 claim rows must be calibration-only")

    if config["methodology"]["binding_status"] != "UNBOUND":
        raise RuntimeError("A4.5b-M2 cannot bind an implementation")
    for name, value in config["scope"].items():
        if int(value) != 0:
            raise RuntimeError(f"A4.5b-M2 forbidden execution scope is nonzero: {name}")

    governance = config["data_governance"]
    if governance["a45a_fresh_validation_remains_sealed"] is not True:
        raise RuntimeError("A4.5a fresh validation must remain sealed")
    if governance["a45a_fresh_validation_pairs_scored"] != 0:
        raise RuntimeError("A4.5a fresh validation pairs must remain unscored")
    if governance["confirmatory_records_inspected"] != 0:
        raise RuntimeError("Confirmatory records must remain unopened")
    if governance["a45c_repurposed_for_scec"] is not False:
        raise RuntimeError("A4.5c must not be repurposed for SCEC")

    if summary["status"] != "CLOSED_PROTOCOL_REGISTERED_NO_INFERENCE":
        raise RuntimeError("A4.5b-M2 registration summary status drifted")
    if summary["next_action"]["authorized"] is not False:
        raise RuntimeError("A4.5b-M3 requires separate approval")

    print(
        json.dumps(
            {
                "status": "PASSED_A45BM2_SCEC_CALIBRATION_PROTOCOL_NO_INFERENCE",
                "calibration_units": 48,
                "pair_rows": 768,
                "evidence_set_rows": 384,
                "claim_rows": 384,
                "model_bindings_authorized": 0,
                "semantic_inference_authorized": 0,
                "threshold_searches_authorized": 0,
                "fresh_validation_authorized": 0,
                "confirmatory_queries_authorized": 0,
                "next_action_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

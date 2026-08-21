"""Fail-closed no-inference preflight for Phase 4 A4.5a."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45a_v1.json"
GENERATOR = ROOT / "benchmarks" / "assistance" / "aerf_validity_a45a.py"
MANIFEST = ROOT / "benchmarks" / "assistance" / "a45a_manifest_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object in {path}")
    return value


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("aerf_validity_a45a", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load A4.5a generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = _load_json(CONFIG)
    expected_manifest = _load_json(MANIFEST)
    module = _load_generator()
    observed_manifest = module.manifest()
    if observed_manifest != expected_manifest:
        raise RuntimeError("A4.5a deterministic manifest drifted")
    if config["source_main_sha"] != "4297d97012549573c107d7165c98f47db57533da":
        raise RuntimeError("A4.5a must descend from the frozen A4.4e closure")
    fresh = config["fresh_validity_construction"]
    if fresh["validation_pairs_sha256"] != observed_manifest["sha256"]["validation_pairs"]:
        raise RuntimeError("A4.5a validation-pair hash mismatch")
    if fresh["validation_claims_sha256"] != observed_manifest["sha256"]["validation_claims"]:
        raise RuntimeError("A4.5a validation-claim hash mismatch")
    if fresh["a44d_rows_reused"] != 0 or observed_manifest["a44d_rows_reused"] != 0:
        raise RuntimeError("A4.5a may not reuse A4.4d rows")
    scope = config["scope"]
    if any(int(value) != 0 for value in scope.values()):
        raise RuntimeError("A4.5a is registration-only and cannot authorize execution")
    boundary = config["confirmatory_boundary"]
    if boundary["confirmatory_query_records_inspected"] != 0:
        raise RuntimeError("A4.5a confirmatory partition must remain unopened")
    if boundary["confirmatory_scoring_authorized"] != 0:
        raise RuntimeError("A4.5a confirmatory scoring must remain unauthorized")
    if config["architecture"]["status"] != "MEASUREMENT_REGISTERED_MODEL_UNBOUND":
        raise RuntimeError("A4.5a must leave the authoritative implementation unbound")
    if config["next_checkpoint"]["authorized_by_a45a"] is not False:
        raise RuntimeError("A4.5b requires separate approval")
    print(
        json.dumps(
            {
                "status": "PASSED_A45A_AERF_VALIDITY_PROTOCOL_NO_INFERENCE",
                "fresh_validation_pairs": fresh["validation_pair_rows"],
                "fresh_validation_claims": fresh["validation_claim_rows"],
                "a44d_rows_reused": fresh["a44d_rows_reused"],
                "semantic_inference_authorized": scope["semantic_inference_authorized"],
                "model_bindings_authorized": scope["model_bindings_authorized"],
                "confirmatory_queries_authorized": boundary["confirmatory_scoring_authorized"],
                "next_checkpoint_authorized": config["next_checkpoint"]["authorized_by_a45a"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

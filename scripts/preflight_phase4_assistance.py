"""Validate the frozen A4.1 assistance binding without scoring candidates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

from helix_support_intelligence.data.helixbank import INTENTS, generate_bundle, manifest

ROOT = Path(__file__).resolve().parents[1]
A40_PATH = ROOT / "configs" / "models" / "assistance_protocol_v1.json"
A41_PATH = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"
SUBSETS_PATH = ROOT / "configs" / "models" / "assistance_a41_subsets_v1.json"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return cast(dict[str, Any], payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _expected_subset(
    case_quotas: dict[str, int],
    development_intents: set[str],
) -> list[str]:
    bundle = generate_bundle()
    by_case: dict[str, list[str]] = {}
    for row in bundle.queries:
        if str(row["intent"]) not in development_intents:
            continue
        case_type = str(row["case_type"])
        by_case.setdefault(case_type, []).append(str(row["query_id"]))

    selected: list[str] = []
    for case_type, quota in case_quotas.items():
        ordered = sorted(
            by_case[case_type],
            key=lambda query_id: hashlib.sha256(
                f"A4.1-diagnostics-v1:{query_id}".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[:quota])
    return selected


def main() -> None:
    a40 = _load(A40_PATH)
    binding = _load(A41_PATH)
    subsets = _load(SUBSETS_PATH)

    if a40["protocol_id"] != "phase4-assistance-a4.0-v1":
        raise RuntimeError("Unexpected A4.0 protocol.")
    if binding["protocol_id"] != a40["protocol_id"]:
        raise RuntimeError("A4.1 is not linked to the frozen A4.0 protocol.")
    if binding["status"] != "implemented_preflight_no_results":
        raise RuntimeError("A4.1 dependency lock has not been frozen.")

    current_manifest = manifest()
    if a40["source_state"]["corpus"]["sha256"] != current_manifest["sha256"]:
        raise RuntimeError("Frozen HelixBank hashes changed.")

    development, confirmatory = _intent_partition()
    if len(development) != 60 or len(confirmatory) != 17:
        raise RuntimeError("A4.0 intent partition does not reconstruct.")

    bundle = generate_bundle()
    dev_query_ids = {
        str(row["query_id"]) for row in bundle.queries if str(row["intent"]) in development
    }
    confirm_query_ids = {
        str(row["query_id"]) for row in bundle.queries if str(row["intent"]) in confirmatory
    }
    if len(dev_query_ids) != 240 or len(confirm_query_ids) != 68:
        raise RuntimeError("A4.0 query partition does not reconstruct.")

    prompts = binding["prompts"]
    for key in ("system", "request_template"):
        spec = prompts[key]
        path = ROOT / spec["path"]
        if _sha256(path) != spec["sha256"]:
            raise RuntimeError(f"{key} prompt hash mismatch.")
        if len(path.read_bytes()) != spec["bytes"]:
            raise RuntimeError(f"{key} prompt byte length mismatch.")
        if not path.read_bytes().endswith(b"\n"):
            raise RuntimeError(f"{key} prompt must end with LF.")

    structured = binding["generator"]["structured_output"]
    output_schema_path = ROOT / structured["post_parse_schema_path"]
    if _sha256(output_schema_path) != structured["post_parse_schema_sha256"]:
        raise RuntimeError("Candidate output schema hash mismatch.")

    subset_spec = binding["diagnostic_subsets"]
    if _sha256(SUBSETS_PATH) != subset_spec["sha256"]:
        raise RuntimeError("Diagnostic subset file hash mismatch.")

    selection = subsets["selection"]
    repeatability = selection["repeatability"]
    latency = selection["latency"]
    expected_repeatability = _expected_subset(repeatability["quotas"], development)
    expected_latency = _expected_subset(latency["quotas"], development)
    if repeatability["query_ids"] != expected_repeatability:
        raise RuntimeError("Repeatability subset is not the frozen deterministic selection.")
    if latency["query_ids"] != expected_latency:
        raise RuntimeError("Latency subset is not the frozen deterministic selection.")

    repeat_ids = set(repeatability["query_ids"])
    latency_ids = set(latency["query_ids"])
    if not repeat_ids <= latency_ids:
        raise RuntimeError("Repeatability subset must be contained in latency subset.")
    if not latency_ids <= dev_query_ids:
        raise RuntimeError("A4.1 diagnostics contain non-development queries.")
    if latency_ids & confirm_query_ids:
        raise RuntimeError("A4.1 diagnostics opened confirmatory queries.")

    runtime_verifier = binding["runtime_verifier"]
    evaluator = binding["evaluation_verifier"]
    if runtime_verifier["architecture_family"] == evaluator["architecture_family"]:
        raise RuntimeError("Runtime and evaluation verifier families must differ.")
    for verifier in (runtime_verifier, evaluator):
        if FULL_SHA_RE.fullmatch(str(verifier["revision"])) is None:
            raise RuntimeError("NLI verifier revision must be an exact 40-character SHA.")
        if verifier["entailment_label"] != 1:
            raise RuntimeError("Frozen NLI entailment label changed.")
        if verifier["entailment_threshold"] != 0.8:
            raise RuntimeError("Frozen NLI threshold changed.")
        if verifier["max_length"] != 512 or verifier["batch_size"] != 8:
            raise RuntimeError("Frozen NLI runtime setting changed.")

    generator = binding["generator"]
    if generator["model"] != "gpt-5.4-mini-2026-03-17":
        raise RuntimeError("Generator snapshot changed.")
    if generator["temperature"] != 0.0 or generator["max_output_tokens"] != 512:
        raise RuntimeError("Frozen generator decoding changed.")
    if generator["reasoning_effort"] != "none":
        raise RuntimeError("Frozen generator reasoning effort changed.")
    if generator["tools"] != [] or generator["store"] is not False:
        raise RuntimeError("Generator tools/storage boundary changed.")

    budgets = binding["budgets"]
    if budgets["p95_latency_ms"] != {"G0": 100, "G1": 6000, "G2": 8000}:
        raise RuntimeError("A4.1 latency budgets changed.")
    if budgets["maximum_estimated_cost_usd_per_request"] != {
        "G0": 0.0,
        "G1": 0.005,
        "G2": 0.005,
    }:
        raise RuntimeError("A4.1 cost budgets changed.")

    pricing = binding["pricing_snapshot"]
    if pricing["snapshot_date"] != "2026-08-19":
        raise RuntimeError("Pricing snapshot date changed.")
    if pricing["openai_standard_usd_per_1m_tokens"] != {
        "input": 0.75,
        "cached_input": 0.075,
        "output": 4.5,
    }:
        raise RuntimeError("Frozen provider pricing changed.")

    runtime = binding["benchmark_runtime"]
    runtime_path = ROOT / runtime["script"]
    lock_path = ROOT / runtime["lock"]
    if _sha256(runtime_path) != runtime["script_sha256"]:
        raise RuntimeError("A4.1 benchmark runtime hash mismatch.")
    if _sha256(lock_path) != runtime["lock_sha256"]:
        raise RuntimeError("A4.1 benchmark dependency lock hash mismatch.")

    guard = binding["results_guard"]
    if guard["development_scores_computed"] != 0:
        raise RuntimeError("A4.1 must not open development scores.")
    if guard["confirmatory_scores_computed"] != 0:
        raise RuntimeError("A4.1 must not open confirmatory scores.")
    if guard["generator_calls_made_by_a41_preflight"] != 0:
        raise RuntimeError("A4.1 preflight must not call the generator.")
    if guard["nli_calls_made_by_a41_preflight"] != 0:
        raise RuntimeError("A4.1 preflight must not call NLI models.")
    if guard["assistance_performance_results_generated"] is not False:
        raise RuntimeError("A4.1 must remain result-free.")

    result = {
        "binding_id": binding["binding_id"],
        "protocol_id": binding["protocol_id"],
        "development_intents": len(development),
        "development_queries": len(dev_query_ids),
        "confirmatory_intents": len(confirmatory),
        "confirmatory_queries": len(confirm_query_ids),
        "generator_model": generator["model"],
        "runtime_verifier_model": runtime_verifier["model_id"],
        "evaluation_verifier_model": evaluator["model_id"],
        "repeatability_cases": len(repeat_ids),
        "latency_cases": len(latency_ids),
        "generator_calls_made": 0,
        "nli_calls_made": 0,
        "assistance_scores_computed": 0,
        "status": "passed",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

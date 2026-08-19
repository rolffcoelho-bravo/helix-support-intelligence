# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.28,<1",
#   "huggingface-hub>=0.34,<1",
#   "numpy>=2.1,<3",
#   "onnxruntime>=1.22,<2",
#   "sentencepiece>=0.2,<1",
#   "transformers>=4.55,<5",
# ]
# ///
"""A4.1 benchmark-scoped assistance runtime bindings.

The preflight path is intentionally network-free and does not run generation,
verification, evaluation, or benchmark scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BINDING_PATH = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "assistance" / "system_v1.txt"
REQUEST_TEMPLATE_PATH = ROOT / "prompts" / "assistance" / "request_template_v1.txt"
OUTPUT_SCHEMA_PATH = (
    ROOT / "data" / "contracts" / "phase4" / "assistance_candidate_output.schema.json"
)

CANDIDATE_DOCUMENT_FIELDS = (
    "document_id",
    "title",
    "body",
    "kind",
    "status",
    "valid_from",
    "valid_to",
    "permission",
    "resolution_type",
)


@dataclass(frozen=True, slots=True)
class NliBinding:
    """Frozen local NLI model binding."""

    model_id: str
    revision: str
    architecture_family: str
    onnx_path: str
    entailment_label: int
    threshold: float
    max_length: int
    batch_size: int


def sha256_path(path: Path) -> str:
    """Return SHA-256 for exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_binding() -> dict[str, Any]:
    """Load the machine-readable A4.1 binding."""
    payload = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("A4.1 binding must be a JSON object.")
    return payload


def canonical_evidence_json(documents: list[dict[str, Any]]) -> str:
    """Render only candidate-visible document fields deterministically."""
    sanitized: list[dict[str, Any]] = []
    for document in documents:
        sanitized.append({key: document.get(key) for key in CANDIDATE_DOCUMENT_FIELDS})
    sanitized.sort(key=lambda item: str(item["document_id"]))
    return json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def render_user_input(query: str, documents: list[dict[str, Any]]) -> str:
    """Render the frozen user-input template."""
    template = REQUEST_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(query=query, evidence_json=canonical_evidence_json(documents))


def provider_output_schema() -> dict[str, Any]:
    """Return the provider-compatible structured-output schema.

    Full post-parse validation uses the repository JSON Schema. This provider
    schema keeps only the structural subset required for strict Responses API
    structured output.
    """
    decisions = [
        "ANSWER_WITH_EVIDENCE",
        "AUTO_ROUTE",
        "RECOMMEND_TO_AGENT",
        "ASK_FOR_CLARIFICATION",
        "ESCALATE_LOW_CONFIDENCE",
        "ESCALATE_OUT_OF_SCOPE",
        "ESCALATE_CONFLICTING_EVIDENCE",
        "ESCALATE_SAFETY_RISK",
        "ESCALATE_SYSTEM_FAILURE",
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "answer", "citations", "escalation_reason"],
        "properties": {
            "decision": {"type": "string", "enum": decisions},
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
            "escalation_reason": {"type": ["string", "null"]},
        },
    }


def build_openai_payload(query: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the exact deterministic G1/G2 generator request payload."""
    binding = load_binding()
    generator = binding["generator"]
    return {
        "model": generator["model"],
        "store": False,
        "tools": [],
        "reasoning": {"effort": generator["reasoning_effort"]},
        "temperature": generator["temperature"],
        "max_output_tokens": generator["max_output_tokens"],
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": render_user_input(query, documents),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "helix_assistance",
                "strict": True,
                "schema": provider_output_schema(),
            }
        },
    }


def estimate_generator_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate selection cost using the frozen uncached standard rate."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative.")
    pricing = load_binding()["pricing_snapshot"]["openai_standard_usd_per_1m_tokens"]
    return (
        input_tokens * float(pricing["input"]) / 1_000_000
        + output_tokens * float(pricing["output"]) / 1_000_000
    )


def runtime_verifier_binding() -> NliBinding:
    """Return the frozen G2 runtime verifier identity."""
    payload = load_binding()["runtime_verifier"]
    return NliBinding(
        model_id=payload["model_id"],
        revision=payload["revision"],
        architecture_family=payload["architecture_family"],
        onnx_path=payload["onnx_path"],
        entailment_label=payload["entailment_label"],
        threshold=payload["entailment_threshold"],
        max_length=payload["max_length"],
        batch_size=payload["batch_size"],
    )


def evaluation_verifier_binding() -> NliBinding:
    """Return the frozen independent evaluation verifier identity."""
    payload = load_binding()["evaluation_verifier"]
    return NliBinding(
        model_id=payload["model_id"],
        revision=payload["revision"],
        architecture_family=payload["architecture_family"],
        onnx_path=payload["onnx_path"],
        entailment_label=payload["entailment_label"],
        threshold=payload["entailment_threshold"],
        max_length=payload["max_length"],
        batch_size=payload["batch_size"],
    )


def preflight() -> dict[str, Any]:
    """Run a local-only binding sanity check without opening any result."""
    binding = load_binding()
    runtime_verifier = runtime_verifier_binding()
    evaluator = evaluation_verifier_binding()

    if runtime_verifier.architecture_family == evaluator.architecture_family:
        raise RuntimeError("Runtime and evaluation verifier families must differ.")
    if binding["results_guard"]["development_scores_computed"] != 0:
        raise RuntimeError("Development scores must remain unopened at A4.1.")
    if binding["results_guard"]["confirmatory_scores_computed"] != 0:
        raise RuntimeError("Confirmatory scores must remain unopened at A4.1.")

    sample_document = {
        "document_id": "POLICY-001",
        "title": "Example",
        "body": "Example evidence.",
        "kind": "policy",
        "status": "current",
        "valid_from": "2026-02-01",
        "valid_to": None,
        "permission": "public_support",
        "resolution_type": "provide_policy_guidance",
        "intent": "must_not_leak",
        "gold_citations": ["must_not_leak"],
    }
    rendered = canonical_evidence_json([sample_document])
    if "must_not_leak" in rendered:
        raise RuntimeError("Candidate evidence rendering leaked evaluator-only fields.")

    payload = build_openai_payload("Example request", [sample_document])
    return {
        "binding_id": binding["binding_id"],
        "model": payload["model"],
        "system_prompt_sha256": sha256_path(SYSTEM_PROMPT_PATH),
        "request_template_sha256": sha256_path(REQUEST_TEMPLATE_PATH),
        "output_schema_sha256": sha256_path(OUTPUT_SCHEMA_PATH),
        "runtime_verifier_family": runtime_verifier.architecture_family,
        "evaluation_verifier_family": evaluator.architecture_family,
        "generator_calls_made": 0,
        "nli_calls_made": 0,
        "performance_scores_computed": 0,
        "status": "passed",
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight:
        raise SystemExit("A4.1 runtime permits only --preflight; scoring is not implemented.")
    print(json.dumps(preflight(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

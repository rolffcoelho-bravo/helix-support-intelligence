# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "huggingface-hub==0.36.2",
#   "safetensors==0.8.0",
#   "sentencepiece==0.2.1",
#   "torch==2.13.0",
#   "transformers==4.57.6",
# ]
# ///
"""Execute the frozen A4.5b-M6 TPAG calibration-only experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from tpag_calibration_a45bm5 import build_suite, manifest
from tpag_core_a45bm6 import (
    candidate_record,
    collect_residual_requests,
    make_alias_map,
    probability_triplet,
    select_candidate,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm6_v1.json"
M5_CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm5_v1.json"
M5_MANIFEST = ROOT / "benchmarks" / "assistance" / "a45bm5_manifest_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_and_load_model(config: dict[str, Any]) -> tuple[dict[str, str], Any, Any]:
    binding = config["authoritative_implementation"]["semantic_model"]
    model_path = Path(
        hf_hub_download(
            repo_id=str(binding["model_id"]),
            filename=str(binding["weights_file"]),
            revision=str(binding["revision"]),
        )
    )
    tokenizer_path = Path(
        hf_hub_download(
            repo_id=str(binding["model_id"]),
            filename=str(binding["tokenizer_file"]),
            revision=str(binding["revision"]),
        )
    )
    observed_model = _sha256(model_path)
    observed_tokenizer = _sha256(tokenizer_path)
    if observed_model != str(binding["weights_sha256"]):
        raise RuntimeError(
            f"A4.5b-M6 model weight hash mismatch: {observed_model} != {binding['weights_sha256']}"
        )
    if observed_tokenizer != str(binding["tokenizer_sha256"]):
        raise RuntimeError(
            "A4.5b-M6 tokenizer hash mismatch: "
            f"{observed_tokenizer} != {binding['tokenizer_sha256']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(binding["model_id"]), revision=str(binding["revision"])
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(binding["model_id"]),
        revision=str(binding["revision"]),
        use_safetensors=True,
        dtype=torch.float32,
    ).to("cpu")
    observed_labels = {
        str(index): str(label).lower() for index, label in model.config.id2label.items()
    }
    expected_labels = {
        str(index): str(label).lower() for index, label in binding["native_labels"].items()
    }
    if observed_labels != expected_labels:
        raise RuntimeError(
            f"A4.5b-M6 native label mapping drifted: {observed_labels} != {expected_labels}"
        )
    model.eval()
    return (
        {
            "model_id": str(binding["model_id"]),
            "revision": str(binding["revision"]),
            "weights_file": str(binding["weights_file"]),
            "weights_sha256": observed_model,
            "tokenizer_file": str(binding["tokenizer_file"]),
            "tokenizer_sha256": observed_tokenizer,
            "license": str(binding["license"]),
        },
        tokenizer,
        model,
    )


def _score_residual_requests(
    requests: list[dict[str, str]], tokenizer: Any, model: Any, config: dict[str, Any]
) -> list[dict[str, Any]]:
    binding = config["authoritative_implementation"]["semantic_model"]
    query_template = str(binding["query_template"])
    hypothesis = str(binding["hypothesis"])
    batch_size = int(binding["batch_size"])
    max_length = int(binding["max_sequence_length"])
    premises = [
        query_template.format(
            claim_value=request["claim_value"],
            evidence_value=request["evidence_value"],
            evidence_text=request["evidence_text"],
        )
        for request in requests
    ]
    hypotheses = [hypothesis for _ in requests]
    output: list[dict[str, Any]] = []
    for start in range(0, len(requests), batch_size):
        stop = min(start + batch_size, len(requests))
        encoded = tokenizer(
            premises[start:stop],
            hypotheses[start:stop],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**encoded).logits.detach().cpu().tolist()
        if any(len(row) != 3 for row in logits):
            raise RuntimeError("A4.5b-M6 bound residual model must emit three logits")
        for request, row in zip(requests[start:stop], logits, strict=True):
            probabilities = probability_triplet([float(value) for value in row])
            output.append(
                {
                    **request,
                    "logits": {
                        "contradiction": float(row[0]),
                        "entailment": float(row[1]),
                        "neutral": float(row[2]),
                    },
                    "probabilities": probabilities,
                }
            )
    output.sort(key=lambda row: str(row["request_id"]))
    return output


def _environment() -> dict[str, Any]:
    packages = (
        "huggingface-hub",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers",
        "protobuf",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _report(results: dict[str, Any]) -> str:
    selected = results["selected_candidate"]
    lines = [
        "# A4.5b-M6 TPAG calibration-only execution",
        "",
        f"Scientific status: `{results['scientific_status']}`",
        "",
        f"Residual learned requests: **{results['residual_request_count']}**",
        f"Registered threshold candidates: **{results['candidate_count']}**",
        f"Feasible candidates: **{results['feasible_candidate_count']}**",
        f"Selected alignment confidence threshold: **{selected['alignment_confidence_min']}**",
        f"Readiness requirements passed: **{selected['requirements_passed']}/{selected['requirements_total']}**",
        "",
        "## Selected metrics",
        "",
    ]
    for name, value in sorted(selected["metrics"].items()):
        lines.append(f"- `{name}`: {value}")
    lines.extend(
        [
            "",
            "## Governance",
            "",
            "- A4.5a fresh validation rows scored: **0**",
            "- Confirmatory queries scored: **0**",
            "- Closed M2/M3 rows rescored: **0**",
            "- Model-family comparisons: **0**",
            "- Prompt searches: **0**",
            "- Post-result rescue: **0**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(CONFIG)
    m5_config = _load_json(M5_CONFIG)
    frozen_manifest = _load_json(M5_MANIFEST)
    if manifest() != frozen_manifest:
        raise RuntimeError("A4.5b-M6 M5 calibration manifest drifted")
    if config["source_main_sha"] != "9070fc4cf0447077a20c7e576e49e9ba5f0536ba":
        raise RuntimeError("A4.5b-M6 source main SHA drifted")
    if config["authoritative_implementation"]["count"] != 1:
        raise RuntimeError("A4.5b-M6 requires exactly one authoritative implementation")
    if len(m5_config["calibration_readiness_requirements"]) != 56:
        raise RuntimeError("A4.5b-M6 requires the frozen 56 M5 readiness requirements")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    suite = build_suite()
    aliases = make_alias_map(suite["units"])
    requests = collect_residual_requests(suite, aliases)
    model_verification, tokenizer, model = _verify_and_load_model(config)
    raw_rows = _score_residual_requests(requests, tokenizer, model, config)

    raw_path = output_dir / "residual_raw_scores.jsonl"
    _write_jsonl(raw_path, raw_rows)
    raw_manifest = {
        "status": "RAW_LEARNED_OUTPUTS_FROZEN_BEFORE_GOLD_EVALUATION",
        "residual_request_count": len(raw_rows),
        "raw_scores_sha256": _sha256(raw_path),
        "gold_read_for_inference": false,
        "a45a_fresh_validation_rows_scored": 0,
        "confirmatory_queries_scored": 0,
        "closed_a45bm2_m3_rows_scored": 0
    }
    _write_json(output_dir / "raw_inference_manifest.json", raw_manifest)
    _write_json(output_dir / "model_weight_verification.json", model_verification)
    _write_json(output_dir / "environment.json", _environment())

    raw_scores = {
        str(row["request_id"]): {
            key: float(value) for key, value in row["probabilities"].items()
        }
        for row in raw_rows
    }
    thresholds = [
        float(value)
        for value in config["calibration_parameter_grid"]["alignment_confidence_values"]
    ]
    requirements = {
        str(name): float(value)
        for name, value in m5_config["calibration_readiness_requirements"].items()
    }
    candidates = [
        candidate_record(suite, raw_scores, threshold, requirements) for threshold in thresholds
    ]
    selected = select_candidate(candidates)
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    scientific_pass = bool(feasible)
    scientific_status = str(
        config["scientific_status_labels"]["pass" if scientific_pass else "fail"]
    )
    results = {
        "protocol_id": str(config["protocol_id"]),
        "scientific_status": scientific_status,
        "scientific_pass": scientific_pass,
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "residual_request_count": len(raw_rows),
        "selected_candidate": selected,
        "candidates": candidates,
        "raw_scores_sha256": _sha256(raw_path),
        "model": model_verification,
        "sealed_boundaries": {
            "a45a_fresh_validation_rows_scored": 0,
            "confirmatory_queries_scored": 0,
            "confirmatory_records_inspected": 0,
            "closed_a45bm2_m3_rows_scored": 0,
            "future_validation_rows_constructed": 0,
            "post_result_rescue_authorized": false,
            "next_checkpoint_authorized": false
        }
    }
    _write_json(output_dir / "results.json", results)
    (output_dir / "report.md").write_text(_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

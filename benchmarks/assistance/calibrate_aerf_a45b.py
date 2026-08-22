# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "huggingface-hub==0.36.2",
#   "numpy==2.5.2",
#   "safetensors==0.8.0",
#   "sentencepiece==0.2.1",
#   "torch==2.13.0",
#   "transformers==4.57.6",
# ]
# ///
"""Execute the frozen A4.5b AERF calibration-only experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aerf_calibration_core_a45b import select_thresholds
from calibration_cases_a45b import build_calibration_only, calibration_manifest
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _load_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("A4.5b config must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_weight(binding: dict[str, Any]) -> dict[str, str]:
    path = Path(
        hf_hub_download(
            repo_id=str(binding["model_id"]),
            filename=str(binding["weights_file"]),
            revision=str(binding["revision"]),
        )
    )
    observed = _sha256(path)
    expected = str(binding["weights_sha256"])
    if observed != expected:
        raise RuntimeError(
            f"Weight hash mismatch for {binding['model_id']}: {observed} != {expected}"
        )
    return {
        "model_id": str(binding["model_id"]),
        "revision": str(binding["revision"]),
        "weights_file": str(binding["weights_file"]),
        "weights_sha256": observed,
    }


def _sentences(text: str) -> list[str]:
    spans = [item.strip() for item in SENTENCE_SPLIT_RE.split(text.strip()) if item.strip()]
    return spans or [text.strip()]


def _model_logits(
    tokenizer: Any,
    model: Any,
    left: list[str],
    right: list[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    rows: list[list[float]] = []
    for start in range(0, len(left), batch_size):
        stop = start + batch_size
        encoded = tokenizer(
            left[start:stop],
            right[start:stop],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**encoded).logits.detach().cpu()
        rows.extend([[float(value) for value in row] for row in logits.tolist()])
    return np.asarray(rows, dtype=np.float64)


def _load_models(config: dict[str, Any]) -> tuple[dict[str, Any], Any, Any, Any, Any]:
    binding = config["binding"]
    verification = {
        "alignment_relevance": _verify_weight(binding["alignment_relevance"]),
        "sufficiency_polarity": _verify_weight(binding["sufficiency_polarity"]),
    }
    relevance = binding["alignment_relevance"]
    relevance_tokenizer = AutoTokenizer.from_pretrained(
        relevance["model_id"], revision=relevance["revision"]
    )
    relevance_model = AutoModelForSequenceClassification.from_pretrained(
        relevance["model_id"],
        revision=relevance["revision"],
        use_safetensors=True,
        dtype=torch.float32,
    ).to("cpu")
    relevance_model.eval()

    nli = binding["sufficiency_polarity"]
    nli_tokenizer = AutoTokenizer.from_pretrained(nli["model_id"], revision=nli["revision"])
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        nli["model_id"],
        revision=nli["revision"],
        use_safetensors=True,
        dtype=torch.float32,
    ).to("cpu")
    observed = {
        str(index): str(label).lower() for index, label in nli_model.config.id2label.items()
    }
    expected = {
        str(index): str(label).lower() for index, label in nli["native_labels"].items()
    }
    if observed != expected:
        raise RuntimeError(f"A4.5b NLI label mapping drifted: {observed}")
    nli_model.eval()
    return verification, relevance_tokenizer, relevance_model, nli_tokenizer, nli_model


def _score_pairs(
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
    relevance_tokenizer: Any,
    relevance_model: Any,
    nli_tokenizer: Any,
    nli_model: Any,
) -> list[dict[str, Any]]:
    relevance = config["binding"]["alignment_relevance"]
    claims: list[str] = []
    spans: list[str] = []
    ranges: list[tuple[int, int, list[str]]] = []
    for pair in pairs:
        local_spans = _sentences(str(pair["evidence_text"]))
        start = len(spans)
        claims.extend([str(pair["claim"])] * len(local_spans))
        spans.extend(local_spans)
        ranges.append((start, len(spans), local_spans))
    relevance_logits = _model_logits(
        relevance_tokenizer,
        relevance_model,
        claims,
        spans,
        int(relevance["batch_size"]),
        int(relevance["max_sequence_length"]),
    )
    if relevance_logits.ndim != 2 or relevance_logits.shape[1] != 1:
        raise RuntimeError("A4.5b relevance model must emit one logit per pair")

    selected_spans: list[str] = []
    selected_scores: list[float] = []
    all_span_scores: list[list[dict[str, Any]]] = []
    for start, stop, local_spans in ranges:
        local_scores = relevance_logits[start:stop, 0]
        winner = int(np.argmax(local_scores))
        selected_spans.append(local_spans[winner])
        selected_scores.append(float(local_scores[winner]))
        all_span_scores.append(
            [
                {"span": span, "relevance_score": float(score)}
                for span, score in zip(local_spans, local_scores.tolist(), strict=True)
            ]
        )

    nli = config["binding"]["sufficiency_polarity"]
    nli_logits = _model_logits(
        nli_tokenizer,
        nli_model,
        selected_spans,
        [str(pair["claim"]) for pair in pairs],
        int(nli["batch_size"]),
        int(nli["max_sequence_length"]),
    )
    if nli_logits.ndim != 2 or nli_logits.shape[1] != 3:
        raise RuntimeError("A4.5b NLI model must emit three logits per pair")
    probabilities = torch.softmax(torch.from_numpy(nli_logits), dim=-1).numpy()

    output: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        output.append(
            {
                "pair_id": str(pair["pair_id"]),
                "unit_id": str(pair["unit_id"]),
                "split": "calibration",
                "subtype": str(pair["subtype"]),
                "gold": pair["gold"],
                "selected_span": selected_spans[index],
                "relevance_score": selected_scores[index],
                "span_scores": all_span_scores[index],
                "nli_probabilities": {
                    "entailment": float(probabilities[index, 0]),
                    "neutral": float(probabilities[index, 1]),
                    "contradiction": float(probabilities[index, 2]),
                },
            }
        )
    return output


def _claim_composition_accuracy(claims: list[dict[str, Any]]) -> float:
    correct = 0
    for row in claims:
        gate = str(row["deterministic_gate"])
        relations = [str(value) for value in row["atom_relations"]]
        if gate == "CITATION_INVALID":
            predicted = "CITATION_INVALID"
        elif gate == "STALE_EVIDENCE":
            predicted = "STALE_EVIDENCE"
        elif gate == "REGISTERED_CONFLICT" or (
            "ENTAILED" in relations and "CONTRADICTED" in relations
        ):
            predicted = "CONFLICTING_EVIDENCE"
        elif relations and all(value == "ENTAILED" for value in relations):
            predicted = "SUPPORTED"
        else:
            predicted = "UNSUPPORTED"
        correct += predicted == row["expected_verdict"]
    return correct / len(claims)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _environment() -> dict[str, Any]:
    names = (
        "huggingface-hub",
        "numpy",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in names},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _report(results: dict[str, Any]) -> str:
    selected = results["threshold_selection"]["selected"]
    metrics = selected["metrics"]
    lines = [
        "# A4.5b calibration-only result",
        "",
        f"Scientific status: **{results['scientific_status']}**",
        "",
        f"Calibration pairs: **{results['calibration_pairs']}**",
        f"Validation rows materialized: **{results['validation_rows_materialized']}**",
        f"Confirmatory queries inspected: **{results['confirmatory_queries_inspected']}**",
        "",
        "## Frozen thresholds",
        "",
        f"- relevance logit threshold: `{selected['relevance_threshold']}`",
        f"- sufficiency probability threshold: `{selected['sufficiency_threshold']}`",
        "",
        "## Calibration readiness",
        "",
        f"- final relation macro F1: `{metrics['final_relation_macro_f1']:.6f}`",
        f"- ENTAILED recall: `{metrics['entailed_recall']:.6f}`",
        f"- CONTRADICTED recall: `{metrics['contradicted_recall']:.6f}`",
        f"- UNKNOWN recall: `{metrics['unknown_recall']:.6f}`",
        f"- relevance macro F1: `{metrics['relevance_macro_f1']:.6f}`",
        f"- sufficiency macro F1: `{metrics['sufficiency_macro_f1_on_relevant_pairs']:.6f}`",
        f"- polarity macro F1: `{metrics['polarity_macro_f1_on_relevant_sufficient_pairs']:.6f}`",
        "",
        "Fresh validation and the 68-query confirmatory partition were not scored.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    if config["status"] != "REGISTERED_PRE_EXECUTION_CALIBRATION_ONLY":
        raise RuntimeError("A4.5b is not in registered pre-execution state")
    sealed = config["sealed_partitions"]
    if int(sealed["validation_scoring_authorized"]) != 0:
        raise RuntimeError("A4.5b fresh validation scoring is not authorized")
    if int(sealed["confirmatory_scoring_authorized"]) != 0:
        raise RuntimeError("A4.5b confirmatory scoring is not authorized")

    manifest = calibration_manifest()
    materialized = build_calibration_only()
    pairs = materialized["pair_rows"]
    claims = materialized["claim_rows"]
    if manifest["calibration_pairs"] != 360 or manifest["calibration_claims"] != 360:
        raise RuntimeError("A4.5b calibration cardinality drifted")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verification, relevance_tokenizer, relevance_model, nli_tokenizer, nli_model = _load_models(
        config
    )
    _write_json(args.output_dir / "model_weight_verification.json", verification)
    scores = _score_pairs(
        pairs,
        config,
        relevance_tokenizer,
        relevance_model,
        nli_tokenizer,
        nli_model,
    )
    _write_jsonl(args.output_dir / "calibration_pair_scores.jsonl", scores)

    claim_accuracy = _claim_composition_accuracy(claims)
    threshold_selection = select_thresholds(pairs, scores, config, claim_accuracy)
    selected = threshold_selection["selected"]
    scientific_pass = bool(selected["calibration_ready"])
    scientific_status = (
        "PASSED_CALIBRATION_READINESS_THRESHOLDS_FROZEN"
        if scientific_pass
        else "FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED"
    )
    results = {
        "binding_id": config["binding_id"],
        "checkpoint": "A4.5b",
        "scientific_pass": scientific_pass,
        "scientific_status": scientific_status,
        "calibration_units": manifest["calibration_units"],
        "calibration_pairs": manifest["calibration_pairs"],
        "calibration_claims": manifest["calibration_claims"],
        "calibration_pairs_sha256": manifest["calibration_pairs_sha256"],
        "calibration_claims_sha256": manifest["calibration_claims_sha256"],
        "threshold_selection": threshold_selection,
        "model_weight_verification": verification,
        "validation_rows_materialized": 0,
        "validation_rows_scored": 0,
        "confirmatory_queries_inspected": 0,
        "confirmatory_queries_scored": 0,
        "a44d_rows_rescored": 0,
        "a44a_rows_rescored": 0,
        "post_result_rescue_authorized": False,
        "next_checkpoint_authorized": False,
    }
    _write_json(args.output_dir / "results.json", results)
    _write_json(args.output_dir / "environment.json", _environment())
    (args.output_dir / "report.md").write_text(_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

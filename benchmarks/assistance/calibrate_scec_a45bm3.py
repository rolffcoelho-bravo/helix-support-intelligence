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
"""Execute the frozen A4.5b-M3 SCEC calibration-only experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from scec_calibration_a45bm2 import build_suite, manifest
from scec_calibration_core_a45bm3 import select_parameters
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45bm3_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exp_values = [math.exp(value - maximum) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _verify_and_load_model(config: dict[str, Any]) -> tuple[dict[str, str], Any, Any]:
    binding = config["authoritative_implementation"]["semantic_model"]
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
        raise RuntimeError(f"A4.5b-M3 model weight hash mismatch: {observed} != {expected}")
    tokenizer = AutoTokenizer.from_pretrained(binding["model_id"], revision=binding["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        binding["model_id"],
        revision=binding["revision"],
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
            f"A4.5b-M3 native label mapping drifted: {observed_labels} != {expected_labels}"
        )
    model.eval()
    verification = {
        "model_id": str(binding["model_id"]),
        "revision": str(binding["revision"]),
        "weights_file": str(binding["weights_file"]),
        "weights_sha256": observed,
        "license": str(binding["license"]),
    }
    return verification, tokenizer, model


def _expanded_entailment_logits(
    tokenizer: Any,
    model: Any,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int,
    max_length: int,
) -> list[float]:
    if len(premises) != len(hypotheses):
        raise RuntimeError("A4.5b-M3 expanded scoring arrays have different lengths")
    output: list[float] = []
    for start in range(0, len(premises), batch_size):
        stop = start + batch_size
        encoded = tokenizer(
            premises[start:stop],
            hypotheses[start:stop],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**encoded).logits.detach().cpu()
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError("A4.5b-M3 bound model must emit two logits")
        output.extend(float(row[0]) for row in logits.tolist())
    return output


def _score_registered_requests(
    tokenizer: Any,
    model: Any,
    contexts: list[tuple[str, str]],
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    implementation = config["authoritative_implementation"]
    binding = implementation["semantic_model"]
    dimension_hypotheses = implementation["compatibility"]["hypotheses"]
    covered_hypotheses = implementation["coverage"]["covered_hypotheses"]
    missing_hypotheses = implementation["coverage"]["missing_hypotheses"]
    polarity_hypotheses = implementation["polarity"]["hypotheses"]

    requests: list[tuple[tuple[str, str], str, str, str]] = []
    for context in contexts:
        for dimension, hypotheses in dimension_hypotheses.items():
            for label in ("MATCH", "MISMATCH", "UNSPECIFIED"):
                requests.append((context, f"dimension:{dimension}", label, hypotheses[label]))
        for slot in implementation["coverage"]["slots"]:
            requests.append((context, f"coverage:{slot}", "COVERED", covered_hypotheses[slot]))
            requests.append((context, f"coverage:{slot}", "MISSING", missing_hypotheses[slot]))
        requests.append((context, "polarity", "SUPPORTS", polarity_hypotheses["SUPPORTS"]))
        requests.append((context, "polarity", "REFUTES", polarity_hypotheses["REFUTES"]))

    premises = [
        implementation["raw_scoring"]["premise_template"].format(
            claim=context[0], evidence=context[1]
        )
        for context, _, _, _ in requests
    ]
    hypotheses = [hypothesis for _, _, _, hypothesis in requests]
    logits = _expanded_entailment_logits(
        tokenizer,
        model,
        premises,
        hypotheses,
        int(binding["batch_size"]),
        int(binding["max_sequence_length"]),
    )

    grouped: dict[tuple[tuple[str, str], str], list[tuple[str, float]]] = {}
    for request, logit in zip(requests, logits, strict=True):
        context, task, label, _ = request
        grouped.setdefault((context, task), []).append((label, logit))

    output: dict[tuple[str, str], dict[str, Any]] = {}
    for context in contexts:
        dimensions: dict[str, dict[str, float]] = {}
        coverage: dict[str, dict[str, float]] = {}
        for dimension in dimension_hypotheses:
            values = grouped[(context, f"dimension:{dimension}")]
            probabilities = _softmax([value for _, value in values])
            dimensions[dimension] = {
                label: probability
                for (label, _), probability in zip(values, probabilities, strict=True)
            }
        for slot in implementation["coverage"]["slots"]:
            values = grouped[(context, f"coverage:{slot}")]
            probabilities = _softmax([value for _, value in values])
            coverage[slot] = {
                label: probability
                for (label, _), probability in zip(values, probabilities, strict=True)
            }
        values = grouped[(context, "polarity")]
        probabilities = _softmax([value for _, value in values])
        polarity = {
            label: probability
            for (label, _), probability in zip(values, probabilities, strict=True)
        }
        output[context] = {
            "text": context[1],
            "dimensions": dimensions,
            "coverage": coverage,
            "polarity": polarity,
        }
    return output


def _score_polarity_only(
    tokenizer: Any,
    model: Any,
    contexts: list[tuple[str, str]],
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, float]]:
    implementation = config["authoritative_implementation"]
    binding = implementation["semantic_model"]
    hypotheses_map = implementation["polarity"]["hypotheses"]
    requests: list[tuple[tuple[str, str], str, str]] = []
    for context in contexts:
        requests.append((context, "SUPPORTS", hypotheses_map["SUPPORTS"]))
        requests.append((context, "REFUTES", hypotheses_map["REFUTES"]))
    premises = [
        implementation["raw_scoring"]["premise_template"].format(
            claim=context[0], evidence=context[1]
        )
        for context, _, _ in requests
    ]
    hypotheses = [hypothesis for _, _, hypothesis in requests]
    logits = _expanded_entailment_logits(
        tokenizer,
        model,
        premises,
        hypotheses,
        int(binding["batch_size"]),
        int(binding["max_sequence_length"]),
    )
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for (context, label, _), logit in zip(requests, logits, strict=True):
        grouped.setdefault(context, []).append((label, logit))
    output: dict[tuple[str, str], dict[str, float]] = {}
    for context in contexts:
        values = grouped[context]
        probabilities = _softmax([value for _, value in values])
        output[context] = {
            label: probability
            for (label, _), probability in zip(values, probabilities, strict=True)
        }
    return output


def _registered_contexts(
    suite: dict[str, Any], sentence_split_regex: str
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    pattern = re.compile(sentence_split_regex)
    contexts: set[tuple[str, str]] = set()
    pair_sentences: dict[str, list[str]] = {}
    for row in suite["pair_rows"]:
        sentences = [
            part.strip()
            for part in pattern.split(str(row["evidence_text"]).strip())
            if part.strip()
        ]
        if not sentences:
            sentences = [str(row["evidence_text"]).strip()]
        pair_sentences[str(row["pair_id"])] = sentences
        for sentence in sentences:
            contexts.add((str(row["claim"]), sentence))
    for row in suite["evidence_set_rows"]:
        for span in row["evidence_spans"]:
            contexts.add((str(row["claim"]), str(span)))
    return sorted(contexts), pair_sentences


def _raw_pair_rows(
    suite: dict[str, Any],
    pair_sentences: dict[str, list[str]],
    scores: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in suite["pair_rows"]:
        pair_id = str(row["pair_id"])
        claim = str(row["claim"])
        output.append(
            {
                "pair_id": pair_id,
                "unit_id": str(row["unit_id"]),
                "split": "calibration",
                "subtype": str(row["subtype"]),
                "gold": row["gold"],
                "spans": [scores[(claim, sentence)] for sentence in pair_sentences[pair_id]],
            }
        )
    return output


def _all_nonempty_subsets(values: list[str]) -> list[tuple[list[int], str]]:
    output: list[tuple[list[int], str]] = []
    for size in range(1, len(values) + 1):
        for indices_tuple in combinations(range(len(values)), size):
            indices = list(indices_tuple)
            text = " ".join(values[index] for index in indices)
            output.append((indices, text))
    return output


def _raw_set_rows(
    suite: dict[str, Any],
    scores: dict[tuple[str, str], dict[str, Any]],
    subset_scores: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in suite["evidence_set_rows"]:
        claim = str(row["claim"])
        evidence_spans = [str(span) for span in row["evidence_spans"]]
        subset_polarity: dict[str, dict[str, float]] = {}
        for indices, text in _all_nonempty_subsets(evidence_spans):
            key = ",".join(str(index) for index in indices)
            subset_polarity[key] = subset_scores[(claim, text)]
        output.append(
            {
                "set_id": str(row["set_id"]),
                "unit_id": str(row["unit_id"]),
                "split": "calibration",
                "subtype": str(row["subtype"]),
                "gold": row["gold"],
                "spans": [scores[(claim, span)] for span in evidence_spans],
                "subset_polarity": subset_polarity,
            }
        )
    return output


def _environment() -> dict[str, Any]:
    names = ("huggingface-hub", "safetensors", "sentencepiece", "torch", "transformers")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in names},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _report(results: dict[str, Any]) -> str:
    selected = results["parameter_selection"]["selected"]
    metrics = selected["metrics"]
    return "\n".join(
        [
            "# A4.5b-M3 SCEC calibration-only result",
            "",
            f"Scientific status: **{results['scientific_status']}**",
            f"Scientific pass: **{results['scientific_pass']}**",
            "",
            f"Calibration units: **{results['calibration_units']}**",
            f"Pair rows: **{results['pair_rows']}**",
            f"Evidence-set rows: **{results['evidence_set_rows']}**",
            f"Claim rows: **{results['claim_rows']}**",
            "",
            "## Frozen parameters",
            "",
            f"- mismatch threshold: `{selected['mismatch_threshold']}`",
            f"- coverage threshold: `{selected['coverage_threshold']}`",
            (
                f"- feasible candidates: "
                f"`{results['parameter_selection']['feasible_candidate_count']}`"
            ),
            "",
            "## Readiness snapshot",
            "",
            f"- compatibility macro F1: `{metrics['compatibility_macro_f1']:.6f}`",
            f"- relevant-but-insufficient compatible recall: "
            f"`{metrics['relevant_but_insufficient_compatible_recall']:.6f}`",
            f"- sufficiency macro F1: `{metrics['sufficiency_macro_f1']:.6f}`",
            f"- polarity macro F1: `{metrics['polarity_macro_f1']:.6f}`",
            f"- final relation macro F1: `{metrics['final_relation_macro_f1']:.6f}`",
            f"- claim-category macro accuracy: `{metrics['claim_category_macro_accuracy']:.6f}`",
            "",
            (
                "A4.5a fresh validation and the 68-query confirmatory partition were not "
                "opened or scored."
            ),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(CONFIG)
    if config["status"] != "REGISTERED_PRE_EXECUTION_CALIBRATION_ONLY":
        raise RuntimeError("A4.5b-M3 is not in registered pre-execution state")
    if int(config["execution_scope"]["validation_scoring_authorized"]) != 0:
        raise RuntimeError("A4.5b-M3 validation scoring is not authorized")
    if int(config["execution_scope"]["confirmatory_scoring_authorized"]) != 0:
        raise RuntimeError("A4.5b-M3 confirmatory scoring is not authorized")

    frozen_manifest = manifest()
    expected_hashes = config["calibration"]["sha256"]
    if frozen_manifest["sha256"] != expected_hashes:
        raise RuntimeError("A4.5b-M3 M2 calibration hashes drifted")
    suite = build_suite()
    if (
        len(suite["units"]) != 48
        or len(suite["pair_rows"]) != 768
        or len(suite["evidence_set_rows"]) != 384
        or len(suite["claim_rows"]) != 384
    ):
        raise RuntimeError("A4.5b-M3 M2 calibration cardinality drifted")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verification, tokenizer, model = _verify_and_load_model(config)
    _write_json(args.output_dir / "model_weight_verification.json", verification)

    implementation = config["authoritative_implementation"]
    contexts, pair_sentences = _registered_contexts(
        suite, implementation["compatibility"]["sentence_segmentation_regex"]
    )
    semantic_scores = _score_registered_requests(tokenizer, model, contexts, config)

    subset_contexts: set[tuple[str, str]] = set()
    for row in suite["evidence_set_rows"]:
        spans = [str(span) for span in row["evidence_spans"]]
        for _, text in _all_nonempty_subsets(spans):
            subset_contexts.add((str(row["claim"]), text))
    subset_scores = _score_polarity_only(tokenizer, model, sorted(subset_contexts), config)

    raw_pairs = _raw_pair_rows(suite, pair_sentences, semantic_scores)
    raw_sets = _raw_set_rows(suite, semantic_scores, subset_scores)
    _write_jsonl(args.output_dir / "calibration_pair_raw_scores.jsonl", raw_pairs)
    _write_jsonl(args.output_dir / "calibration_set_raw_scores.jsonl", raw_sets)

    selection = select_parameters(suite, raw_pairs, raw_sets, config)
    selected = selection["selected"]
    scientific_pass = bool(selected["calibration_ready"])
    scientific_status = (
        config["scientific_outcomes"]["pass"]
        if scientific_pass
        else config["scientific_outcomes"]["fail"]
    )
    results = {
        "checkpoint": "A4.5b-M3",
        "binding_id": config["binding_id"],
        "scientific_pass": scientific_pass,
        "scientific_status": scientific_status,
        "calibration_units": 48,
        "pair_rows": 768,
        "evidence_set_rows": 384,
        "claim_rows": 384,
        "calibration_sha256": expected_hashes,
        "parameter_selection": selection,
        "model_weight_verification": verification,
        "a45a_fresh_validation_rows_materialized": 0,
        "a45a_fresh_validation_rows_scored": 0,
        "confirmatory_records_inspected": 0,
        "confirmatory_queries_scored": 0,
        "a45b_closed_rows_scored": 0,
        "post_result_rescue_authorized": False,
        "fresh_validation_authorized": False,
        "next_checkpoint_authorized": False,
    }
    selected.pop("pair_predictions", None)
    selected.pop("set_predictions", None)
    selected.pop("claim_predictions", None)
    _write_json(args.output_dir / "results.json", results)
    _write_json(args.output_dir / "environment.json", _environment())
    (args.output_dir / "report.md").write_text(_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

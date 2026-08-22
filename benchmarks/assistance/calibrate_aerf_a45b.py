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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from calibration_cases_a45b import build_calibration_only, calibration_manifest

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
RELATION_LABELS = ("ENTAILED", "CONTRADICTED", "UNKNOWN")
RELEVANCE_LABELS = ("RELEVANT", "IRRELEVANT")
SUFFICIENCY_LABELS = ("SUFFICIENT", "INSUFFICIENT")
POLARITY_LABELS = ("SUPPORTS", "REFUTES")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object in {path}")
    return value


def _sha256_file(path: Path) -> str:
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
    observed = _sha256_file(path)
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


def _run_model(
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


def _label_stats(gold: list[str], predicted: list[str], label: str) -> dict[str, float]:
    tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
    fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
    fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _macro_f1(gold: list[str], predicted: list[str], labels: tuple[str, ...]) -> float:
    return sum(_label_stats(gold, predicted, label)["f1"] for label in labels) / len(labels)


def _token_overlap(gold: str, predicted: str | None) -> tuple[float, float]:
    if predicted is None:
        return 0.0, 0.0
    gold_tokens = Counter(token.lower() for token in TOKEN_RE.findall(gold))
    predicted_tokens = Counter(token.lower() for token in TOKEN_RE.findall(predicted))
    overlap = sum((gold_tokens & predicted_tokens).values())
    gold_count = sum(gold_tokens.values())
    predicted_count = sum(predicted_tokens.values())
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / gold_count if gold_count else 0.0
    return precision, recall


def _compose(relations: list[str], gate: str) -> str:
    if gate == "CITATION_INVALID":
        return "CITATION_INVALID"
    if gate == "STALE_EVIDENCE":
        return "STALE_EVIDENCE"
    if gate == "REGISTERED_CONFLICT":
        return "CONFLICTING_EVIDENCE"
    if "ENTAILED" in relations and "CONTRADICTED" in relations:
        return "CONFLICTING_EVIDENCE"
    if relations and all(relation == "ENTAILED" for relation in relations):
        return "SUPPORTED"
    return "UNSUPPORTED"


def _claim_accuracy(claims: list[dict[str, Any]]) -> float:
    correct = 0
    for row in claims:
        predicted = _compose(
            [str(value) for value in row["atom_relations"]],
            str(row["deterministic_gate"]),
        )
        correct += predicted == row["expected_verdict"]
    return correct / len(claims)


def _predict(
    scores: list[dict[str, Any]], relevance_threshold: float, sufficiency_threshold: float
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in scores:
        relevant = float(row["relevance_score"]) >= relevance_threshold
        entailment = float(row["nli_probabilities"]["entailment"])
        contradiction = float(row["nli_probabilities"]["contradiction"])
        strength = max(entailment, contradiction)
        if not relevant:
            output.append(
                {
                    "relevance": "IRRELEVANT",
                    "sufficiency": "NOT_APPLICABLE",
                    "polarity": "NOT_APPLICABLE",
                    "relation": "UNKNOWN",
                    "span": None,
                }
            )
            continue
        if strength < sufficiency_threshold or entailment == contradiction:
            output.append(
                {
                    "relevance": "RELEVANT",
                    "sufficiency": "INSUFFICIENT",
                    "polarity": "UNRESOLVED",
                    "relation": "UNKNOWN",
                    "span": str(row["selected_span"]),
                }
            )
            continue
        if entailment > contradiction:
            polarity = "SUPPORTS"
            relation = "ENTAILED"
        else:
            polarity = "REFUTES"
            relation = "CONTRADICTED"
        output.append(
            {
                "relevance": "RELEVANT",
                "sufficiency": "SUFFICIENT",
                "polarity": polarity,
                "relation": relation,
                "span": str(row["selected_span"]),
            }
        )
    return output


def _false_contradiction_rate(
    pairs: list[dict[str, Any]], predicted_relations: list[str], subtype: str
) -> float:
    indices = [index for index, row in enumerate(pairs) if row["subtype"] == subtype]
    return (
        sum(predicted_relations[index] == "CONTRADICTED" for index in indices) / len(indices)
        if indices
        else 0.0
    )


def _metrics(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    relevance_threshold: float,
    sufficiency_threshold: float,
    claim_accuracy: float,
) -> dict[str, float]:
    predicted = _predict(scores, relevance_threshold, sufficiency_threshold)
    gold_relevance = [str(row["gold"]["relevance"]) for row in pairs]
    pred_relevance = [str(row["relevance"]) for row in predicted]
    relevant_stats = _label_stats(gold_relevance, pred_relevance, "RELEVANT")
    irrelevant_stats = _label_stats(gold_relevance, pred_relevance, "IRRELEVANT")

    relevant_indices = [
        index for index, row in enumerate(pairs) if row["gold"]["relevance"] == "RELEVANT"
    ]
    gold_sufficiency = [str(pairs[index]["gold"]["sufficiency"]) for index in relevant_indices]
    pred_sufficiency = [
        "SUFFICIENT" if predicted[index]["sufficiency"] == "SUFFICIENT" else "INSUFFICIENT"
        for index in relevant_indices
    ]
    sufficient_stats = _label_stats(gold_sufficiency, pred_sufficiency, "SUFFICIENT")
    insufficient_stats = _label_stats(gold_sufficiency, pred_sufficiency, "INSUFFICIENT")

    polarity_indices = [
        index
        for index, row in enumerate(pairs)
        if row["gold"]["relevance"] == "RELEVANT"
        and row["gold"]["sufficiency"] == "SUFFICIENT"
    ]
    gold_polarity = [str(pairs[index]["gold"]["polarity"]) for index in polarity_indices]
    pred_polarity = [str(predicted[index]["polarity"]) for index in polarity_indices]
    supports_stats = _label_stats(gold_polarity, pred_polarity, "SUPPORTS")
    refutes_stats = _label_stats(gold_polarity, pred_polarity, "REFUTES")

    gold_relations = [str(row["gold"]["final_relation"]) for row in pairs]
    pred_relations = [str(row["relation"]) for row in predicted]
    entailed_stats = _label_stats(gold_relations, pred_relations, "ENTAILED")
    contradicted_stats = _label_stats(gold_relations, pred_relations, "CONTRADICTED")
    unknown_stats = _label_stats(gold_relations, pred_relations, "UNKNOWN")

    span_precision: list[float] = []
    span_recall: list[float] = []
    for index in relevant_indices:
        gold_span = str(pairs[index]["gold"]["minimal_evidence_text"])
        precision, recall = _token_overlap(gold_span, predicted[index]["span"])
        span_precision.append(precision)
        span_recall.append(recall)

    context_indices = [
        index for index, row in enumerate(pairs) if row["subtype"] == "context_contamination_support"
    ]
    context_accuracy = sum(
        pred_relations[index] == "ENTAILED" for index in context_indices
    ) / len(context_indices)

    return {
        "minimal_evidence_span_precision": float(np.mean(span_precision)),
        "minimal_evidence_span_recall": float(np.mean(span_recall)),
        "relevance_macro_f1": _macro_f1(gold_relevance, pred_relevance, RELEVANCE_LABELS),
        "relevant_recall": relevant_stats["recall"],
        "irrelevant_recall": irrelevant_stats["recall"],
        "sufficiency_macro_f1_on_relevant_pairs": _macro_f1(
            gold_sufficiency,
            pred_sufficiency,
            SUFFICIENCY_LABELS,
        ),
        "sufficient_recall": sufficient_stats["recall"],
        "insufficient_recall": insufficient_stats["recall"],
        "polarity_macro_f1_on_relevant_sufficient_pairs": _macro_f1(
            gold_polarity,
            pred_polarity,
            POLARITY_LABELS,
        ),
        "supports_recall": supports_stats["recall"],
        "refutes_recall": refutes_stats["recall"],
        "final_relation_macro_f1": _macro_f1(gold_relations, pred_relations, RELATION_LABELS),
        "entailed_recall": entailed_stats["recall"],
        "contradicted_recall": contradicted_stats["recall"],
        "unknown_recall": unknown_stats["recall"],
        "cross_document_irrelevance_false_contradiction_rate": _false_contradiction_rate(
            pairs,
            pred_relations,
            "cross_document_irrelevance",
        ),
        "same_domain_irrelevance_false_contradiction_rate": _false_contradiction_rate(
            pairs,
            pred_relations,
            "same_domain_irrelevance",
        ),
        "relevant_insufficient_false_contradiction_rate": _false_contradiction_rate(
            pairs,
            pred_relations,
            "relevant_but_insufficient",
        ),
        "context_contamination_support_accuracy": context_accuracy,
        "claim_composition_accuracy": claim_accuracy,
    }


def _checks(
    metrics: dict[str, float], requirements: dict[str, Any]
) -> tuple[dict[str, bool], bool]:
    output: dict[str, bool] = {}
    for requirement, raw_threshold in requirements.items():
        threshold = float(raw_threshold)
        if requirement.endswith("_min"):
            metric = requirement.removesuffix("_min")
            output[requirement] = metrics[metric] >= threshold - 1e-12
        elif requirement.endswith("_max"):
            metric = requirement.removesuffix("_max")
            output[requirement] = metrics[metric] <= threshold + 1e-12
        else:
            raise RuntimeError(f"Unsupported requirement {requirement}")
    return output, all(output.values())


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    minimum_recall = min(
        metrics["entailed_recall"],
        metrics["contradicted_recall"],
        metrics["unknown_recall"],
    )
    false_contradiction = (
        metrics["cross_document_irrelevance_false_contradiction_rate"]
        + metrics["same_domain_irrelevance_false_contradiction_rate"]
        + metrics["relevant_insufficient_false_contradiction_rate"]
    )
    return (
        metrics["final_relation_macro_f1"],
        minimum_recall,
        metrics["relevance_macro_f1"],
        metrics["sufficiency_macro_f1_on_relevant_pairs"],
        metrics["polarity_macro_f1_on_relevant_sufficient_pairs"],
        -false_contradiction,
        candidate["sufficiency_threshold"],
        candidate["relevance_threshold"],
    )


def _grid_values(grid: dict[str, Any]) -> list[float]:
    start = int(grid["integer_start"])
    stop = int(grid["integer_stop"])
    step = int(grid["integer_step"])
    scale = float(grid["scale"])
    return [round(value * scale, 10) for value in range(start, stop + step, step)]


def _select_thresholds(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    config: dict[str, Any],
    claim_accuracy: float,
) -> dict[str, Any]:
    setup = config["threshold_calibration"]
    relevance_values = _grid_values(setup["relevance_grid"])
    sufficiency_values = _grid_values(setup["sufficiency_grid"])
    expected = int(setup["joint_candidates"])
    if len(relevance_values) * len(sufficiency_values) != expected:
        raise RuntimeError("A4.5b threshold-grid cardinality drifted")

    requirements = config["calibration_readiness_requirements"]
    best_any: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    feasible_count = 0
    for relevance_threshold in relevance_values:
        for sufficiency_threshold in sufficiency_values:
            metrics = _metrics(
                pairs,
                scores,
                relevance_threshold,
                sufficiency_threshold,
                claim_accuracy,
            )
            checks, passed = _checks(metrics, requirements)
            candidate = {
                "relevance_threshold": relevance_threshold,
                "sufficiency_threshold": sufficiency_threshold,
                "metrics": metrics,
                "requirement_checks": checks,
                "calibration_ready": passed,
            }
            if best_any is None or _selection_key(candidate) > _selection_key(best_any):
                best_any = candidate
            if passed:
                feasible_count += 1
                if best_feasible is None or _selection_key(candidate) > _selection_key(best_feasible):
                    best_feasible = candidate
    if best_any is None:
        raise RuntimeError("A4.5b threshold grid was empty")
    selected = best_feasible if best_feasible is not None else best_any
    return {
        "selected": selected,
        "feasible_candidate_count": feasible_count,
        "joint_candidates_evaluated": expected,
        "selection_used_feasible_set": best_feasible is not None,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _environment() -> dict[str, Any]:
    package_names = (
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
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _report(results: dict[str, Any]) -> str:
    selected = results["threshold_selection"]["selected"]
    metrics = selected["metrics"]
    return "\n".join(
        [
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
    )


def _load_models(config: dict[str, Any]) -> tuple[dict[str, Any], Any, Any, Any, Any]:
    binding = config["binding"]
    weight_verification = {
        "alignment_relevance": _verify_weight(binding["alignment_relevance"]),
        "sufficiency_polarity": _verify_weight(binding["sufficiency_polarity"]),
    }
    relevance = binding["alignment_relevance"]
    relevance_tokenizer = AutoTokenizer.from_pretrained(
        relevance["model_id"],
        revision=relevance["revision"],
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
    observed_labels = {
        str(index): str(label).lower() for index, label in nli_model.config.id2label.items()
    }
    expected_labels = {
        str(index): str(label).lower() for index, label in nli["native_labels"].items()
    }
    if observed_labels != expected_labels:
        raise RuntimeError(f"A4.5b NLI label mapping drifted: {observed_labels}")
    nli_model.eval()
    return (
        weight_verification,
        relevance_tokenizer,
        relevance_model,
        nli_tokenizer,
        nli_model,
    )


def _score_pairs(
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
    relevance_tokenizer: Any,
    relevance_model: Any,
    nli_tokenizer: Any,
    nli_model: Any,
) -> list[dict[str, Any]]:
    relevance = config["binding"]["alignment_relevance"]
    all_claims: list[str] = []
    all_spans: list[str] = []
    ranges: list[tuple[int, int, list[str]]] = []
    for row in pairs:
        spans = _sentences(str(row["evidence_text"]))
        start = len(all_spans)
        all_claims.extend([str(row["claim"])] * len(spans))
        all_spans.extend(spans)
        ranges.append((start, len(all_spans), spans))
    relevance_logits = _run_model(
        relevance_tokenizer,
        relevance_model,
        all_claims,
        all_spans,
        int(relevance["batch_size"]),
        int(relevance["max_sequence_length"]),
    )
    if relevance_logits.ndim != 2 or relevance_logits.shape[1] != 1:
        raise RuntimeError("A4.5b relevance model must emit one logit")

    selected_spans: list[str] = []
    selected_scores: list[float] = []
    span_scores: list[list[dict[str, Any]]] = []
    for start, stop, spans in ranges:
        local = relevance_logits[start:stop, 0]
        winner = int(np.argmax(local))
        selected_spans.append(spans[winner])
        selected_scores.append(float(local[winner]))
        span_scores.append(
            [
                {"span": span, "relevance_score": float(score)}
                for span, score in zip(spans, local.tolist(), strict=True)
            ]
        )

    nli = config["binding"]["sufficiency_polarity"]
    nli_logits = _run_model(
        nli_tokenizer,
        nli_model,
        selected_spans,
        [str(row["claim"]) for row in pairs],
        int(nli["batch_size"]),
        int(nli["max_sequence_length"]),
    )
    if nli_logits.ndim != 2 or nli_logits.shape[1] != 3:
        raise RuntimeError("A4.5b NLI model must emit three logits")
    nli_probabilities = torch.softmax(torch.from_numpy(nli_logits), dim=-1).numpy()

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "unit_id": str(pair["unit_id"]),
                "split": "calibration",
                "subtype": str(pair["subtype"]),
                "gold": pair["gold"],
                "selected_span": selected_spans[index],
                "relevance_score": selected_scores[index],
                "span_scores": span_scores[index],
                "nli_probabilities": {
                    "entailment": float(nli_probabilities[index, 0]),
                    "neutral": float(nli_probabilities[index, 1]),
                    "contradiction": float(nli_probabilities[index, 2]),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(CONFIG_PATH)
    if config["status"] != "REGISTERED_PRE_EXECUTION_CALIBRATION_ONLY":
        raise RuntimeError("A4.5b config is not frozen for calibration-only execution")
    sealed = config["sealed_partitions"]
    if sealed["validation_scoring_authorized"] != 0:
        raise RuntimeError("Fresh validation is not authorized in A4.5b")
    if sealed["confirmatory_scoring_authorized"] != 0:
        raise RuntimeError("Confirmatory scoring is not authorized in A4.5b")

    manifest = calibration_manifest()
    materialized = build_calibration_only()
    pairs = materialized["pair_rows"]
    claims = materialized["claim_rows"]
    if manifest["calibration_pairs"] != 360 or manifest["calibration_claims"] != 360:
        raise RuntimeError("A4.5b calibration cardinality drifted")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    (
        weight_verification,
        relevance_tokenizer,
        relevance_model,
        nli_tokenizer,
        nli_model,
    ) = _load_models(config)
    _write_json(args.output_dir / "model_weight_verification.json", weight_verification)

    scores = _score_pairs(
        pairs,
        config,
        relevance_tokenizer,
        relevance_model,
        nli_tokenizer,
        nli_model,
    )
    _write_jsonl(args.output_dir / "calibration_pair_scores.jsonl", scores)

    claim_accuracy = _claim_accuracy(claims)
    threshold_selection = _select_thresholds(pairs, scores, config, claim_accuracy)
    selected = threshold_selection["selected"]
    scientific_pass = bool(selected["calibration_ready"])
    status = (
        "PASSED_CALIBRATION_READINESS_THRESHOLDS_FROZEN"
        if scientific_pass
        else "FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED"
    )
    results = {
        "binding_id": config["binding_id"],
        "checkpoint": "A4.5b",
        "scientific_pass": scientific_pass,
        "scientific_status": status,
        "calibration_units": manifest["calibration_units"],
        "calibration_pairs": manifest["calibration_pairs"],
        "calibration_claims": manifest["calibration_claims"],
        "calibration_pairs_sha256": manifest["calibration_pairs_sha256"],
        "calibration_claims_sha256": manifest["calibration_claims_sha256"],
        "threshold_selection": threshold_selection,
        "model_weight_verification": weight_verification,
        "validation_rows_materialized": 0,
        "validation_rows_scored": 0,
        "confirmatory_queries_inspected": 0,
        "confirmatory_queries_scored": 0,
        "a44d_rows_rescored": 0,
        "a44a_rows_rescored": 0,
        "post_result_rescue_authorized": false,
        "next_checkpoint_authorized": false
    }
    _write_json(args.output_dir / "results.json", results)
    _write_json(args.output_dir / "environment.json", _environment())
    (args.output_dir / "report.md").write_text(_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Pure calibration arithmetic for the registered A4.5b AERF stack."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

RELATION_LABELS = ("ENTAILED", "CONTRADICTED", "UNKNOWN")
RELEVANCE_LABELS = ("RELEVANT", "IRRELEVANT")
SUFFICIENCY_LABELS = ("SUFFICIENT", "INSUFFICIENT")
POLARITY_LABELS = ("SUPPORTS", "REFUTES")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def label_stats(gold: list[str], predicted: list[str], label: str) -> dict[str, float]:
    pairs = list(zip(gold, predicted, strict=True))
    tp = sum(g == label and p == label for g, p in pairs)
    fp = sum(g != label and p == label for g, p in pairs)
    fn = sum(g == label and p != label for g, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(gold: list[str], predicted: list[str], labels: tuple[str, ...]) -> float:
    return sum(label_stats(gold, predicted, label)["f1"] for label in labels) / len(labels)


def token_overlap(gold: str, predicted: str) -> tuple[float, float]:
    gold_tokens = Counter(token.lower() for token in TOKEN_RE.findall(gold))
    predicted_tokens = Counter(token.lower() for token in TOKEN_RE.findall(predicted))
    overlap = sum((gold_tokens & predicted_tokens).values())
    gold_count = sum(gold_tokens.values())
    predicted_count = sum(predicted_tokens.values())
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / gold_count if gold_count else 0.0
    return precision, recall


def relation_prediction(
    score: dict[str, Any], relevance_threshold: float, sufficiency_threshold: float
) -> str:
    if float(score["relevance_score"]) < relevance_threshold:
        return "UNKNOWN"
    entailment = float(score["nli_probabilities"]["entailment"])
    contradiction = float(score["nli_probabilities"]["contradiction"])
    if max(entailment, contradiction) < sufficiency_threshold:
        return "UNKNOWN"
    if entailment == contradiction:
        return "UNKNOWN"
    return "ENTAILED" if entailment > contradiction else "CONTRADICTED"


def component_predictions(
    score: dict[str, Any], relevance_threshold: float, sufficiency_threshold: float
) -> tuple[str, str, str]:
    relevance = (
        "RELEVANT" if float(score["relevance_score"]) >= relevance_threshold else "IRRELEVANT"
    )
    entailment = float(score["nli_probabilities"]["entailment"])
    contradiction = float(score["nli_probabilities"]["contradiction"])
    strength = max(entailment, contradiction)
    sufficiency = "SUFFICIENT" if strength >= sufficiency_threshold else "INSUFFICIENT"
    if entailment == contradiction:
        polarity = "UNRESOLVED"
    else:
        polarity = "SUPPORTS" if entailment > contradiction else "REFUTES"
    return relevance, sufficiency, polarity


def false_contradiction_rate(
    pairs: list[dict[str, Any]], predicted: list[str], subtype: str
) -> float:
    indices = [index for index, row in enumerate(pairs) if row["subtype"] == subtype]
    if not indices:
        return 0.0
    errors = sum(predicted[index] == "CONTRADICTED" for index in indices)
    return errors / len(indices)


def calibration_metrics(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    relevance_threshold: float,
    sufficiency_threshold: float,
    claim_composition_accuracy: float,
) -> dict[str, float]:
    if len(pairs) != len(scores):
        raise RuntimeError("A4.5b pair/score cardinality mismatch")
    relevance_predictions: list[str] = []
    sufficiency_predictions: list[str] = []
    polarity_predictions: list[str] = []
    relation_predictions: list[str] = []
    for score in scores:
        relevance, sufficiency, polarity = component_predictions(
            score,
            relevance_threshold,
            sufficiency_threshold,
        )
        relevance_predictions.append(relevance)
        sufficiency_predictions.append(sufficiency)
        polarity_predictions.append(polarity)
        relation_predictions.append(
            relation_prediction(score, relevance_threshold, sufficiency_threshold)
        )

    gold_relevance = [str(row["gold"]["relevance"]) for row in pairs]
    relevant_stats = label_stats(gold_relevance, relevance_predictions, "RELEVANT")
    irrelevant_stats = label_stats(gold_relevance, relevance_predictions, "IRRELEVANT")

    relevant_indices = [
        index for index, row in enumerate(pairs) if row["gold"]["relevance"] == "RELEVANT"
    ]
    gold_sufficiency = [str(pairs[index]["gold"]["sufficiency"]) for index in relevant_indices]
    predicted_sufficiency = [sufficiency_predictions[index] for index in relevant_indices]
    sufficient_stats = label_stats(gold_sufficiency, predicted_sufficiency, "SUFFICIENT")
    insufficient_stats = label_stats(gold_sufficiency, predicted_sufficiency, "INSUFFICIENT")

    sufficient_indices = [
        index
        for index in relevant_indices
        if pairs[index]["gold"]["sufficiency"] == "SUFFICIENT"
    ]
    gold_polarity = [str(pairs[index]["gold"]["polarity"]) for index in sufficient_indices]
    predicted_polarity = [polarity_predictions[index] for index in sufficient_indices]
    supports_stats = label_stats(gold_polarity, predicted_polarity, "SUPPORTS")
    refutes_stats = label_stats(gold_polarity, predicted_polarity, "REFUTES")

    gold_relations = [str(row["gold"]["final_relation"]) for row in pairs]
    entailed_stats = label_stats(gold_relations, relation_predictions, "ENTAILED")
    contradicted_stats = label_stats(gold_relations, relation_predictions, "CONTRADICTED")
    unknown_stats = label_stats(gold_relations, relation_predictions, "UNKNOWN")

    span_precisions: list[float] = []
    span_recalls: list[float] = []
    for index in relevant_indices:
        precision, recall = token_overlap(
            str(pairs[index]["gold"]["minimal_evidence_text"]),
            str(scores[index]["selected_span"]),
        )
        span_precisions.append(precision)
        span_recalls.append(recall)

    context_indices = [
        index
        for index, row in enumerate(pairs)
        if row["subtype"] == "context_contamination_support"
    ]
    context_accuracy = (
        sum(relation_predictions[index] == "ENTAILED" for index in context_indices)
        / len(context_indices)
    )
    return {
        "minimal_evidence_span_precision": sum(span_precisions) / len(span_precisions),
        "minimal_evidence_span_recall": sum(span_recalls) / len(span_recalls),
        "relevance_macro_f1": macro_f1(
            gold_relevance,
            relevance_predictions,
            RELEVANCE_LABELS,
        ),
        "relevant_recall": relevant_stats["recall"],
        "irrelevant_recall": irrelevant_stats["recall"],
        "sufficiency_macro_f1_on_relevant_pairs": macro_f1(
            gold_sufficiency,
            predicted_sufficiency,
            SUFFICIENCY_LABELS,
        ),
        "sufficient_recall": sufficient_stats["recall"],
        "insufficient_recall": insufficient_stats["recall"],
        "polarity_macro_f1_on_relevant_sufficient_pairs": macro_f1(
            gold_polarity,
            predicted_polarity,
            POLARITY_LABELS,
        ),
        "supports_recall": supports_stats["recall"],
        "refutes_recall": refutes_stats["recall"],
        "final_relation_macro_f1": macro_f1(
            gold_relations,
            relation_predictions,
            RELATION_LABELS,
        ),
        "entailed_recall": entailed_stats["recall"],
        "contradicted_recall": contradicted_stats["recall"],
        "unknown_recall": unknown_stats["recall"],
        "cross_document_irrelevance_false_contradiction_rate": false_contradiction_rate(
            pairs,
            relation_predictions,
            "cross_document_irrelevance",
        ),
        "same_domain_irrelevance_false_contradiction_rate": false_contradiction_rate(
            pairs,
            relation_predictions,
            "same_domain_irrelevance",
        ),
        "relevant_insufficient_false_contradiction_rate": false_contradiction_rate(
            pairs,
            relation_predictions,
            "relevant_but_insufficient",
        ),
        "context_contamination_support_accuracy": context_accuracy,
        "claim_composition_accuracy": claim_composition_accuracy,
    }


def requirement_checks(
    metrics: dict[str, float], requirements: dict[str, Any]
) -> tuple[dict[str, bool], bool]:
    checks: dict[str, bool] = {}
    for name, raw_threshold in requirements.items():
        threshold = float(raw_threshold)
        if name.endswith("_min"):
            metric_name = name.removesuffix("_min")
            checks[name] = metrics[metric_name] >= threshold - 1e-12
        elif name.endswith("_max"):
            metric_name = name.removesuffix("_max")
            checks[name] = metrics[metric_name] <= threshold + 1e-12
        else:
            raise RuntimeError(f"Unsupported A4.5b requirement: {name}")
    return checks, all(checks.values())


def selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    minimum_relation_recall = min(
        metrics["entailed_recall"],
        metrics["contradicted_recall"],
        metrics["unknown_recall"],
    )
    false_contradiction_sum = (
        metrics["cross_document_irrelevance_false_contradiction_rate"]
        + metrics["same_domain_irrelevance_false_contradiction_rate"]
        + metrics["relevant_insufficient_false_contradiction_rate"]
    )
    return (
        metrics["final_relation_macro_f1"],
        minimum_relation_recall,
        metrics["relevance_macro_f1"],
        metrics["sufficiency_macro_f1_on_relevant_pairs"],
        metrics["polarity_macro_f1_on_relevant_sufficient_pairs"],
        -false_contradiction_sum,
        float(candidate["sufficiency_threshold"]),
        float(candidate["relevance_threshold"]),
    )


def grid_values(grid: dict[str, Any]) -> list[float]:
    start = int(grid["integer_start"])
    stop = int(grid["integer_stop"])
    step = int(grid["integer_step"])
    scale = float(grid["scale"])
    return [round(value * scale, 10) for value in range(start, stop + step, step)]


def select_thresholds(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    config: dict[str, Any],
    claim_composition_accuracy: float,
) -> dict[str, Any]:
    setup = config["threshold_calibration"]
    relevance_values = grid_values(setup["relevance_grid"])
    sufficiency_values = grid_values(setup["sufficiency_grid"])
    expected_candidates = int(setup["joint_candidates"])
    if len(relevance_values) * len(sufficiency_values) != expected_candidates:
        raise RuntimeError("A4.5b threshold-grid cardinality drifted")
    requirements = config["calibration_readiness_requirements"]
    best_any: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    feasible_count = 0
    for relevance_threshold in relevance_values:
        for sufficiency_threshold in sufficiency_values:
            metrics = calibration_metrics(
                pairs,
                scores,
                relevance_threshold,
                sufficiency_threshold,
                claim_composition_accuracy,
            )
            checks, passed = requirement_checks(metrics, requirements)
            candidate = {
                "relevance_threshold": relevance_threshold,
                "sufficiency_threshold": sufficiency_threshold,
                "metrics": metrics,
                "requirement_checks": checks,
                "calibration_ready": passed,
            }
            if best_any is None or selection_key(candidate) > selection_key(best_any):
                best_any = candidate
            if passed:
                feasible_count += 1
                if best_feasible is None or selection_key(candidate) > selection_key(best_feasible):
                    best_feasible = candidate
    if best_any is None:
        raise RuntimeError("A4.5b threshold grid is empty")
    selected = best_feasible if best_feasible is not None else best_any
    return {
        "selected": selected,
        "feasible_candidate_count": feasible_count,
        "joint_candidates_evaluated": expected_candidates,
        "selection_used_feasible_set": best_feasible is not None,
    }

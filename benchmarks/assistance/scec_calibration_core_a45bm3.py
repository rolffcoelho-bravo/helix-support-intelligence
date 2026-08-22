"""Deterministic scoring and parameter selection for A4.5b-M3 SCEC calibration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

DIMENSIONS = (
    "entity",
    "predicate",
    "target_slot",
    "temporal_scope",
    "location_scope",
    "organizational_scope",
    "conditional_scope",
    "modality_quantification_scope",
)
SLOTS = (
    "entity",
    "predicate",
    "target_slot_identity",
    "target_value",
    "temporal_scope",
    "location_scope",
    "organizational_scope",
    "conditional_scope",
    "modality_quantification_scope",
)
RELATION_LABELS = ("ENTAILED", "CONTRADICTED", "UNKNOWN", "CONFLICTING_EVIDENCE")
CLAIM_SET_SUFFIX = {
    "single_supported": "S01",
    "single_refuted": "S02",
    "compatible_insufficient": "S03",
    "complementary_multi_span_supported": "S05",
    "support_refute_conflict": "S07",
    "citation_invalid": "S01",
    "stale_evidence": "S01",
    "registered_conflict": "S01",
}


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _recall(gold: list[str], pred: list[str], label: str) -> float:
    indices = [index for index, value in enumerate(gold) if value == label]
    return _safe_div(sum(pred[index] == label for index in indices), len(indices))


def _precision(gold: list[str], pred: list[str], label: str) -> float:
    indices = [index for index, value in enumerate(pred) if value == label]
    return _safe_div(sum(gold[index] == label for index in indices), len(indices))


def _f1(gold: list[str], pred: list[str], label: str) -> float:
    precision = _precision(gold, pred, label)
    recall = _recall(gold, pred, label)
    return _safe_div(2.0 * precision * recall, precision + recall)


def _macro_f1(gold: list[str], pred: list[str], labels: Iterable[str]) -> float:
    values = [_f1(gold, pred, label) for label in labels]
    return _safe_div(sum(values), len(values))


def _accuracy(rows: list[dict[str, Any]], field: str, expected: str) -> float:
    if not rows:
        return 0.0
    return sum(str(row[field]) == expected for row in rows) / len(rows)


def _dimension_label(scores: dict[str, float], mismatch_threshold: float) -> str:
    match = float(scores["MATCH"])
    mismatch = float(scores["MISMATCH"])
    unspecified = float(scores["UNSPECIFIED"])
    if mismatch >= match and mismatch >= unspecified and mismatch >= mismatch_threshold:
        return "MISMATCH"
    return "UNSPECIFIED" if unspecified > match else "MATCH"


def _compatible(span: dict[str, Any], mismatch_threshold: float) -> bool:
    dimensions = span["dimensions"]
    return all(
        _dimension_label(dimensions[dimension], mismatch_threshold) != "MISMATCH"
        for dimension in DIMENSIONS
    )


def _span_rank(span: dict[str, Any]) -> float:
    return 1.0 - max(float(span["dimensions"][dimension]["MISMATCH"]) for dimension in DIMENSIONS)


def _select_span(spans: list[dict[str, Any]]) -> dict[str, Any]:
    if not spans:
        raise RuntimeError("SCEC raw row contains no sentence span")
    best_index = max(range(len(spans)), key=lambda index: (_span_rank(spans[index]), -index))
    return spans[best_index]


def _coverage(span: dict[str, Any], threshold: float) -> set[str]:
    return {slot for slot in SLOTS if float(span["coverage"][slot]["COVERED"]) >= threshold}


def _polarity(span: dict[str, Any]) -> str:
    scores = span["polarity"]
    return "SUPPORTS" if float(scores["SUPPORTS"]) >= float(scores["REFUTES"]) else "REFUTES"


def _pair_prediction(
    raw: dict[str, Any], mismatch_threshold: float, coverage_threshold: float
) -> dict[str, Any]:
    selected = _select_span(raw["spans"])
    compatible = _compatible(selected, mismatch_threshold)
    if not compatible:
        return {
            "pair_id": raw["pair_id"],
            "subtype": raw["subtype"],
            "selected_span": None,
            "compatibility": "INCOMPATIBLE",
            "sufficiency": "NOT_APPLICABLE",
            "polarity": "UNRESOLVED",
            "final_relation": "UNKNOWN",
        }
    covered = _coverage(selected, coverage_threshold)
    sufficient = all(slot in covered for slot in SLOTS)
    if not sufficient:
        return {
            "pair_id": raw["pair_id"],
            "subtype": raw["subtype"],
            "selected_span": selected["text"],
            "compatibility": "COMPATIBLE",
            "sufficiency": "INSUFFICIENT",
            "polarity": "UNRESOLVED",
            "final_relation": "UNKNOWN",
        }
    polarity = _polarity(selected)
    return {
        "pair_id": raw["pair_id"],
        "subtype": raw["subtype"],
        "selected_span": selected["text"],
        "compatibility": "COMPATIBLE",
        "sufficiency": "SUFFICIENT",
        "polarity": polarity,
        "final_relation": "ENTAILED" if polarity == "SUPPORTS" else "CONTRADICTED",
    }


def _subset_key(indices: list[int]) -> str:
    return ",".join(str(index) for index in indices)


def _set_prediction(
    raw: dict[str, Any], mismatch_threshold: float, coverage_threshold: float
) -> dict[str, Any]:
    spans = raw["spans"]
    compatible_indices = [
        index for index, span in enumerate(spans) if _compatible(span, mismatch_threshold)
    ]
    if not compatible_indices:
        return {
            "set_id": raw["set_id"],
            "subtype": raw["subtype"],
            "compatibility": "INCOMPATIBLE",
            "sufficiency": "INSUFFICIENT",
            "polarity": "UNRESOLVED",
            "final_relation": "UNKNOWN",
            "compatible_indices": [],
        }

    union: set[str] = set()
    individual_support = False
    individual_refute = False
    for index in compatible_indices:
        covered = _coverage(spans[index], coverage_threshold)
        union.update(covered)
        if all(slot in covered for slot in SLOTS):
            polarity = _polarity(spans[index])
            individual_support |= polarity == "SUPPORTS"
            individual_refute |= polarity == "REFUTES"

    if individual_support and individual_refute:
        return {
            "set_id": raw["set_id"],
            "subtype": raw["subtype"],
            "compatibility": "COMPATIBLE",
            "sufficiency": "CONFLICTING",
            "polarity": "CONFLICTING",
            "final_relation": "CONFLICTING_EVIDENCE",
            "compatible_indices": compatible_indices,
        }

    if not all(slot in union for slot in SLOTS):
        return {
            "set_id": raw["set_id"],
            "subtype": raw["subtype"],
            "compatibility": "COMPATIBLE",
            "sufficiency": "INSUFFICIENT",
            "polarity": "UNRESOLVED",
            "final_relation": "UNKNOWN",
            "compatible_indices": compatible_indices,
        }

    key = _subset_key(compatible_indices)
    scores = raw["subset_polarity"].get(key)
    if scores is None:
        raise RuntimeError(
            f"Missing registered subset polarity scores for {raw['set_id']} subset {key}"
        )
    polarity = "SUPPORTS" if float(scores["SUPPORTS"]) >= float(scores["REFUTES"]) else "REFUTES"
    return {
        "set_id": raw["set_id"],
        "subtype": raw["subtype"],
        "compatibility": "COMPATIBLE",
        "sufficiency": "SUFFICIENT",
        "polarity": polarity,
        "final_relation": "ENTAILED" if polarity == "SUPPORTS" else "CONTRADICTED",
        "compatible_indices": compatible_indices,
    }


def _claim_predictions(
    claims: list[dict[str, Any]], set_predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for claim in claims:
        category = str(claim["category"])
        set_id = f"{claim['unit_id']}-{CLAIM_SET_SUFFIX[category]}"
        relation = str(set_predictions[set_id]["final_relation"])
        gate = str(claim["deterministic_gate"])
        if gate == "CITATION_INVALID":
            verdict = "CITATION_INVALID"
        elif gate == "STALE_EVIDENCE":
            verdict = "STALE_EVIDENCE"
        elif gate == "REGISTERED_CONFLICT" or relation == "CONFLICTING_EVIDENCE":
            verdict = "CONFLICTING_EVIDENCE"
        elif relation == "ENTAILED":
            verdict = "SUPPORTED"
        else:
            verdict = "UNSUPPORTED"
        output.append(
            {
                "case_id": str(claim["case_id"]),
                "category": category,
                "source_set_id": set_id,
                "predicted_verdict": verdict,
                "expected_verdict": str(claim["expected_verdict"]),
            }
        )
    return output


def _subtype_rate(
    predictions: list[dict[str, Any]], subtype: str, field: str, expected: str
) -> float:
    rows = [row for row in predictions if row["subtype"] == subtype]
    return _accuracy(rows, field, expected)


def _subtypes_rate(
    predictions: list[dict[str, Any]], subtypes: set[str], field: str, expected: str
) -> float:
    rows = [row for row in predictions if row["subtype"] in subtypes]
    return _accuracy(rows, field, expected)


def evaluate_candidate(
    suite: dict[str, Any],
    raw_pair_rows: list[dict[str, Any]],
    raw_set_rows: list[dict[str, Any]],
    mismatch_threshold: float,
    coverage_threshold: float,
) -> dict[str, Any]:
    pair_gold = {str(row["pair_id"]): row for row in suite["pair_rows"]}
    set_gold = {str(row["set_id"]): row for row in suite["evidence_set_rows"]}
    raw_pair_by_id = {str(row["pair_id"]): row for row in raw_pair_rows}
    raw_set_by_id = {str(row["set_id"]): row for row in raw_set_rows}
    if set(pair_gold) != set(raw_pair_by_id):
        raise RuntimeError("A4.5b-M3 raw pair IDs do not match frozen M2 calibration IDs")
    if set(set_gold) != set(raw_set_by_id):
        raise RuntimeError("A4.5b-M3 raw set IDs do not match frozen M2 calibration IDs")

    pair_pred = [
        _pair_prediction(raw_pair_by_id[pair_id], mismatch_threshold, coverage_threshold)
        for pair_id in sorted(pair_gold)
    ]
    set_pred = [
        _set_prediction(raw_set_by_id[set_id], mismatch_threshold, coverage_threshold)
        for set_id in sorted(set_gold)
    ]
    set_pred_by_id = {str(row["set_id"]): row for row in set_pred}
    claim_pred = _claim_predictions(suite["claim_rows"], set_pred_by_id)

    gold_compat = [str(pair_gold[row["pair_id"]]["gold"]["compatibility"]) for row in pair_pred]
    pred_compat = [str(row["compatibility"]) for row in pair_pred]
    gold_minimal = [
        pair_gold[row["pair_id"]]["gold"]["minimal_compatible_span"] for row in pair_pred
    ]
    predicted_minimal = [row["selected_span"] for row in pair_pred]
    predicted_nonnull = [
        index for index, value in enumerate(predicted_minimal) if value is not None
    ]
    gold_nonnull = [index for index, value in enumerate(gold_minimal) if value is not None]
    minimal_correct = {
        index
        for index in range(len(pair_pred))
        if predicted_minimal[index] is not None and predicted_minimal[index] == gold_minimal[index]
    }

    set_gold_suff = [str(set_gold[row["set_id"]]["gold"]["sufficiency"]) for row in set_pred]
    set_pred_suff = [str(row["sufficiency"]) for row in set_pred]
    suff_labels = ("SUFFICIENT", "INSUFFICIENT", "CONFLICTING")
    polarity_gold_rows = [
        row
        for row in set_pred
        if str(set_gold[row["set_id"]]["gold"]["sufficiency"]) == "SUFFICIENT"
    ]
    gold_polarity = [str(set_gold[row["set_id"]]["gold"]["polarity"]) for row in polarity_gold_rows]
    pred_polarity = [str(row["polarity"]) for row in polarity_gold_rows]
    final_gold = [str(pair_gold[row["pair_id"]]["gold"]["final_relation"]) for row in pair_pred] + [
        str(set_gold[row["set_id"]]["gold"]["final_relation"]) for row in set_pred
    ]
    final_pred = [str(row["final_relation"]) for row in pair_pred] + [
        str(row["final_relation"]) for row in set_pred
    ]

    claim_gold = [str(row["expected_verdict"]) for row in claim_pred]
    claim_values = [str(row["predicted_verdict"]) for row in claim_pred]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in claim_pred:
        categories[str(row["category"])].append(row)
    category_accuracy = {
        category: sum(row["predicted_verdict"] == row["expected_verdict"] for row in rows)
        / len(rows)
        for category, rows in sorted(categories.items())
    }
    safety_indices = [index for index, value in enumerate(claim_gold) if value != "SUPPORTED"]
    false_supported_safety = _safe_div(
        sum(claim_values[index] == "SUPPORTED" for index in safety_indices),
        len(safety_indices),
    )

    metrics = {
        "minimal_compatible_span_precision": _safe_div(
            len(minimal_correct), len(predicted_nonnull)
        ),
        "minimal_compatible_span_recall": _safe_div(len(minimal_correct), len(gold_nonnull)),
        "compatibility_macro_f1": _macro_f1(
            gold_compat, pred_compat, ("COMPATIBLE", "INCOMPATIBLE")
        ),
        "compatible_recall": _recall(gold_compat, pred_compat, "COMPATIBLE"),
        "incompatible_recall": _recall(gold_compat, pred_compat, "INCOMPATIBLE"),
        "relevant_but_insufficient_compatible_recall": _subtypes_rate(
            pair_pred,
            {
                "relevant_but_insufficient_missing_value",
                "relevant_but_insufficient_missing_temporal_scope",
                "relevant_but_insufficient_missing_conditional_scope",
            },
            "compatibility",
            "COMPATIBLE",
        ),
        "missing_temporal_scope_stays_compatible": _subtype_rate(
            pair_pred,
            "relevant_but_insufficient_missing_temporal_scope",
            "compatibility",
            "COMPATIBLE",
        ),
        "missing_conditional_scope_stays_compatible": _subtype_rate(
            pair_pred,
            "relevant_but_insufficient_missing_conditional_scope",
            "compatibility",
            "COMPATIBLE",
        ),
        "entity_mismatch_rejection": _subtype_rate(
            pair_pred, "entity_scope_mismatch", "compatibility", "INCOMPATIBLE"
        ),
        "predicate_mismatch_rejection": _subtype_rate(
            pair_pred, "predicate_scope_mismatch", "compatibility", "INCOMPATIBLE"
        ),
        "target_slot_mismatch_rejection": _subtype_rate(
            pair_pred, "target_slot_mismatch", "compatibility", "INCOMPATIBLE"
        ),
        "temporal_scope_mismatch_rejection": _subtype_rate(
            pair_pred, "temporal_scope_mismatch", "compatibility", "INCOMPATIBLE"
        ),
        "location_scope_mismatch_rejection": _subtype_rate(
            pair_pred, "location_scope_mismatch", "compatibility", "INCOMPATIBLE"
        ),
        "organizational_scope_mismatch_rejection": _subtype_rate(
            pair_pred, "organizational_scope_mismatch", "compatibility", "INCOMPATIBLE"
        ),
        "conditional_scope_mismatch_rejection": _subtype_rate(
            pair_pred, "conditional_scope_mismatch", "compatibility", "INCOMPATIBLE"
        ),
        "modality_quantification_scope_mismatch_rejection": _subtype_rate(
            pair_pred,
            "modality_quantification_scope_mismatch",
            "compatibility",
            "INCOMPATIBLE",
        ),
        "cross_document_irrelevance_rejection": _subtype_rate(
            pair_pred, "cross_document_irrelevance", "compatibility", "INCOMPATIBLE"
        ),
        "same_domain_irrelevance_rejection": _subtype_rate(
            pair_pred, "same_domain_irrelevance", "compatibility", "INCOMPATIBLE"
        ),
        "sufficiency_macro_f1": _macro_f1(set_gold_suff, set_pred_suff, suff_labels),
        "sufficient_recall": _recall(set_gold_suff, set_pred_suff, "SUFFICIENT"),
        "insufficient_recall": _recall(set_gold_suff, set_pred_suff, "INSUFFICIENT"),
        "complementary_evidence_sufficiency_recall": _subtype_rate(
            set_pred, "complementary_two_span_support", "sufficiency", "SUFFICIENT"
        ),
        "unresolved_scope_gap_insufficiency_recall": _subtypes_rate(
            set_pred,
            {"compatible_incomplete_scope_gap", "compatible_multi_span_unresolved_scope_gap"},
            "sufficiency",
            "INSUFFICIENT",
        ),
        "polarity_macro_f1": _macro_f1(gold_polarity, pred_polarity, ("SUPPORTS", "REFUTES")),
        "supports_recall": _recall(gold_polarity, pred_polarity, "SUPPORTS"),
        "refutes_recall": _recall(gold_polarity, pred_polarity, "REFUTES"),
        "final_relation_macro_f1": _macro_f1(final_gold, final_pred, RELATION_LABELS),
        "entailed_recall": _recall(final_gold, final_pred, "ENTAILED"),
        "contradicted_recall": _recall(final_gold, final_pred, "CONTRADICTED"),
        "unknown_recall": _recall(final_gold, final_pred, "UNKNOWN"),
        "conflict_detection_accuracy": _subtype_rate(
            set_pred, "support_refute_conflict", "final_relation", "CONFLICTING_EVIDENCE"
        ),
        "irrelevant_distractor_robustness": _subtype_rate(
            set_pred,
            "complete_support_with_irrelevant_distractor",
            "final_relation",
            "ENTAILED",
        ),
        "claim_category_macro_accuracy": sum(category_accuracy.values()) / len(category_accuracy),
        "supported_precision": _precision(claim_gold, claim_values, "SUPPORTED"),
        "supported_recall": _recall(claim_gold, claim_values, "SUPPORTED"),
        "complementary_multi_span_supported_recall": category_accuracy[
            "complementary_multi_span_supported"
        ],
        "compatible_insufficient_unsupported_accuracy": category_accuracy[
            "compatible_insufficient"
        ],
        "support_refute_conflict_accuracy": category_accuracy["support_refute_conflict"],
        "citation_invalid_accuracy": category_accuracy["citation_invalid"],
        "stale_evidence_accuracy": category_accuracy["stale_evidence"],
        "registered_conflict_accuracy": category_accuracy["registered_conflict"],
        "false_supported_safety": false_supported_safety,
    }
    return {
        "mismatch_threshold": mismatch_threshold,
        "coverage_threshold": coverage_threshold,
        "metrics": metrics,
        "category_accuracy": category_accuracy,
        "pair_predictions": pair_pred,
        "set_predictions": set_pred,
        "claim_predictions": claim_pred,
    }


def requirement_checks(
    metrics: dict[str, float], requirements: dict[str, float]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, threshold in requirements.items():
        if name.endswith("_min"):
            metric = name[:-4]
            checks[name] = float(metrics[metric]) >= float(threshold)
        elif name.endswith("_max"):
            metric = name[:-4]
            checks[name] = float(metrics[metric]) <= float(threshold)
        else:
            raise RuntimeError(f"Unsupported A4.5b-M3 readiness requirement: {name}")
    return checks


def _minimum_core_recall(metrics: dict[str, float]) -> float:
    names = (
        "compatible_recall",
        "incompatible_recall",
        "sufficient_recall",
        "insufficient_recall",
        "supports_recall",
        "refutes_recall",
        "entailed_recall",
        "contradicted_recall",
        "unknown_recall",
    )
    return min(float(metrics[name]) for name in names)


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    checks = candidate["requirement_checks"]
    return (
        bool(candidate["calibration_ready"]),
        sum(checks.values()),
        float(metrics["final_relation_macro_f1"]),
        _minimum_core_recall(metrics),
        float(metrics["claim_category_macro_accuracy"]),
        float(metrics["compatibility_macro_f1"]),
        float(metrics["sufficiency_macro_f1"]),
        float(metrics["polarity_macro_f1"]),
        float(candidate["coverage_threshold"]),
        float(candidate["mismatch_threshold"]),
    )


def select_parameters(
    suite: dict[str, Any],
    raw_pair_rows: list[dict[str, Any]],
    raw_set_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    calibration = config["calibration"]
    requirements = calibration["calibration_readiness_requirements"]
    candidates: list[dict[str, Any]] = []
    feasible_count = 0
    for mismatch in calibration["mismatch_threshold_grid"]:
        for coverage in calibration["coverage_threshold_grid"]:
            evaluated = evaluate_candidate(
                suite, raw_pair_rows, raw_set_rows, float(mismatch), float(coverage)
            )
            checks = requirement_checks(evaluated["metrics"], requirements)
            evaluated["requirement_checks"] = checks
            evaluated["requirements_passed"] = sum(checks.values())
            evaluated["calibration_ready"] = all(checks.values())
            feasible_count += int(evaluated["calibration_ready"])
            candidates.append(evaluated)
    if len(candidates) != int(calibration["joint_candidate_count"]):
        raise RuntimeError("A4.5b-M3 joint parameter grid cardinality drifted")
    selected = max(candidates, key=candidate_key)
    return {
        "candidate_count": len(candidates),
        "feasible_candidate_count": feasible_count,
        "selected": selected,
    }

"""Independent reconstruction audit for the A4.5b calibration-only artifact."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from calibration_cases_a45b import build_calibration_only, calibration_manifest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models" / "assistance_grounding_a45b_v1.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
RELATIONS = ("ENTAILED", "CONTRADICTED", "UNKNOWN")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected object row in {path}")
        rows.append(value)
    return rows


def _stats(gold: list[str], predicted: list[str], label: str) -> tuple[float, float, float]:
    pairs = list(zip(gold, predicted, strict=True))
    tp = sum(g == label and p == label for g, p in pairs)
    fp = sum(g != label and p == label for g, p in pairs)
    fn = sum(g == label and p != label for g, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _macro_f1(gold: list[str], predicted: list[str], labels: tuple[str, ...]) -> float:
    return sum(_stats(gold, predicted, label)[2] for label in labels) / len(labels)


def _token_overlap(gold: str, predicted: str) -> tuple[float, float]:
    gold_tokens = Counter(token.lower() for token in TOKEN_RE.findall(gold))
    pred_tokens = Counter(token.lower() for token in TOKEN_RE.findall(predicted))
    overlap = sum((gold_tokens & pred_tokens).values())
    precision = overlap / sum(pred_tokens.values()) if pred_tokens else 0.0
    recall = overlap / sum(gold_tokens.values()) if gold_tokens else 0.0
    return precision, recall


def _component(score: dict[str, Any], rel_t: float, suff_t: float) -> tuple[str, str, str, str]:
    relevance = "RELEVANT" if float(score["relevance_score"]) >= rel_t else "IRRELEVANT"
    probs = score["nli_probabilities"]
    entailment = float(probs["entailment"])
    contradiction = float(probs["contradiction"])
    strength = max(entailment, contradiction)
    sufficiency = "SUFFICIENT" if strength >= suff_t else "INSUFFICIENT"
    if entailment == contradiction:
        polarity = "UNRESOLVED"
    else:
        polarity = "SUPPORTS" if entailment > contradiction else "REFUTES"
    if relevance == "IRRELEVANT" or sufficiency == "INSUFFICIENT" or polarity == "UNRESOLVED":
        relation = "UNKNOWN"
    else:
        relation = "ENTAILED" if polarity == "SUPPORTS" else "CONTRADICTED"
    return relevance, sufficiency, polarity, relation


def _false_contradiction(
    pairs: list[dict[str, Any]], relations: list[str], subtype: str
) -> float:
    indices = [index for index, row in enumerate(pairs) if row["subtype"] == subtype]
    errors = sum(relations[index] == "CONTRADICTED" for index in indices)
    return errors / len(indices)


def _claim_accuracy(claims: list[dict[str, Any]]) -> float:
    correct = 0
    for row in claims:
        gate = str(row["deterministic_gate"])
        relations = [str(value) for value in row["atom_relations"]]
        if gate == "CITATION_INVALID":
            predicted = "CITATION_INVALID"
        elif gate == "STALE_EVIDENCE":
            predicted = "STALE_EVIDENCE"
        elif gate == "REGISTERED_CONFLICT":
            predicted = "CONFLICTING_EVIDENCE"
        elif "ENTAILED" in relations and "CONTRADICTED" in relations:
            predicted = "CONFLICTING_EVIDENCE"
        elif relations and all(value == "ENTAILED" for value in relations):
            predicted = "SUPPORTED"
        else:
            predicted = "UNSUPPORTED"
        correct += predicted == row["expected_verdict"]
    return correct / len(claims)


def _metrics(
    pairs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    rel_t: float,
    suff_t: float,
    claim_accuracy: float,
) -> dict[str, float]:
    predicted = [_component(row, rel_t, suff_t) for row in scores]
    pred_relevance = [row[0] for row in predicted]
    pred_sufficiency = [row[1] for row in predicted]
    pred_polarity = [row[2] for row in predicted]
    pred_relations = [row[3] for row in predicted]

    gold_relevance = [str(row["gold"]["relevance"]) for row in pairs]
    relevant_recall = _stats(gold_relevance, pred_relevance, "RELEVANT")[1]
    irrelevant_recall = _stats(gold_relevance, pred_relevance, "IRRELEVANT")[1]
    relevant_indices = [
        index for index, row in enumerate(pairs) if row["gold"]["relevance"] == "RELEVANT"
    ]
    gold_sufficiency = [str(pairs[index]["gold"]["sufficiency"]) for index in relevant_indices]
    pred_sufficiency_relevant = [pred_sufficiency[index] for index in relevant_indices]
    sufficient_recall = _stats(gold_sufficiency, pred_sufficiency_relevant, "SUFFICIENT")[1]
    insufficient_recall = _stats(gold_sufficiency, pred_sufficiency_relevant, "INSUFFICIENT")[1]

    sufficient_indices = [
        index
        for index in relevant_indices
        if pairs[index]["gold"]["sufficiency"] == "SUFFICIENT"
    ]
    gold_polarity = [str(pairs[index]["gold"]["polarity"]) for index in sufficient_indices]
    pred_polarity_sufficient = [pred_polarity[index] for index in sufficient_indices]
    supports_recall = _stats(gold_polarity, pred_polarity_sufficient, "SUPPORTS")[1]
    refutes_recall = _stats(gold_polarity, pred_polarity_sufficient, "REFUTES")[1]

    gold_relations = [str(row["gold"]["final_relation"]) for row in pairs]
    entailed_recall = _stats(gold_relations, pred_relations, "ENTAILED")[1]
    contradicted_recall = _stats(gold_relations, pred_relations, "CONTRADICTED")[1]
    unknown_recall = _stats(gold_relations, pred_relations, "UNKNOWN")[1]

    span_precision: list[float] = []
    span_recall: list[float] = []
    for index in relevant_indices:
        precision, recall = _token_overlap(
            str(pairs[index]["gold"]["minimal_evidence_text"]),
            str(scores[index]["selected_span"]),
        )
        span_precision.append(precision)
        span_recall.append(recall)
    context = [
        index for index, row in enumerate(pairs) if row["subtype"] == "context_contamination_support"
    ]
    context_accuracy = sum(pred_relations[index] == "ENTAILED" for index in context) / len(context)
    return {
        "minimal_evidence_span_precision": sum(span_precision) / len(span_precision),
        "minimal_evidence_span_recall": sum(span_recall) / len(span_recall),
        "relevance_macro_f1": _macro_f1(
            gold_relevance,
            pred_relevance,
            ("RELEVANT", "IRRELEVANT"),
        ),
        "relevant_recall": relevant_recall,
        "irrelevant_recall": irrelevant_recall,
        "sufficiency_macro_f1_on_relevant_pairs": _macro_f1(
            gold_sufficiency,
            pred_sufficiency_relevant,
            ("SUFFICIENT", "INSUFFICIENT"),
        ),
        "sufficient_recall": sufficient_recall,
        "insufficient_recall": insufficient_recall,
        "polarity_macro_f1_on_relevant_sufficient_pairs": _macro_f1(
            gold_polarity,
            pred_polarity_sufficient,
            ("SUPPORTS", "REFUTES"),
        ),
        "supports_recall": supports_recall,
        "refutes_recall": refutes_recall,
        "final_relation_macro_f1": _macro_f1(gold_relations, pred_relations, RELATIONS),
        "entailed_recall": entailed_recall,
        "contradicted_recall": contradicted_recall,
        "unknown_recall": unknown_recall,
        "cross_document_irrelevance_false_contradiction_rate": _false_contradiction(
            pairs,
            pred_relations,
            "cross_document_irrelevance",
        ),
        "same_domain_irrelevance_false_contradiction_rate": _false_contradiction(
            pairs,
            pred_relations,
            "same_domain_irrelevance",
        ),
        "relevant_insufficient_false_contradiction_rate": _false_contradiction(
            pairs,
            pred_relations,
            "relevant_but_insufficient",
        ),
        "context_contamination_support_accuracy": context_accuracy,
        "claim_composition_accuracy": claim_accuracy,
    }


def _checks(metrics: dict[str, float], requirements: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    output: dict[str, bool] = {}
    for name, raw_threshold in requirements.items():
        threshold = float(raw_threshold)
        if name.endswith("_min"):
            output[name] = metrics[name.removesuffix("_min")] >= threshold - 1e-12
        elif name.endswith("_max"):
            output[name] = metrics[name.removesuffix("_max")] <= threshold + 1e-12
        else:
            raise RuntimeError(f"Unsupported requirement {name}")
    return output, all(output.values())


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    minimum_recall = min(
        metrics["entailed_recall"],
        metrics["contradicted_recall"],
        metrics["unknown_recall"],
    )
    false_sum = (
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
        -false_sum,
        float(candidate["sufficiency_threshold"]),
        float(candidate["relevance_threshold"]),
    )


def _grid(grid: dict[str, Any]) -> list[float]:
    start = int(grid["integer_start"])
    stop = int(grid["integer_stop"])
    step = int(grid["integer_step"])
    scale = float(grid["scale"])
    return [round(value * scale, 10) for value in range(start, stop + step, step)]


def _reconstruct_selection(
    pairs: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    setup = config["threshold_calibration"]
    relevance_values = _grid(setup["relevance_grid"])
    sufficiency_values = _grid(setup["sufficiency_grid"])
    if len(relevance_values) * len(sufficiency_values) != 12050:
        raise RuntimeError("Independent A4.5b grid reconstruction drifted")
    claim_accuracy = _claim_accuracy(claims)
    requirements = config["calibration_readiness_requirements"]
    best_any: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    feasible_count = 0
    for rel_t in relevance_values:
        for suff_t in sufficiency_values:
            metrics = _metrics(pairs, scores, rel_t, suff_t, claim_accuracy)
            checks, passed = _checks(metrics, requirements)
            candidate = {
                "relevance_threshold": rel_t,
                "sufficiency_threshold": suff_t,
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
        raise RuntimeError("Independent A4.5b grid was empty")
    selected = best_feasible if best_feasible is not None else best_any
    return {
        "selected": selected,
        "feasible_candidate_count": feasible_count,
        "joint_candidates_evaluated": 12050,
        "selection_used_feasible_set": best_feasible is not None,
    }


def _assert_close(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"Key mismatch at {path}")
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"List length mismatch at {path}")
        for index, (l_value, r_value) in enumerate(zip(left, right, strict=True)):
            _assert_close(l_value, r_value, f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Numeric mismatch at {path}: {left} != {right}")
        return
    if left != right:
        raise RuntimeError(f"Mismatch at {path}: {left!r} != {right!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = _load_json(CONFIG)
    results = _load_json(args.output_dir / "results.json")
    scores = _load_jsonl(args.output_dir / "calibration_pair_scores.jsonl")
    materialized = build_calibration_only()
    manifest = calibration_manifest()
    pairs = materialized["pair_rows"]
    claims = materialized["claim_rows"]

    pair_ids = [str(row["pair_id"]) for row in pairs]
    score_ids = [str(row["pair_id"]) for row in scores]
    if len(scores) != 360 or len(set(score_ids)) != 360 or score_ids != pair_ids:
        raise RuntimeError("A4.5b raw score rows do not match registered calibration pairs")
    for pair, score in zip(pairs, scores, strict=True):
        if score["split"] != "calibration":
            raise RuntimeError("A4.5b artifact contains a non-calibration score row")
        if score["gold"] != pair["gold"]:
            raise RuntimeError(f"Gold drift for {pair['pair_id']}")
        probabilities = score["nli_probabilities"]
        values = [float(probabilities[name]) for name in ("entailment", "neutral", "contradiction")]
        if any(value < 0.0 or value > 1.0 for value in values):
            raise RuntimeError(f"Invalid NLI probability for {pair['pair_id']}")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"NLI probabilities do not sum to one for {pair['pair_id']}")

    reconstructed = _reconstruct_selection(pairs, claims, scores, config)
    _assert_close(reconstructed, results["threshold_selection"], "threshold_selection")
    scientific_pass = bool(reconstructed["selected"]["calibration_ready"])
    expected_status = (
        "PASSED_CALIBRATION_READINESS_THRESHOLDS_FROZEN"
        if scientific_pass
        else "FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED"
    )
    if results["scientific_pass"] is not scientific_pass:
        raise RuntimeError("A4.5b scientific pass bit failed reconstruction")
    if results["scientific_status"] != expected_status:
        raise RuntimeError("A4.5b scientific status failed reconstruction")
    if results["calibration_pairs_sha256"] != manifest["calibration_pairs_sha256"]:
        raise RuntimeError("A4.5b calibration pair hash drifted in results")
    if results["calibration_claims_sha256"] != manifest["calibration_claims_sha256"]:
        raise RuntimeError("A4.5b calibration claim hash drifted in results")
    for field in (
        "validation_rows_materialized",
        "validation_rows_scored",
        "confirmatory_queries_inspected",
        "confirmatory_queries_scored",
        "a44d_rows_rescored",
        "a44a_rows_rescored",
    ):
        if int(results[field]) != 0:
            raise RuntimeError(f"Forbidden A4.5b result counter is nonzero: {field}")
    if results["post_result_rescue_authorized"] is not False:
        raise RuntimeError("A4.5b post-result rescue must remain forbidden")
    if results["next_checkpoint_authorized"] is not False:
        raise RuntimeError("A4.5c must remain separately gated")

    audit = {
        "status": "PASSED_A45B_CALIBRATION_ONLY_RECONSTRUCTION",
        "scientific_status": expected_status,
        "scientific_pass": scientific_pass,
        "calibration_pairs": 360,
        "pair_ids_exact_and_unique": True,
        "all_rows_calibration_only": True,
        "raw_probability_simplex_verified": True,
        "threshold_grid_reconstructed": 12050,
        "threshold_selection_reconstructed": True,
        "registered_metrics_reconstructed": True,
        "requirement_checks_reconstructed": True,
        "validation_rows_materialized": 0,
        "validation_rows_scored": 0,
        "confirmatory_queries_inspected": 0,
        "confirmatory_queries_scored": 0,
        "post_result_rescue_authorized": False,
        "next_checkpoint_authorized": False,
    }
    (args.output_dir / "post_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# A4.5b independent calibration audit",
        "",
        f"Audit status: **{audit['status']}**",
        f"Scientific status: **{audit['scientific_status']}**",
        "",
        "The 360 registered calibration pairs, raw probability rows, all 12,050 threshold",
        "candidates, selected thresholds, metrics, requirement checks, and scientific status",
        "were independently reconstructed. Fresh validation and confirmatory counters are zero.",
        "",
    ]
    (args.output_dir / "post_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

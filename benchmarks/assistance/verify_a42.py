"""Independently reconstruct and audit A4.2 development assistance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from helix_support_intelligence.data.helixbank import (  # noqa: E402
    INTENTS,
    generate_bundle,
)

PROTOCOL_PATH = ROOT / "configs" / "models" / "assistance_protocol_v1.json"
EXECUTION_PATH = ROOT / "configs" / "models" / "assistance_execution_a42_v1.json"
BINDING_PATH = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"
SUBSETS_PATH = ROOT / "configs" / "models" / "assistance_a41_subsets_v1.json"
CANDIDATES = ("G0", "G1", "G2")
ATTACK_TYPES = (
    "direct_injection",
    "citation_spoof",
    "indirect_injection",
    "archived_distractor",
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return cast(dict[str, Any], payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} contains a non-object JSONL row.")
        rows.append(cast(dict[str, Any], payload))
    return rows


def partition() -> tuple[set[str], set[str]]:
    bundle = generate_bundle()
    conflicts = {
        str(row["intent"])
        for row in bundle.queries
        if row["case_type"] == "conflicting_evidence"
    }
    non_conflicts = set(INTENTS) - conflicts

    def ordered(values: set[str]) -> list[str]:
        return sorted(
            values,
            key=lambda intent: hashlib.sha256(
                f"20260819:{intent}".encode()
            ).hexdigest(),
        )

    development = set(ordered(conflicts)[:5]) | set(ordered(non_conflicts)[:55])
    return development, set(INTENTS) - development


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    factual = sum(int(row["factual_sentence_count"]) for row in rows)
    unsupported = sum(int(row["unsupported_sentence_count"]) for row in rows)
    cited = sum(len(row["citations"]) for row in rows)
    valid = sum(
        len(row["citations"]) * float(row["citation_precision"])
        for row in rows
    )
    stale = sum(
        len(row["citations"]) * float(row["stale_citation_rate"])
        for row in rows
    )
    applicable = [row for row in rows if row["citation_completeness"] is not None]
    gold = sum(len(row["gold_citations"]) for row in applicable)
    recovered = sum(
        len(row["gold_citations"]) * float(row["citation_completeness"])
        for row in applicable
    )
    precision = valid / cited if cited else 1.0
    completeness = recovered / gold if gold else 1.0
    f1 = (
        2.0 * precision * completeness / (precision + completeness)
        if precision + completeness
        else 0.0
    )
    nonanswerable = [
        row
        for row in rows
        if row["expected_decision"] != "ANSWER_WITH_EVIDENCE"
    ]
    answerable = [
        row
        for row in rows
        if row["expected_decision"] == "ANSWER_WITH_EVIDENCE"
    ]
    costs = [
        float(row["estimated_cost_usd"])
        for row in rows
        if row["estimated_cost_usd"] is not None
    ]
    return {
        "strict_grounded_success_rate": sum(
            bool(row["strict_grounded_success"]) for row in rows
        )
        / total,
        "decision_exact_match": sum(
            row["decision"] == row["expected_decision"] for row in rows
        )
        / total,
        "unsupported_sentence_rate": unsupported / factual if factual else 0.0,
        "answer_level_unsupported_rate": sum(
            int(row["unsupported_sentence_count"]) > 0 for row in rows
        )
        / total,
        "citation_precision": precision,
        "citation_completeness": completeness,
        "citation_f1": f1,
        "stale_citation_rate": stale / cited if cited else 0.0,
        "unsafe_answer_rate_on_nonanswerable": sum(
            row["decision"] == "ANSWER_WITH_EVIDENCE" for row in nonanswerable
        )
        / len(nonanswerable),
        "over_abstention_rate_on_answerable": sum(
            row["decision"] != "ANSWER_WITH_EVIDENCE" for row in answerable
        )
        / len(answerable),
        "schema_valid_rate": sum(bool(row["schema_valid"]) for row in rows) / total,
        "provider_failure_rate": sum(row["failure"] is not None for row in rows) / total,
        "mean_estimated_cost_usd": statistics.fmean(costs) if costs else 0.0,
        "max_estimated_cost_usd": max(costs) if costs else 0.0,
        "total_estimated_cost_usd": sum(costs),
    }


def strict_bootstrap(
    candidate_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    candidate = {
        (str(row["intent"]), str(row["query_id"])): float(
            row["strict_grounded_success"]
        )
        for row in candidate_rows
    }
    comparator = {
        (str(row["intent"]), str(row["query_id"])): float(
            row["strict_grounded_success"]
        )
        for row in comparator_rows
    }
    intents = sorted({key[0] for key in candidate})

    def sample_mean(values: dict[tuple[str, str], float], sampled: list[str]) -> float:
        data: list[float] = []
        for intent in sampled:
            qids = sorted(key[1] for key in values if key[0] == intent)
            data.extend(values[(intent, query_id)] for query_id in qids)
        return statistics.fmean(data)

    point = sample_mean(candidate, intents) - sample_mean(comparator, intents)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = [
            str(item)
            for item in rng.choice(intents, size=len(intents), replace=True)
        ]
        draws[index] = sample_mean(candidate, sampled) - sample_mean(
            comparator,
            sampled,
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"point": point, "ci95": [float(lower), float(upper)]}


def unsupported_bootstrap(
    candidate_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    def grouped(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
        by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_intent[str(row["intent"])].append(row)
        return {
            intent: (
                sum(int(row["unsupported_sentence_count"]) for row in values),
                sum(int(row["factual_sentence_count"]) for row in values),
            )
            for intent, values in by_intent.items()
        }

    candidate = grouped(candidate_rows)
    comparator = grouped(comparator_rows)
    intents = sorted(candidate)

    def rate(values: dict[str, tuple[int, int]], sampled: list[str]) -> float:
        unsupported = sum(values[intent][0] for intent in sampled)
        factual = sum(values[intent][1] for intent in sampled)
        return unsupported / factual if factual else 0.0

    point = rate(candidate, intents) - rate(comparator, intents)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = [
            str(item)
            for item in rng.choice(intents, size=len(intents), replace=True)
        ]
        draws[index] = rate(candidate, sampled) - rate(comparator, sampled)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"point": point, "ci95": [float(lower), float(upper)]}


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray([float(row["latency_ms"]) for row in rows], dtype=float)
    mean = float(np.mean(values))
    return {
        "mean_ms": mean,
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "requests_per_second": 1000.0 / mean if mean > 0 else math.inf,
    }


def zero_failure_upper(total: int, failures: int) -> float | None:
    if total <= 0 or failures != 0:
        return None
    return 1.0 - 0.05 ** (1.0 / total)


def adversarial_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in CANDIDATES:
        result[candidate] = {}
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
        for attack_type in ATTACK_TYPES:
            selected = [
                row
                for row in candidate_rows
                if row["attack_type"] == attack_type
            ]
            failures = sum(bool(row["attack_failure"]) for row in selected)
            result[candidate][attack_type] = {
                "cases": len(selected),
                "failures": failures,
                "failure_rate": failures / len(selected) if selected else 0.0,
                "zero_failure_one_sided_95_upper": zero_failure_upper(
                    len(selected),
                    failures,
                ),
            }
    return result


def repeatability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in CANDIDATES:
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            if row["candidate_id"] == candidate:
                grouped[str(row["query_id"])].append(
                    str(row["canonical_output"])
                )
        stable = sum(len(set(values)) == 1 for values in grouped.values())
        result[candidate] = {
            "cases": len(grouped),
            "all_three_identical_cases": stable,
            "exact_repeatability_rate": stable / len(grouped) if grouped else 0.0,
        }
    return result


def diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = generate_bundle()
    documents = {str(row["document_id"]): row for row in bundle.documents}
    result: dict[str, Any] = {"case_type": {}, "intent": {}, "document_kind": {}}
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
        for family in ("case_type", "intent"):
            groups = sorted({str(row[family]) for row in candidate_rows})
            result[family][candidate] = {
                group: aggregate(
                    [row for row in candidate_rows if str(row[family]) == group]
                )
                for group in groups
            }
        kind_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            kinds = sorted(
                {
                    str(documents[document_id]["kind"])
                    for document_id in row["presented_document_ids"]
                }
            )
            label = "+".join(kinds) if kinds else "none"
            kind_groups[label].append(row)
        result["document_kind"][candidate] = {
            group: aggregate(values)
            for group, values in sorted(kind_groups.items())
        }
    return result


def adoption(
    metrics: dict[str, dict[str, float]],
    quality: dict[str, list[dict[str, Any]]],
    latency: dict[str, dict[str, float]],
    binding: dict[str, Any],
    protocol: dict[str, Any],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    winner = "G0"
    decisions: list[dict[str, Any]] = []
    rule = protocol["complexity_adoption"]["candidate_rule"]
    budgets = binding["budgets"]
    for candidate in ("G1", "G2"):
        comparator = winner
        bootstrap = strict_bootstrap(
            quality[candidate],
            quality[comparator],
            seed,
            replicates,
        )
        delta_strict = (
            metrics[candidate]["strict_grounded_success_rate"]
            - metrics[comparator]["strict_grounded_success_rate"]
        )
        delta_unsupported = (
            metrics[candidate]["unsupported_sentence_rate"]
            - metrics[comparator]["unsupported_sentence_rate"]
        )
        delta_unsafe = (
            metrics[candidate]["unsafe_answer_rate_on_nonanswerable"]
            - metrics[comparator]["unsafe_answer_rate_on_nonanswerable"]
        )
        delta_f1 = (
            metrics[candidate]["citation_f1"]
            - metrics[comparator]["citation_f1"]
        )
        checks = {
            "delta_strict_at_least_0_01": delta_strict
            >= float(rule["minimum_delta_strict_grounded_success_rate"]),
            "strict_ci_lower_above_zero": bootstrap["ci95"][0] > 0.0,
            "unsupported_increase_within_0_005": delta_unsupported
            <= float(rule["maximum_unsupported_sentence_rate_increase"]),
            "unsafe_increase_within_0_005": delta_unsafe
            <= float(rule["maximum_unsafe_answer_rate_increase"]),
            "citation_f1_decrease_within_0_005": delta_f1
            >= -float(rule["maximum_citation_f1_decrease"]),
            "p95_within_budget": latency[candidate]["p95_ms"]
            <= float(binding["budgets"]["p95_latency_ms"][candidate]),
            "max_quality_cost_within_budget": metrics[candidate][
                "max_estimated_cost_usd"
            ]
            <= float(
                budgets["maximum_estimated_cost_usd_per_request"][candidate]
            ),
        }
        qualifies = all(checks.values())
        decisions.append(
            {
                "candidate": candidate,
                "comparator": comparator,
                "delta_strict_grounded_success_rate": delta_strict,
                "strict_cluster_bootstrap": bootstrap,
                "delta_unsupported_sentence_rate": delta_unsupported,
                "delta_unsafe_answer_rate": delta_unsafe,
                "delta_citation_f1": delta_f1,
                "candidate_p95_ms": latency[candidate]["p95_ms"],
                "candidate_max_quality_cost_usd": metrics[candidate][
                    "max_estimated_cost_usd"
                ],
                "checks": checks,
                "qualifies": qualifies,
            }
        )
        if qualifies:
            winner = candidate
    return {"registered_winner": winner, "decisions": decisions}


def verify_counts(
    quality_rows: list[dict[str, Any]],
    adversarial_rows: list[dict[str, Any]],
    repeatability_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    development, confirmatory = partition()
    bundle = generate_bundle()
    development_ids = {
        str(row["query_id"])
        for row in bundle.queries
        if str(row["intent"]) in development
    }
    confirmatory_ids = {
        str(row["query_id"])
        for row in bundle.queries
        if str(row["intent"]) in confirmatory
    }
    checks: dict[str, bool] = {
        "quality_records_720": len(quality_rows) == 720,
        "adversarial_records_429": len(adversarial_rows) == 429,
        "repeatability_records_270": len(repeatability_rows) == 270,
        "latency_samples_540": len(latency_rows) == 540,
    }
    all_rows = quality_rows + adversarial_rows + repeatability_rows + latency_rows
    checks["no_confirmatory_query_id_opened"] = all(
        str(row["query_id"]) not in confirmatory_ids for row in all_rows
    )
    checks["all_query_ids_are_development"] = all(
        str(row["query_id"]) in development_ids for row in all_rows
    )
    for candidate in CANDIDATES:
        candidate_quality = [
            row for row in quality_rows if row["candidate_id"] == candidate
        ]
        checks[f"{candidate}_quality_240"] = len(candidate_quality) == 240
        checks[f"{candidate}_quality_unique_queries"] = (
            len({str(row["query_id"]) for row in candidate_quality}) == 240
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"A4.2 raw-count audit failed: {failed}")
    return checks


def write_report(results: dict[str, Any], path: Path) -> None:
    metrics = results["metrics"]
    latency = results["latency"]
    lines = [
        "# Phase 4 A4.2 development assistance result",
        "",
        f"**Registered development winner: {results['complexity_adoption']['registered_winner']}**",
        "",
        "| Candidate | Strict grounded success | Decision EM | Unsupported sentence rate | Citation F1 | P95 ms | Max quality cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        item = metrics[candidate]
        lines.append(
            f"| {candidate} | {item['strict_grounded_success_rate']:.4f} | "
            f"{item['decision_exact_match']:.4f} | "
            f"{item['unsupported_sentence_rate']:.4f} | "
            f"{item['citation_f1']:.4f} | {latency[candidate]['p95_ms']:.2f} | "
            f"{item['max_estimated_cost_usd']:.6f} |"
        )
    h1 = results["hypotheses"]["H1"]
    h2 = results["hypotheses"]["H2"]
    lines.extend(
        [
            "",
            f"H1 G1-G0 strict grounded success: {h1['point']:.6f}, 95% CI "
            f"[{h1['ci95'][0]:.6f}, {h1['ci95'][1]:.6f}], "
            f"**{h1['verdict']}**.",
            f"H2 G2-G1 unsupported sentence rate: {h2['point']:.6f}, 95% CI "
            f"[{h2['ci95'][0]:.6f}, {h2['ci95'][1]:.6f}], "
            f"**{h2['verdict']}**.",
            "",
            "The 68-query confirmatory assistance partition remained unopened.",
            "The adversarial evidence in A4.2 is development-intent only; the full registered "
            "77-intent adversarial surface remains deferred.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir

    execution = load_json(EXECUTION_PATH)
    protocol = load_json(PROTOCOL_PATH)
    binding = load_json(BINDING_PATH)
    subsets = load_json(SUBSETS_PATH)
    compatibility = load_json(output_dir / "compatibility.json")
    quality_rows = load_jsonl(output_dir / "quality_records.jsonl")
    adversarial_rows = load_jsonl(output_dir / "adversarial_records.jsonl")
    repeatability_rows = load_jsonl(output_dir / "repeatability_records.jsonl")
    latency_rows = load_jsonl(output_dir / "latency_samples.jsonl")

    raw_checks = verify_counts(
        quality_rows,
        adversarial_rows,
        repeatability_rows,
        latency_rows,
    )
    if compatibility["performance_score_computed"] is not False:
        raise RuntimeError("Compatibility probe was incorrectly treated as performance evidence.")

    quality = {
        candidate: [row for row in quality_rows if row["candidate_id"] == candidate]
        for candidate in CANDIDATES
    }
    metrics = {candidate: aggregate(quality[candidate]) for candidate in CANDIDATES}
    latency = {
        candidate: latency_summary(
            [row for row in latency_rows if row["candidate_id"] == candidate]
        )
        for candidate in CANDIDATES
    }
    seed = int(execution["inference"]["seed"])
    replicates = int(execution["inference"]["replicates"])
    h1 = strict_bootstrap(quality["G1"], quality["G0"], seed, replicates)
    h1["verdict"] = (
        "SUPPORTED"
        if h1["point"] > 0 and h1["ci95"][0] > 0
        else "ADVERSE"
        if h1["point"] < 0 and h1["ci95"][1] < 0
        else "INCONCLUSIVE"
    )
    h2 = unsupported_bootstrap(quality["G2"], quality["G1"], seed, replicates)
    h2["verdict"] = (
        "SUPPORTED"
        if h2["point"] < 0 and h2["ci95"][1] < 0
        else "ADVERSE"
        if h2["point"] > 0 and h2["ci95"][0] > 0
        else "INCONCLUSIVE"
    )
    adversarial = adversarial_summary(adversarial_rows)
    repeatability = repeatability_summary(repeatability_rows)
    adoption_result = adoption(
        metrics,
        quality,
        latency,
        binding,
        protocol,
        seed,
        replicates,
    )
    diagnostic_slices = diagnostics(quality_rows)
    (output_dir / "diagnostic_slices.json").write_text(
        json.dumps(diagnostic_slices, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    all_cost_rows = (
        quality_rows
        + adversarial_rows
        + repeatability_rows
        + latency_rows
    )
    total_cost = sum(
        float(row["estimated_cost_usd"] or 0.0)
        for row in all_cost_rows
        if row["candidate_id"] in {"G1", "G2"}
    )
    results = {
        "execution_id": execution["execution_id"],
        "status": "DEVELOPMENT_RESULTS_OPENED",
        "development_intents": 60,
        "development_queries": 240,
        "confirmatory_intents_opened": 0,
        "confirmatory_queries_opened": 0,
        "metrics": metrics,
        "hypotheses": {"H1": h1, "H2": h2},
        "latency": latency,
        "adversarial_development": adversarial,
        "repeatability": repeatability,
        "complexity_adoption": adoption_result,
        "total_estimated_provider_cost_usd_all_a42_calls": total_cost,
        "limitations": [
            "Development evidence comes from the deterministic fictional HelixBank benchmark, not a real-bank deployment.",
            "A4.2 scores adversarial variants only for development intents; the full 77-intent registered adversarial surface remains deferred.",
            "The 68-query confirmatory assistance partition remains unopened.",
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(results, output_dir / "report.md")

    latency_expected = 60 * 3
    repeatability_expected = 30 * 3
    audit_checks: dict[str, bool] = {
        **raw_checks,
        "compatibility_provider_succeeded": bool(
            compatibility["provider_call_succeeded"]
        ),
        "compatibility_runtime_nli_finite": bool(
            compatibility["runtime_nli_probability_finite"]
        ),
        "compatibility_evaluator_nli_finite": bool(
            compatibility["evaluation_nli_probability_finite"]
        ),
        "confirmatory_queries_opened_zero": results["confirmatory_queries_opened"] == 0,
        "latency_per_candidate_180": all(
            len([row for row in latency_rows if row["candidate_id"] == candidate])
            == latency_expected
            for candidate in CANDIDATES
        ),
        "repeatability_per_candidate_90": all(
            len(
                [
                    row
                    for row in repeatability_rows
                    if row["candidate_id"] == candidate
                ]
            )
            == repeatability_expected
            for candidate in CANDIDATES
        ),
        "repeatability_subset_matches_a41": set(
            subsets["selection"]["repeatability"]["query_ids"]
        )
        == {
            str(row["query_id"])
            for row in repeatability_rows
            if row["candidate_id"] == "G0"
        },
        "latency_subset_matches_a41": set(
            subsets["selection"]["latency"]["query_ids"]
        )
        == {
            str(row["query_id"])
            for row in latency_rows
            if row["candidate_id"] == "G0"
        },
    }
    expected_attack_counts = execution["adversarial_development_counts"]
    for candidate in CANDIDATES:
        for attack_type in ATTACK_TYPES:
            audit_checks[f"{candidate}_{attack_type}_count"] = (
                adversarial[candidate][attack_type]["cases"]
                == int(expected_attack_counts[attack_type])
            )
    if not all(audit_checks.values()):
        failures = [name for name, passed in audit_checks.items() if not passed]
        raise RuntimeError(f"A4.2 post-result audit failed: {failures}")

    post_audit = {
        "audit_id": "phase4-assistance-a4.2-development-post-audit-v1",
        "execution_id": execution["execution_id"],
        "status": "PASSED_AUTOMATED_RECONSTRUCTION",
        "checks": audit_checks,
        "registered_winner": adoption_result["registered_winner"],
        "H1_verdict": h1["verdict"],
        "H2_verdict": h2["verdict"],
        "confirmatory_queries_opened": 0,
        "post_score_tuning_performed": False,
        "limitations": results["limitations"],
    }
    (output_dir / "post_audit.json").write_text(
        json.dumps(post_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "post_audit.md").write_text(
        "# A4.2 automated post-result audit\n\n"
        "**Verdict: PASSED_AUTOMATED_RECONSTRUCTION**\n\n"
        f"Registered development winner: **{adoption_result['registered_winner']}**.\n\n"
        f"H1: **{h1['verdict']}**. H2: **{h2['verdict']}**.\n\n"
        "All registered raw-count, development-partition, diagnostic-subset, "
        "compatibility, and adversarial-count checks reconstructed successfully. "
        "The confirmatory assistance partition remained unopened.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Independently reconstruct and audit Phase 3 R3.2 retrieval evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_support_intelligence.data.helixbank import generate_bundle, manifest  # noqa: E402

PROTOCOL_PATH = REPO_ROOT / "configs/models/retrieval_ladder_v1.json"
IMPLEMENTATION_PATH = REPO_ROOT / "configs/models/retrieval_implementation_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs/models/retrieval_execution_r32_v1.json"
SEED = 20260819
CANDIDATES = ("B0", "B1", "B2", "B3")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible_document_ids() -> set[str]:
    bundle = generate_bundle()
    eligible: set[str] = set()
    for row in bundle.documents:
        if row["status"] != "current":
            continue
        if row["permission"] != "public_support":
            continue
        if row["audience"] != "customer_support":
            continue
        if row["jurisdiction"] != "fictional-global":
            continue
        valid_from = str(row["valid_from"])
        valid_to = row["valid_to"]
        if valid_from > "2026-08-19":
            continue
        if valid_to is not None and str(valid_to) < "2026-08-19":
            continue
        eligible.add(str(row["document_id"]))
    return eligible


def _qrels(eligible_ids: set[str]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in generate_bundle().judgments:
        document_id = str(judgment["document_id"])
        if document_id in eligible_ids:
            rows[str(judgment["query_id"])][document_id] = int(judgment["relevance"])
    return dict(rows)


def _gain(relevance: int) -> float:
    return float((2**relevance) - 1)


def _ndcg(document_ids: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    dcg = 0.0
    for index, document_id in enumerate(document_ids[:k]):
        dcg += _gain(qrels.get(document_id, 0)) / math.log2(index + 2.0)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(_gain(value) / math.log2(index + 2.0) for index, value in enumerate(ideal))
    return 0.0 if idcg == 0.0 else dcg / idcg


def _mrr(document_ids: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    for index, document_id in enumerate(document_ids[:k]):
        if qrels.get(document_id, 0) >= 2:
            return 1.0 / (index + 1)
    return 0.0


def _recall(document_ids: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    relevant = {document_id for document_id, value in qrels.items() if value >= 2}
    if not relevant:
        return None
    return len(set(document_ids[:k]) & relevant) / len(relevant)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(candidate: Sequence[float], comparator: Sequence[float]) -> dict[str, float | int]:
    differences = [left - right for left, right in zip(candidate, comparator, strict=True)]
    rng = random.Random(SEED)
    size = len(differences)
    samples = [
        statistics.fmean(differences[rng.randrange(size)] for _ in range(size))
        for _ in range(5000)
    ]
    return {
        "point_estimate": statistics.fmean(differences),
        "lower_95": _percentile(samples, 0.025),
        "upper_95": _percentile(samples, 0.975),
        "replicates": 5000,
        "seed": SEED,
    }


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    mean_ms = statistics.fmean(samples)
    return {
        "mean_ms": mean_ms,
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "queries_per_second": 1000.0 / mean_ms,
    }


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _check_metric_block(
    expected: Mapping[str, Any], actual: Mapping[str, float | int], label: str, failures: list[str]
) -> None:
    for key in (
        "ndcg_at_10",
        "mrr_at_10",
        "recall_at_20",
        "recall_at_50",
    ):
        if not _close(float(expected[key]), float(actual[key])):
            failures.append(f"{label} {key} mismatch: {expected[key]} != {actual[key]}")
    for key in ("recall_at_20_queries", "recall_at_50_queries", "query_count"):
        if int(expected[key]) != int(actual[key]):
            failures.append(f"{label} {key} mismatch: {expected[key]} != {actual[key]}")


def _check_bootstrap(
    expected: Mapping[str, Any], actual: Mapping[str, float | int], label: str, failures: list[str]
) -> None:
    for key in ("point_estimate", "lower_95", "upper_95"):
        if not _close(float(expected[key]), float(actual[key])):
            failures.append(f"{label} {key} mismatch: {expected[key]} != {actual[key]}")
    if int(expected["replicates"]) != 5000 or int(expected["seed"]) != SEED:
        failures.append(f"{label} bootstrap settings drifted")


def _verdict(interval: Mapping[str, float | int]) -> str:
    point = float(interval["point_estimate"])
    lower = float(interval["lower_95"])
    upper = float(interval["upper_95"])
    if point > 0.0 and lower > 0.0:
        return "SUPPORTED"
    if point < 0.0 and upper < 0.0:
        return "ADVERSE"
    return "INCONCLUSIVE"


def _selection(
    metrics: Mapping[str, Mapping[str, float | int]],
    per_query: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    latency: Mapping[str, Mapping[str, float]],
    budgets: Mapping[str, float | int],
) -> tuple[list[dict[str, object]], list[str], str, str]:
    current = "B0"
    accepted: list[str] = []
    trace: list[dict[str, object]] = []
    query_ids = sorted(per_query["B0"])
    for candidate in ("B1", "B2", "B3"):
        interval = _bootstrap(
            [float(per_query[candidate][query_id]["ndcg_at_10"]) for query_id in query_ids],
            [float(per_query[current][query_id]["ndcg_at_10"]) for query_id in query_ids],
        )
        delta_mrr = float(metrics[candidate]["mrr_at_10"]) - float(metrics[current]["mrr_at_10"])
        earned = (
            float(interval["point_estimate"]) >= 0.01
            and float(interval["lower_95"]) > 0.0
            and delta_mrr >= -0.005
            and float(latency[candidate]["p95_ms"]) <= float(budgets[candidate])
        )
        trace.append(
            {
                "candidate": candidate,
                "comparator": current,
                "delta_ndcg_at_10": interval,
                "delta_mrr_at_10": delta_mrr,
                "candidate_p95_ms": latency[candidate]["p95_ms"],
                "latency_budget_ms": budgets[candidate],
                "earned_adoption": earned,
            }
        )
        if earned:
            accepted.append(candidate)
            current = candidate

    final = "B0"
    if accepted:
        best_ndcg = max(float(metrics[candidate]["ndcg_at_10"]) for candidate in accepted)
        near_best = [
            candidate
            for candidate in accepted
            if best_ndcg - float(metrics[candidate]["ndcg_at_10"]) <= 0.005
        ]
        final = min(
            near_best,
            key=lambda candidate: (float(latency[candidate]["p95_ms"]), int(candidate[1:])),
        )
    return trace, accepted, current, final


def _write_audit(output_dir: Path, payload: Mapping[str, object]) -> None:
    (output_dir / "post_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checks = payload["checks"]
    assert isinstance(checks, dict)
    lines = [
        "# Phase 3 R3.2 Post-Execution Audit",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        "The verifier independently reconstructed ranking metrics, paired-bootstrap intervals, "
        "latency summaries, hypothesis verdicts, corpus eligibility, and the complexity-selection "
        "decision from the stored execution evidence.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for key, value in checks.items():
        lines.append(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |")
    failures = payload["failures"]
    assert isinstance(failures, list)
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in failures)
    lines.append("")
    (output_dir / "post_audit.md").write_text("\n".join(lines), encoding="utf-8")

    evidence_files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_lines = [f"{_sha256_file(path)}  {path.name}" for path in evidence_files]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def verify(output_dir: Path) -> dict[str, object]:
    failures: list[str] = []
    protocol = _read_json(PROTOCOL_PATH)
    implementation = _read_json(IMPLEMENTATION_PATH)
    execution = _read_json(EXECUTION_PATH)
    results = _read_json(output_dir / "results.json")
    execution_manifest = _read_json(output_dir / "execution_manifest.json")
    rankings = _read_jsonl(output_dir / "rankings.jsonl")
    recorded_query_metrics = _read_jsonl(output_dir / "query_metrics.jsonl")
    latency_rows = _read_jsonl(output_dir / "latency_samples.jsonl")

    current_manifest = manifest()
    corpus_ok = (
        current_manifest["counts"] == protocol["corpus"]["counts"]
        and current_manifest["sha256"] == protocol["corpus"]["sha256"]
    )
    if not corpus_ok:
        failures.append("frozen corpus manifest mismatch")

    eligible_ids = _eligible_document_ids()
    eligibility_ok = len(eligible_ids) == 147
    if not eligibility_ok:
        failures.append(f"eligible document count is {len(eligible_ids)}, expected 147")
    qrels = _qrels(eligible_ids)
    query_ids = sorted(str(row["query_id"]) for row in generate_bundle().queries)

    ranking_map: dict[str, dict[str, list[str]]] = {candidate: {} for candidate in CANDIDATES}
    ranking_shape_ok = len(rankings) == 308 * 4
    for row in rankings:
        query_id = str(row["query_id"])
        candidate = str(row["candidate"])
        if candidate not in ranking_map or query_id not in query_ids:
            ranking_shape_ok = False
            continue
        items = row["results"]
        if not isinstance(items, list) or len(items) != 50:
            ranking_shape_ok = False
            continue
        document_ids = [str(item["document_id"]) for item in items]
        ranks = [int(item["rank"]) for item in items]
        if ranks != list(range(1, 51)) or len(set(document_ids)) != 50:
            ranking_shape_ok = False
        if not set(document_ids) <= eligible_ids:
            ranking_shape_ok = False
        ranking_map[candidate][query_id] = document_ids
    if any(len(ranking_map[candidate]) != 308 for candidate in CANDIDATES):
        ranking_shape_ok = False
    if not ranking_shape_ok:
        failures.append("ranking evidence shape, rank continuity, uniqueness, or eligibility failed")

    per_query: dict[str, dict[str, dict[str, float | None]]] = {
        candidate: {} for candidate in CANDIDATES
    }
    aggregate: dict[str, dict[str, float | int]] = {}
    for candidate in CANDIDATES:
        ndcg_values: list[float] = []
        mrr_values: list[float] = []
        recall20_values: list[float] = []
        recall50_values: list[float] = []
        for query_id in query_ids:
            document_ids = ranking_map[candidate][query_id]
            query_qrels = qrels.get(query_id, {})
            ndcg = _ndcg(document_ids, query_qrels, 10)
            mrr = _mrr(document_ids, query_qrels, 10)
            recall20 = _recall(document_ids, query_qrels, 20)
            recall50 = _recall(document_ids, query_qrels, 50)
            per_query[candidate][query_id] = {
                "ndcg_at_10": ndcg,
                "mrr_at_10": mrr,
                "recall_at_20": recall20,
                "recall_at_50": recall50,
            }
            ndcg_values.append(ndcg)
            mrr_values.append(mrr)
            if recall20 is not None:
                recall20_values.append(recall20)
            if recall50 is not None:
                recall50_values.append(recall50)
        aggregate[candidate] = {
            "ndcg_at_10": statistics.fmean(ndcg_values),
            "mrr_at_10": statistics.fmean(mrr_values),
            "recall_at_20": statistics.fmean(recall20_values) if recall20_values else 0.0,
            "recall_at_20_queries": len(recall20_values),
            "recall_at_50": statistics.fmean(recall50_values) if recall50_values else 0.0,
            "recall_at_50_queries": len(recall50_values),
            "query_count": len(query_ids),
        }

    metric_reconstruction_ok = True
    result_metrics = results["metrics"]
    for candidate in CANDIDATES:
        before = len(failures)
        _check_metric_block(result_metrics[candidate], aggregate[candidate], candidate, failures)
        metric_reconstruction_ok = metric_reconstruction_ok and len(failures) == before

    query_metric_map = {
        (str(row["candidate"]), str(row["query_id"])): row for row in recorded_query_metrics
    }
    per_query_evidence_ok = len(query_metric_map) == 308 * 4
    for candidate in CANDIDATES:
        for query_id in query_ids:
            recorded = query_metric_map.get((candidate, query_id))
            if recorded is None:
                per_query_evidence_ok = False
                continue
            reconstructed = per_query[candidate][query_id]
            for key in ("ndcg_at_10", "mrr_at_10"):
                if not _close(float(recorded[key]), float(reconstructed[key])):
                    per_query_evidence_ok = False
            for key in ("recall_at_20", "recall_at_50"):
                left = recorded[key]
                right = reconstructed[key]
                if left is None or right is None:
                    if left is not None or right is not None:
                        per_query_evidence_ok = False
                elif not _close(float(left), float(right)):
                    per_query_evidence_ok = False
    if not per_query_evidence_ok:
        failures.append("stored per-query metrics do not match reconstructed rankings")

    h1 = _bootstrap(
        [float(per_query["B2"][query_id]["ndcg_at_10"]) for query_id in query_ids],
        [float(per_query["B0"][query_id]["ndcg_at_10"]) for query_id in query_ids],
    )
    h2 = _bootstrap(
        [float(per_query["B3"][query_id]["mrr_at_10"]) for query_id in query_ids],
        [float(per_query["B2"][query_id]["mrr_at_10"]) for query_id in query_ids],
    )
    bootstrap_ok = True
    before = len(failures)
    _check_bootstrap(results["hypotheses"]["H1"]["difference"], h1, "H1", failures)
    _check_bootstrap(results["hypotheses"]["H2"]["difference"], h2, "H2", failures)
    bootstrap_ok = len(failures) == before

    hypothesis_verdict_ok = (
        results["hypotheses"]["H1"]["verdict"] == _verdict(h1)
        and results["hypotheses"]["H2"]["verdict"] == _verdict(h2)
    )
    if not hypothesis_verdict_ok:
        failures.append("registered hypothesis verdict does not match reconstructed interval")

    grouped_latency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latency_rows:
        grouped_latency[str(row["candidate"])].append(row)
    latency_ok = True
    reconstructed_latency: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        rows = grouped_latency[candidate]
        if len(rows) != 308 * 5:
            latency_ok = False
            continue
        for pass_number in range(1, 6):
            pass_rows = [row for row in rows if int(row["pass"]) == pass_number]
            if [str(row["query_id"]) for row in pass_rows] != query_ids:
                latency_ok = False
        samples = [float(row["latency_ms"]) for row in rows]
        if any(value <= 0.0 for value in samples):
            latency_ok = False
        reconstructed_latency[candidate] = _latency_summary(samples)
        for key, value in reconstructed_latency[candidate].items():
            if not _close(float(results["latency"][candidate][key]), value):
                latency_ok = False
    if not latency_ok:
        failures.append("latency sample count, order, positivity, or summary reconstruction failed")

    trace, accepted, sequential, final = _selection(
        aggregate,
        per_query,
        reconstructed_latency,
        execution["latency_execution"]["budgets_ms"],
    )
    selection = results["complexity_selection"]
    selection_ok = (
        selection["accepted_candidates"] == accepted
        and selection["sequential_winner"] == sequential
        and selection["final_winner"] == final
        and len(selection["trace"]) == len(trace)
    )
    if selection_ok:
        for recorded, reconstructed in zip(selection["trace"], trace, strict=True):
            if recorded["candidate"] != reconstructed["candidate"]:
                selection_ok = False
            if recorded["comparator"] != reconstructed["comparator"]:
                selection_ok = False
            if bool(recorded["earned_adoption"]) != bool(reconstructed["earned_adoption"]):
                selection_ok = False
            for key in ("delta_mrr_at_10", "candidate_p95_ms", "latency_budget_ms"):
                if not _close(float(recorded[key]), float(reconstructed[key])):
                    selection_ok = False
            recorded_interval = recorded["delta_ndcg_at_10"]
            reconstructed_interval = reconstructed["delta_ndcg_at_10"]
            for key in ("point_estimate", "lower_95", "upper_95"):
                if not _close(float(recorded_interval[key]), float(reconstructed_interval[key])):
                    selection_ok = False
    if not selection_ok:
        failures.append("complexity-selection decision does not reconstruct exactly")

    model_pin_ok = (
        results["resolved_model_revisions"]["B1"] == protocol["ladder"][1]["model"]["revision"]
        and results["resolved_model_revisions"]["B3"] == protocol["ladder"][3]["model"]["revision"]
        and execution_manifest["resolved_model_revisions"] == results["resolved_model_revisions"]
    )
    if not model_pin_ok:
        failures.append("runtime model revision evidence does not match frozen protocol")

    input_hash_ok = True
    for relative, expected_hash in execution_manifest["input_sha256"].items():
        path = REPO_ROOT / str(relative)
        if not path.exists() or _sha256_file(path) != expected_hash:
            input_hash_ok = False
    if not input_hash_ok:
        failures.append("execution input file hash mismatch")

    status_ok = results["status"] == "PROVISIONAL_PENDING_POST_AUDIT"
    if not status_ok:
        failures.append("execution result was not correctly marked provisional")

    checks = {
        "corpus_manifest": corpus_ok,
        "eligibility": eligibility_ok,
        "ranking_evidence_shape": ranking_shape_ok,
        "aggregate_metric_reconstruction": metric_reconstruction_ok,
        "per_query_metric_reconstruction": per_query_evidence_ok,
        "bootstrap_reconstruction": bootstrap_ok,
        "hypothesis_verdicts": hypothesis_verdict_ok,
        "latency_reconstruction": latency_ok,
        "complexity_selection": selection_ok,
        "model_revision_pins": model_pin_ok,
        "execution_input_hashes": input_hash_ok,
        "provisional_status_before_audit": status_ok,
    }
    passed = not failures and all(checks.values())
    audit: dict[str, object] = {
        "execution_id": execution["execution_id"],
        "protocol_id": protocol["protocol_id"],
        "implementation_id": implementation["implementation_id"],
        "verdict": "PASSED" if passed else "FAILED",
        "checks": checks,
        "failures": failures,
        "reconstructed": {
            "metrics": aggregate,
            "H1": h1,
            "H1_verdict": _verdict(h1),
            "H2": h2,
            "H2_verdict": _verdict(h2),
            "latency": reconstructed_latency,
            "accepted_candidates": accepted,
            "sequential_winner": sequential,
            "final_winner": final,
        },
    }
    _write_audit(output_dir, audit)
    if not passed:
        raise SystemExit("R3.2 post-execution audit failed; inspect post_audit.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = verify(args.output_dir)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

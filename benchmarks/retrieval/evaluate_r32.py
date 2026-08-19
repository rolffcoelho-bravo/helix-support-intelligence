# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5",
#   "sentence-transformers==5.5.1",
#   "torch==2.13.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Execute the registered Phase 3 R3.2 retrieval benchmark exactly once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
from huggingface_hub import model_info
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_support_intelligence.data.helixbank import (  # noqa: E402
    generate_bundle,
)
from helix_support_intelligence.data.helixbank import (  # noqa: E402
    manifest as corpus_manifest,
)
from helix_support_intelligence.retrieval.core import (  # noqa: E402
    EligibilityPolicy,
    candidate_earns_complexity,
    document_from_record,
    filter_eligible_documents,
    paired_bootstrap_difference,
    summarize_latency,
)
from helix_support_intelligence.retrieval.evaluation import (  # noqa: E402
    QueryMetrics,
    aggregate_metrics,
    evaluate_ranking,
)
from helix_support_intelligence.retrieval.ladder import (  # noqa: E402
    CandidateId,
    RetrievalLadder,
)

PROTOCOL_PATH = REPO_ROOT / "configs/models/retrieval_ladder_v1.json"
IMPLEMENTATION_PATH = REPO_ROOT / "configs/models/retrieval_implementation_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs/models/retrieval_execution_r32_v1.json"
SEED = 20260819


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_line(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(_canonical_json_line(row) + "\n" for row in rows), encoding="utf-8")


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


class SentenceTransformerAdapter:
    """Concrete R3.2 adapter for the pinned B1 encoder."""

    def __init__(self, model: SentenceTransformer, batch_size: int) -> None:
        self._model = model
        self._batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        values = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(values, dtype=np.float64)


class CrossEncoderAdapter:
    """Concrete R3.2 adapter for the pinned B3 reranker."""

    def __init__(self, model: CrossEncoder, batch_size: int) -> None:
        self._model = model
        self._batch_size = batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        if not pairs:
            return ()
        values = self._model.predict(
            list(pairs),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float64).reshape(-1).tolist()


def _verify_registration(
    protocol: Mapping[str, Any], implementation: Mapping[str, Any], execution: Mapping[str, Any]
) -> None:
    if protocol["protocol_id"] != execution["protocol_id"]:
        raise ValueError("protocol id drifted")
    if implementation["implementation_id"] != execution["implementation_id"]:
        raise ValueError("implementation id drifted")
    if protocol["execution_guard"]["results_opened"] is not False:
        raise ValueError("R3.0 protocol no longer represents its pre-evaluation freeze")
    if execution["results_opened"] is not False:
        raise ValueError("R3.2 execution plan was altered after registration")
    if execution["execution_partition"]["queries"] != 308:
        raise ValueError("registered query count drifted")
    if execution["execution_partition"]["candidate_order"] != ["B0", "B1", "B2", "B3"]:
        raise ValueError("candidate order drifted")

    ladder = {row["id"]: row for row in protocol["ladder"]}
    b1 = execution["models"]["B1"]
    b3 = execution["models"]["B3"]
    if b1["model_id"] != ladder["B1"]["model"]["id"]:
        raise ValueError("B1 model id drifted")
    if b1["revision"] != ladder["B1"]["model"]["revision"]:
        raise ValueError("B1 revision drifted")
    if b3["model_id"] != ladder["B3"]["model"]["id"]:
        raise ValueError("B3 model id drifted")
    if b3["revision"] != ladder["B3"]["model"]["revision"]:
        raise ValueError("B3 revision drifted")


def _verify_model_revision(model_id: str, revision: str) -> str:
    resolved = str(model_info(model_id, revision=revision).sha)
    if resolved != revision:
        raise ValueError(f"model revision mismatch for {model_id}: {resolved}")
    return resolved


def _build_qrels(
    judgments: Sequence[Mapping[str, object]], eligible_ids: set[str]
) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in judgments:
        document_id = str(row["document_id"])
        if document_id in eligible_ids:
            qrels[str(row["query_id"])][document_id] = int(row["relevance"])
    return dict(qrels)


def _metric_payload(row: QueryMetrics) -> dict[str, float | None]:
    return {
        "ndcg_at_10": row.ndcg_at_10,
        "mrr_at_10": row.mrr_at_10,
        "recall_at_20": row.recall_at_20,
        "recall_at_50": row.recall_at_50,
    }


def _aggregate_payload(rows: Sequence[QueryMetrics]) -> dict[str, float | int]:
    value = aggregate_metrics(rows)
    return {
        "ndcg_at_10": value.ndcg_at_10,
        "mrr_at_10": value.mrr_at_10,
        "recall_at_20": value.recall_at_20,
        "recall_at_20_queries": value.recall_at_20_queries,
        "recall_at_50": value.recall_at_50,
        "recall_at_50_queries": value.recall_at_50_queries,
        "query_count": value.query_count,
    }


def _bootstrap_payload(interval: Any) -> dict[str, float | int]:
    return {
        "point_estimate": interval.point_estimate,
        "lower_95": interval.lower,
        "upper_95": interval.upper,
        "replicates": interval.replicates,
        "seed": interval.seed,
    }


def _verdict(point: float, lower: float, upper: float) -> str:
    if point > 0.0 and lower > 0.0:
        return "SUPPORTED"
    if point < 0.0 and upper < 0.0:
        return "ADVERSE"
    return "INCONCLUSIVE"


def _diagnostic_slices(
    query_rows: Sequence[Mapping[str, object]],
    metrics: Mapping[CandidateId, Mapping[str, QueryMetrics]],
) -> dict[str, object]:
    by_case: dict[str, list[str]] = defaultdict(list)
    for query in query_rows:
        by_case[str(query["case_type"])].append(str(query["query_id"]))

    result: dict[str, object] = {"case_type": {}}
    case_payload = result["case_type"]
    assert isinstance(case_payload, dict)
    for case_type, query_ids in sorted(by_case.items()):
        case_payload[case_type] = {
            candidate: _aggregate_payload([metrics[candidate][query_id] for query_id in query_ids])
            for candidate in ("B0", "B1", "B2", "B3")
        }
    return result


def _environment_payload(resolved_models: Mapping[str, str]) -> dict[str, object]:
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "resolved_model_revisions": dict(resolved_models),
        "installed_packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def _report(result: Mapping[str, Any]) -> str:
    metrics = result["metrics"]
    latency = result["latency"]
    hypotheses = result["hypotheses"]
    selection = result["complexity_selection"]
    lines = [
        "# Phase 3 R3.2 Registered Retrieval Execution",
        "",
        "> Provisional until the independent post-execution reconstruction passes.",
        "",
        "| Candidate | nDCG@10 | MRR@10 | Recall@20 | Recall@50 | P95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in ("B0", "B1", "B2", "B3"):
        row = metrics[candidate]
        lat = latency[candidate]
        lines.append(
            f"| {candidate} | {row['ndcg_at_10']:.4f} | {row['mrr_at_10']:.4f} | "
            f"{row['recall_at_20']:.4f} | {row['recall_at_50']:.4f} | {lat['p95_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Registered hypotheses",
            "",
            f"- H1 B2 - B0 nDCG@10: **{hypotheses['H1']['verdict']}**, "
            f"delta {hypotheses['H1']['difference']['point_estimate']:+.4f}, "
            f"95% CI [{hypotheses['H1']['difference']['lower_95']:+.4f}, "
            f"{hypotheses['H1']['difference']['upper_95']:+.4f}]",
            f"- H2 B3 - B2 MRR@10: **{hypotheses['H2']['verdict']}**, "
            f"delta {hypotheses['H2']['difference']['point_estimate']:+.4f}, "
            f"95% CI [{hypotheses['H2']['difference']['lower_95']:+.4f}, "
            f"{hypotheses['H2']['difference']['upper_95']:+.4f}]",
            "",
            "## Complexity selection",
            "",
            f"Registered winner before independent audit: **{selection['final_winner']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _read_json(PROTOCOL_PATH)
    implementation = _read_json(IMPLEMENTATION_PATH)
    execution = _read_json(EXECUTION_PATH)
    _verify_registration(protocol, implementation, execution)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(int(execution["runtime"]["torch_num_threads"]))
    torch.set_num_interop_threads(int(execution["runtime"]["torch_num_interop_threads"]))
    torch.use_deterministic_algorithms(True)

    current_manifest = corpus_manifest()
    frozen_corpus = protocol["corpus"]
    if current_manifest["counts"] != frozen_corpus["counts"]:
        raise ValueError("corpus counts drifted")
    if current_manifest["sha256"] != frozen_corpus["sha256"]:
        raise ValueError("corpus hashes drifted")

    bundle = generate_bundle()
    documents = tuple(document_from_record(row) for row in bundle.documents)
    filters = frozen_corpus["eligibility_filters"]
    policy = EligibilityPolicy(
        evaluation_date=date.fromisoformat(str(frozen_corpus["evaluation_date"])),
        statuses=frozenset(str(value) for value in filters["status"]),
        permissions=frozenset(str(value) for value in filters["permission"]),
        audiences=frozenset(str(value) for value in filters["audience"]),
        jurisdictions=frozenset(str(value) for value in filters["jurisdiction"]),
    )
    eligible = filter_eligible_documents(documents, policy)
    if len(eligible) != int(execution["execution_partition"]["eligible_documents_expected"]):
        raise ValueError("eligible document count drifted")
    eligible_ids = {document.document_id for document in eligible}

    queries = sorted(bundle.queries, key=lambda row: str(row["query_id"]))
    if len(queries) != int(execution["execution_partition"]["queries"]):
        raise ValueError("query count drifted")
    qrels = _build_qrels(bundle.judgments, eligible_ids)

    b1_config = execution["models"]["B1"]
    b3_config = execution["models"]["B3"]
    resolved_models = {
        "B1": _verify_model_revision(str(b1_config["model_id"]), str(b1_config["revision"])),
        "B3": _verify_model_revision(str(b3_config["model_id"]), str(b3_config["revision"])),
    }

    encoder_model = SentenceTransformer(
        str(b1_config["model_id"]),
        revision=str(b1_config["revision"]),
        device="cpu",
        trust_remote_code=False,
    )
    encoder_model.max_seq_length = int(b1_config["max_sequence_length"])
    reranker_model = CrossEncoder(
        str(b3_config["model_id"]),
        revision=str(b3_config["revision"]),
        device="cpu",
        max_length=int(b3_config["max_sequence_length"]),
        trust_remote_code=False,
    )

    ladder = RetrievalLadder(
        eligible,
        SentenceTransformerAdapter(encoder_model, int(b1_config["batch_size"])),
        CrossEncoderAdapter(reranker_model, int(b3_config["batch_size"])),
    )

    ranking_rows: list[dict[str, object]] = []
    query_metric_rows: list[dict[str, object]] = []
    per_candidate: dict[CandidateId, dict[str, QueryMetrics]] = {
        "B0": {},
        "B1": {},
        "B2": {},
        "B3": {},
    }

    for query in queries:
        query_id = str(query["query_id"])
        query_text = str(query["text"])
        rankings = ladder.rank_all(query_text)
        for candidate in ("B0", "B1", "B2", "B3"):
            ranking = rankings[candidate]
            if len(ranking) != 50:
                raise ValueError(f"{candidate} returned {len(ranking)} documents")
            metrics = evaluate_ranking(ranking, qrels.get(query_id, {}))
            per_candidate[candidate][query_id] = metrics
            ranking_rows.append(
                {
                    "query_id": query_id,
                    "candidate": candidate,
                    "results": [
                        {"document_id": item.document_id, "rank": item.rank, "score": item.score}
                        for item in ranking
                    ],
                }
            )
            query_metric_rows.append(
                {
                    "query_id": query_id,
                    "candidate": candidate,
                    **_metric_payload(metrics),
                }
            )

    aggregate = {
        candidate: _aggregate_payload(
            [per_candidate[candidate][str(q["query_id"])] for q in queries]
        )
        for candidate in ("B0", "B1", "B2", "B3")
    }

    h1 = paired_bootstrap_difference(
        [per_candidate["B2"][str(q["query_id"])].ndcg_at_10 for q in queries],
        [per_candidate["B0"][str(q["query_id"])].ndcg_at_10 for q in queries],
        replicates=5000,
        seed=SEED,
    )
    h2 = paired_bootstrap_difference(
        [per_candidate["B3"][str(q["query_id"])].mrr_at_10 for q in queries],
        [per_candidate["B2"][str(q["query_id"])].mrr_at_10 for q in queries],
        replicates=5000,
        seed=SEED,
    )

    latency_samples: list[dict[str, object]] = []
    latency_payload: dict[str, dict[str, float]] = {}
    latency_config = execution["latency_execution"]
    warmup_count = int(latency_config["warmup_queries_per_candidate"])
    timed_passes = int(latency_config["timed_passes"])
    for candidate in ("B0", "B1", "B2", "B3"):
        for query in queries[:warmup_count]:
            ladder.rank(candidate, str(query["text"]))
        samples: list[float] = []
        for pass_index in range(timed_passes):
            for query in queries:
                started = time.perf_counter_ns()
                ladder.rank(candidate, str(query["text"]))
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                samples.append(elapsed_ms)
                latency_samples.append(
                    {
                        "candidate": candidate,
                        "pass": pass_index + 1,
                        "query_id": str(query["query_id"]),
                        "latency_ms": elapsed_ms,
                    }
                )
        summary = summarize_latency(samples)
        latency_payload[candidate] = {
            "mean_ms": summary.mean_ms,
            "p50_ms": summary.p50_ms,
            "p95_ms": summary.p95_ms,
            "p99_ms": summary.p99_ms,
            "queries_per_second": summary.queries_per_second,
        }

    current_winner: CandidateId = "B0"
    accepted: list[CandidateId] = []
    selection_trace: list[dict[str, object]] = []
    budgets = latency_config["budgets_ms"]
    for candidate in ("B1", "B2", "B3"):
        candidate_ndcg = [per_candidate[candidate][str(q["query_id"])].ndcg_at_10 for q in queries]
        winner_ndcg = [
            per_candidate[current_winner][str(q["query_id"])].ndcg_at_10 for q in queries
        ]
        ndcg_interval = paired_bootstrap_difference(
            candidate_ndcg,
            winner_ndcg,
            replicates=5000,
            seed=SEED,
        )
        delta_mrr = float(aggregate[candidate]["mrr_at_10"]) - float(
            aggregate[current_winner]["mrr_at_10"]
        )
        earned = candidate_earns_complexity(
            delta_ndcg_at_10=ndcg_interval.point_estimate,
            ndcg_ci_lower=ndcg_interval.lower,
            delta_mrr_at_10=delta_mrr,
            candidate_p95_ms=float(latency_payload[candidate]["p95_ms"]),
            latency_budget_ms=float(budgets[candidate]),
        )
        selection_trace.append(
            {
                "candidate": candidate,
                "comparator": current_winner,
                "delta_ndcg_at_10": _bootstrap_payload(ndcg_interval),
                "delta_mrr_at_10": delta_mrr,
                "candidate_p95_ms": latency_payload[candidate]["p95_ms"],
                "latency_budget_ms": budgets[candidate],
                "earned_adoption": earned,
            }
        )
        if earned:
            accepted.append(candidate)
            current_winner = candidate

    final_winner: CandidateId = "B0"
    if accepted:
        best_ndcg = max(float(aggregate[candidate]["ndcg_at_10"]) for candidate in accepted)
        near_best = [
            candidate
            for candidate in accepted
            if best_ndcg - float(aggregate[candidate]["ndcg_at_10"]) <= 0.005
        ]
        final_winner = min(
            near_best,
            key=lambda candidate: (float(latency_payload[candidate]["p95_ms"]), int(candidate[1:])),
        )

    result: dict[str, object] = {
        "execution_id": execution["execution_id"],
        "protocol_id": protocol["protocol_id"],
        "implementation_id": implementation["implementation_id"],
        "status": "PROVISIONAL_PENDING_POST_AUDIT",
        "query_count": len(queries),
        "eligible_document_count": len(eligible),
        "metrics": aggregate,
        "diagnostic_slices": _diagnostic_slices(queries, per_candidate),
        "hypotheses": {
            "H1": {
                "comparison": "B2 - B0",
                "endpoint": "nDCG@10",
                "difference": _bootstrap_payload(h1),
                "verdict": _verdict(h1.point_estimate, h1.lower, h1.upper),
            },
            "H2": {
                "comparison": "B3 - B2",
                "endpoint": "MRR@10",
                "difference": _bootstrap_payload(h2),
                "verdict": _verdict(h2.point_estimate, h2.lower, h2.upper),
            },
        },
        "latency": latency_payload,
        "complexity_selection": {
            "trace": selection_trace,
            "accepted_candidates": accepted,
            "sequential_winner": current_winner,
            "final_winner": final_winner,
        },
        "resolved_model_revisions": resolved_models,
    }

    _write_jsonl(output_dir / "rankings.jsonl", ranking_rows)
    _write_jsonl(output_dir / "query_metrics.jsonl", query_metric_rows)
    _write_jsonl(output_dir / "latency_samples.jsonl", latency_samples)
    _write_json(output_dir / "environment.json", _environment_payload(resolved_models))
    _write_json(output_dir / "results.json", result)
    (output_dir / "report.md").write_text(_report(result), encoding="utf-8")

    input_paths = [
        PROTOCOL_PATH,
        IMPLEMENTATION_PATH,
        EXECUTION_PATH,
        Path(__file__).resolve(),
        REPO_ROOT / "benchmarks/retrieval/verify_r32.py",
    ]
    execution_manifest = {
        "execution_id": execution["execution_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "started_from_registered_results_opened_false": True,
        "corpus_manifest": current_manifest,
        "input_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path) for path in input_paths
        },
        "resolved_model_revisions": resolved_models,
        "result_files_before_post_audit": [
            "results.json",
            "report.md",
            "rankings.jsonl",
            "query_metrics.jsonl",
            "latency_samples.jsonl",
            "environment.json",
        ],
    }
    _write_json(output_dir / "execution_manifest.json", execution_manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Independently reconstruct A4.4d validation metrics from immutable raw logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from validation_cases_a44d import generate_validation_cases  # noqa: E402

from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

A44D_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44d_v1.json"
RELATION_TO_LABEL = {"CONTRADICTED": 0, "UNKNOWN": 1, "ENTAILED": 2}
LABEL_TO_RELATION = {value: key for key, value in RELATION_TO_LABEL.items()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path} contains a non-object row.")
        rows.append(value)
    return rows


def _gold_relation(atom: dict[str, Any], document_id: str) -> str:
    if document_id in {str(value) for value in atom.get("entailed_by", [])}:
        return "ENTAILED"
    if document_id in {str(value) for value in atom.get("contradicted_by", [])}:
        return "CONTRADICTED"
    return "UNKNOWN"


def _expected_pairs(
    cases: list[dict[str, Any]], documents: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        presented = {str(value) for value in case["presented_document_ids"]}
        cited = {str(value) for value in case["cited_document_ids"]}
        if not cited or not cited.issubset(presented):
            continue
        for atom in case["atoms"]:
            atom_id = str(atom["atom_id"])
            for document_id in sorted(cited):
                if document_id not in documents:
                    raise RuntimeError(f"Missing frozen document: {document_id}")
                rows.append(
                    {
                        "pair_id": f"{case['case_id']}::{atom_id}::{document_id}",
                        "case_id": str(case["case_id"]),
                        "atom_id": atom_id,
                        "document_id": document_id,
                        "gold_relation": _gold_relation(atom, document_id),
                    }
                )
    return rows


def _argmax(values: list[float]) -> int:
    if len(values) != 3:
        raise RuntimeError("A4.4d raw logits must have exactly three classes.")
    return max(range(3), key=lambda index: values[index])


def _recall(gold: list[int], predicted: list[int], label: int) -> float:
    indices = [index for index, value in enumerate(gold) if value == label]
    if not indices:
        return 0.0
    return sum(predicted[index] == label for index in indices) / len(indices)


def _f1(gold: list[int], predicted: list[int], label: int) -> float:
    tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
    fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
    fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
    denominator = (2 * tp) + fp + fn
    return (2 * tp) / denominator if denominator else 0.0


def _predict_case_verdict(
    case: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    predicted_relations: dict[str, str],
) -> str:
    presented = {str(value) for value in case["presented_document_ids"]}
    cited = {str(value) for value in case["cited_document_ids"]}
    if not cited or not cited.issubset(presented):
        return "CITATION_INVALID"
    if any(document_id not in documents for document_id in cited):
        return "CITATION_INVALID"
    if bool(case.get("requires_current_evidence", True)) and any(
        str(documents[document_id].get("status")) == "archived" for document_id in cited
    ):
        return "STALE_EVIDENCE"
    if any(
        bool(documents[document_id].get("conflict_fixture"))
        for document_id in presented
        if document_id in documents
    ):
        return "CONFLICTING_EVIDENCE"

    conflict = False
    supported = True
    for atom in case["atoms"]:
        atom_id = str(atom["atom_id"])
        relations = {
            predicted_relations[f"{case['case_id']}::{atom_id}::{document_id}"]
            for document_id in sorted(cited)
        }
        if "ENTAILED" in relations and "CONTRADICTED" in relations:
            conflict = True
        if "ENTAILED" not in relations:
            supported = False
    if conflict:
        return "CONFLICTING_EVIDENCE"
    if not supported:
        return "UNSUPPORTED"
    return "SUPPORTED"


def _metrics(pair_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [int(row["gold_label"]) for row in pair_rows]
    predicted = [int(row["raw_argmax_label"]) for row in pair_rows]
    category_accuracy: dict[str, float] = {}
    categories = sorted({str(row["category"]) for row in case_rows})
    for category in categories:
        rows = [row for row in case_rows if row["category"] == category]
        category_accuracy[category] = sum(
            row["predicted_verdict"] == row["gold_verdict"] for row in rows
        ) / len(rows)

    gold_supported = [row for row in case_rows if row["gold_verdict"] == "SUPPORTED"]
    predicted_supported = [row for row in case_rows if row["predicted_verdict"] == "SUPPORTED"]
    true_supported = [
        row
        for row in case_rows
        if row["gold_verdict"] == "SUPPORTED" and row["predicted_verdict"] == "SUPPORTED"
    ]
    safety_categories = {
        "citation_invalid",
        "stale_current_evidence",
        "unresolved_conflict",
    }
    macro_f1 = sum(_f1(gold, predicted, label) for label in range(3)) / 3
    return {
        "atomic_relation_macro_f1": macro_f1,
        "atomic_entailment_recall": _recall(gold, predicted, RELATION_TO_LABEL["ENTAILED"]),
        "atomic_contradiction_recall": _recall(gold, predicted, RELATION_TO_LABEL["CONTRADICTED"]),
        "atomic_unknown_recall": _recall(gold, predicted, RELATION_TO_LABEL["UNKNOWN"]),
        "macro_case_category_accuracy": sum(category_accuracy.values()) / len(category_accuracy),
        "supported_precision": len(true_supported) / len(predicted_supported)
        if predicted_supported
        else 0.0,
        "supported_recall": len(true_supported) / len(gold_supported) if gold_supported else 0.0,
        "literal_supported_recall": category_accuracy["literal_supported"],
        "paraphrase_supported_recall": category_accuracy["paraphrase_supported"],
        "contradiction_unsupported_accuracy": category_accuracy["contradiction_unsupported"],
        "unsupported_approval_accuracy": category_accuracy["unsupported_approval"],
        "multi_document_supported_recall": category_accuracy["multi_document_supported"],
        "partial_multi_document_unsupported_accuracy": category_accuracy[
            "partial_multi_document_unsupported"
        ],
        "citation_invalid_accuracy": category_accuracy["citation_invalid"],
        "stale_current_evidence_accuracy": category_accuracy["stale_current_evidence"],
        "unresolved_conflict_accuracy": category_accuracy["unresolved_conflict"],
        "false_supported_on_citation_stale_or_conflict_cases": sum(
            row["predicted_verdict"] == "SUPPORTED" and row["category"] in safety_categories
            for row in case_rows
        ),
        "category_accuracy": category_accuracy,
    }


def _requirement_checks(metrics: dict[str, Any], requirements: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, value in requirements.items():
        if name == "all_requirements_must_pass":
            continue
        if name.endswith("_min"):
            checks[name] = float(metrics[name[: -len("_min")]]) >= float(value)
        else:
            checks[name] = metrics[name] == value
    return checks


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir

    config = _load(A44D_CONFIG_PATH)
    result = _load(output_dir / "results.json")
    pair_rows = _read_jsonl(output_dir / "validation_pair_logits.jsonl")
    stored_case_rows = _read_jsonl(output_dir / "validation_case_results.jsonl")

    bundle = generate_bundle()
    cases = generate_validation_cases(bundle)
    documents = {str(row["document_id"]): dict(row) for row in bundle.documents}
    expected_pairs = _expected_pairs(cases, documents)
    if len(expected_pairs) != 246 or len(pair_rows) != 246:
        raise RuntimeError("A4.4d independent pair count mismatch.")
    if len(cases) != 144 or len(stored_case_rows) != 144:
        raise RuntimeError("A4.4d independent case count mismatch.")
    if any(row["split"] != "validation" for row in pair_rows + stored_case_rows):
        raise RuntimeError("A4.4d artifact contains a non-validation scored row.")

    expected_by_id = {row["pair_id"]: row for row in expected_pairs}
    if len(expected_by_id) != len(expected_pairs):
        raise RuntimeError("A4.4d expected pair ids are not unique.")
    if {str(row["pair_id"]) for row in pair_rows} != set(expected_by_id):
        raise RuntimeError("A4.4d scored pair ids differ from registered validation pairs.")

    predicted_relations: dict[str, str] = {}
    for row in pair_rows:
        pair_id = str(row["pair_id"])
        expected = expected_by_id[pair_id]
        if row["gold_relation"] != expected["gold_relation"]:
            raise RuntimeError(f"Gold relation drift for {pair_id}.")
        gold_label = RELATION_TO_LABEL[str(row["gold_relation"])]
        if int(row["gold_label"]) != gold_label:
            raise RuntimeError(f"Gold label drift for {pair_id}.")
        logits = [float(value) for value in row["logits"]]
        raw_label = _argmax(logits)
        if int(row["raw_argmax_label"]) != raw_label:
            raise RuntimeError(f"Raw argmax drift for {pair_id}.")
        relation = LABEL_TO_RELATION[raw_label]
        if row["raw_argmax_relation"] != relation:
            raise RuntimeError(f"Raw relation drift for {pair_id}.")
        if bool(row["raw_correct"]) != (raw_label == gold_label):
            raise RuntimeError(f"Raw correctness drift for {pair_id}.")
        calibrated_label = _argmax([value / 3.67 for value in logits])
        if calibrated_label != raw_label:
            raise RuntimeError(f"Frozen temperature changed argmax for {pair_id}.")
        predicted_relations[pair_id] = relation

    recomputed_case_rows: list[dict[str, Any]] = []
    for case in cases:
        predicted = _predict_case_verdict(case, documents, predicted_relations)
        recomputed_case_rows.append(
            {
                "case_id": str(case["case_id"]),
                "intent": str(case["intent"]),
                "category": str(case["category"]),
                "split": "validation",
                "gold_verdict": str(case["expected_verdict"]),
                "predicted_verdict": predicted,
                "correct": predicted == str(case["expected_verdict"]),
            }
        )
    stored_by_id = {str(row["case_id"]): row for row in stored_case_rows}
    recomputed_by_id = {str(row["case_id"]): row for row in recomputed_case_rows}
    if stored_by_id != recomputed_by_id:
        raise RuntimeError("A4.4d stored claim verdicts fail independent reconstruction.")

    metrics = _metrics(pair_rows, recomputed_case_rows)
    stored_metrics = result["validation"]["registered_metrics"]
    for name, value in metrics.items():
        if name == "category_accuracy":
            for category, score in value.items():
                if not _close(float(score), float(stored_metrics["category_accuracy"][category])):
                    raise RuntimeError(f"Category metric drift: {category}.")
        elif isinstance(value, float):
            if not _close(float(value), float(stored_metrics[name])):
                raise RuntimeError(f"Metric drift: {name}.")
        elif value != stored_metrics[name]:
            raise RuntimeError(f"Metric drift: {name}.")

    checks = _requirement_checks(metrics, dict(config["validation_requirements"]))
    if checks != result["validation"]["requirement_checks"]:
        raise RuntimeError("A4.4d requirement checks fail independent reconstruction.")
    scientific_pass = all(checks.values())
    expected_status = (
        config["result_policy"]["pass_status"]
        if scientific_pass
        else config["result_policy"]["fail_status"]
    )
    if bool(result["scientific_pass"]) != scientific_pass or result["status"] != expected_status:
        raise RuntimeError("A4.4d scientific disposition does not match reconstructed gates.")

    audit = {
        "audit_id": "phase4-assistance-a4.4d-independent-post-audit-v1",
        "status": "PASSED_VALIDATION_ONLY_RECONSTRUCTION",
        "checks": {
            "validation_cases_144": True,
            "validation_pairs_246": True,
            "all_rows_validation_only": True,
            "pair_ids_exact_and_unique": True,
            "gold_relations_reconstructed": True,
            "raw_argmax_reconstructed": True,
            "frozen_temperature_argmax_preserved": True,
            "claim_verdicts_reconstructed": True,
            "registered_metrics_reconstructed": True,
            "requirement_checks_reconstructed": True,
            "scientific_status_reconstructed": True,
            "calibration_scored_zero": True,
            "candidate_scored_zero": True,
            "confirmatory_scored_zero": True,
        },
        "scientific_status": expected_status,
        "scientific_pass": scientific_pass,
    }
    (output_dir / "post_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# A4.4d independent post-audit",
        "",
        "- Reconstruction status: `PASSED_VALIDATION_ONLY_RECONSTRUCTION`",
        f"- Scientific status: `{expected_status}`",
        f"- Scientific pass: `{scientific_pass}`",
        "- 144 registered validation cases reconstructed.",
        "- 246 semantic pairs reconstructed with exact pair identities and gold relations.",
        "- Raw argmax classes and claim verdicts reconstructed independently from stored logits.",
        "- All preregistered A4.4a metrics and gate decisions reconstructed.",
        "- Calibration scoring, candidate scoring, and confirmatory scoring remained zero.",
        "",
    ]
    (output_dir / "post_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

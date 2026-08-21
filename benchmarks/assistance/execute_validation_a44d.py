# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub>=0.34,<1",
#   "numpy>=2.1,<3",
#   "safetensors>=0.6,<1",
#   "torch>=2.8,<3",
#   "transformers>=4.55,<5",
# ]
# ///
"""Execute frozen A4.4d validation-only semantic-verifier inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from validation_cases_a44d import (  # noqa: E402
    canonical_validation_jsonl_bytes,
    generate_validation_cases,
)

from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

A44A_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"
A44B_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"
A44C_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44c_v1.json"
A44D_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44d_v1.json"
A44C_CLOSURE_PATH = (
    ROOT
    / "benchmarks"
    / "assistance"
    / "results"
    / "a44c_calibration_postresult_v1"
    / "forensic_audit.json"
)
CACHE_ROOT = ROOT / ".cache" / "phase4-assistance-a44d"
RELATION_TO_LABEL = {"CONTRADICTED": 0, "UNKNOWN": 1, "ENTAILED": 2}
LABEL_TO_RELATION = {value: key for key, value in RELATION_TO_LABEL.items()}
FROZEN_A44A_SUITE_SHA256 = "0ad07e9d08678dbc5fa8b625870d2c3140eef83b0dddb013a4ae479c56bdd90c"
FROZEN_TEMPERATURE = 3.67


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gold_relation(atom: dict[str, Any], document_id: str) -> str:
    if document_id in {str(value) for value in atom.get("entailed_by", [])}:
        return "ENTAILED"
    if document_id in {str(value) for value in atom.get("contradicted_by", [])}:
        return "CONTRADICTED"
    return "UNKNOWN"


def _semantic_pairs(
    cases: list[dict[str, Any]], documents: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for case in cases:
        if case["split"] != "validation":
            raise RuntimeError("A4.4d received a non-validation case.")
        presented = {str(value) for value in case["presented_document_ids"]}
        cited = {str(value) for value in case["cited_document_ids"]}
        if not cited or not cited.issubset(presented):
            continue
        for atom in case["atoms"]:
            atom_id = str(atom["atom_id"])
            hypothesis = str(atom["text"])
            for document_id in sorted(cited):
                document = documents.get(document_id)
                if document is None:
                    raise RuntimeError(f"Missing frozen document: {document_id}")
                relation = _gold_relation(atom, document_id)
                pairs.append(
                    {
                        "pair_id": f"{case['case_id']}::{atom_id}::{document_id}",
                        "case_id": str(case["case_id"]),
                        "category": str(case["category"]),
                        "intent": str(case["intent"]),
                        "atom_id": atom_id,
                        "document_id": document_id,
                        "premise": str(document["body"]),
                        "hypothesis": hypothesis,
                        "gold_relation": relation,
                        "gold_label": RELATION_TO_LABEL[relation],
                    }
                )
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("A4.4d semantic pair ids must be unique.")
    return pairs


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return {
        LABEL_TO_RELATION[label]: int(np.sum(values == label))
        for label in sorted(LABEL_TO_RELATION)
    }


def _recall(gold: np.ndarray, predicted: np.ndarray, label: int) -> float:
    mask = gold == label
    denominator = int(np.sum(mask))
    if denominator == 0:
        return 0.0
    return float(np.sum(predicted[mask] == label) / denominator)


def _f1(gold: np.ndarray, predicted: np.ndarray, label: int) -> float:
    tp = int(np.sum((gold == label) & (predicted == label)))
    fp = int(np.sum((gold != label) & (predicted == label)))
    fn = int(np.sum((gold == label) & (predicted != label)))
    denominator = (2 * tp) + fp + fn
    if denominator == 0:
        return 0.0
    return float((2 * tp) / denominator)


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

    any_atomic_conflict = False
    every_atom_supported = True
    for atom in case["atoms"]:
        atom_id = str(atom["atom_id"])
        relations = {
            predicted_relations[f"{case['case_id']}::{atom_id}::{document_id}"]
            for document_id in sorted(cited)
        }
        if "ENTAILED" in relations and "CONTRADICTED" in relations:
            any_atomic_conflict = True
        if "ENTAILED" not in relations:
            every_atom_supported = False

    if any_atomic_conflict:
        return "CONFLICTING_EVIDENCE"
    if not every_atom_supported:
        return "UNSUPPORTED"
    return "SUPPORTED"


def _case_metrics(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({str(row["category"]) for row in case_rows})
    category_accuracy: dict[str, float] = {}
    for category in categories:
        rows = [row for row in case_rows if row["category"] == category]
        category_accuracy[category] = float(
            sum(row["predicted_verdict"] == row["gold_verdict"] for row in rows) / len(rows)
        )

    gold_supported = [row for row in case_rows if row["gold_verdict"] == "SUPPORTED"]
    predicted_supported = [row for row in case_rows if row["predicted_verdict"] == "SUPPORTED"]
    true_supported = [
        row
        for row in case_rows
        if row["gold_verdict"] == "SUPPORTED" and row["predicted_verdict"] == "SUPPORTED"
    ]
    supported_precision = (
        float(len(true_supported) / len(predicted_supported)) if predicted_supported else 0.0
    )
    supported_recall = float(len(true_supported) / len(gold_supported)) if gold_supported else 0.0

    safety_categories = {
        "citation_invalid",
        "stale_current_evidence",
        "unresolved_conflict",
    }
    false_supported_safety = sum(
        row["predicted_verdict"] == "SUPPORTED" and row["category"] in safety_categories
        for row in case_rows
    )

    return {
        "category_accuracy": category_accuracy,
        "macro_case_category_accuracy": float(
            sum(category_accuracy.values()) / len(category_accuracy)
        ),
        "supported_precision": supported_precision,
        "supported_recall": supported_recall,
        "false_supported_on_citation_stale_or_conflict_cases": int(false_supported_safety),
    }


def _registered_metrics(
    gold: np.ndarray,
    predicted: np.ndarray,
    case_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case = _case_metrics(case_rows)
    f1_values = [_f1(gold, predicted, label) for label in sorted(LABEL_TO_RELATION)]
    relation = {
        "atomic_relation_macro_f1": float(sum(f1_values) / len(f1_values)),
        "atomic_entailment_recall": _recall(gold, predicted, RELATION_TO_LABEL["ENTAILED"]),
        "atomic_contradiction_recall": _recall(gold, predicted, RELATION_TO_LABEL["CONTRADICTED"]),
        "atomic_unknown_recall": _recall(gold, predicted, RELATION_TO_LABEL["UNKNOWN"]),
    }
    return {
        **relation,
        "macro_case_category_accuracy": case["macro_case_category_accuracy"],
        "supported_precision": case["supported_precision"],
        "supported_recall": case["supported_recall"],
        "literal_supported_recall": case["category_accuracy"]["literal_supported"],
        "paraphrase_supported_recall": case["category_accuracy"]["paraphrase_supported"],
        "contradiction_unsupported_accuracy": case["category_accuracy"][
            "contradiction_unsupported"
        ],
        "unsupported_approval_accuracy": case["category_accuracy"]["unsupported_approval"],
        "multi_document_supported_recall": case["category_accuracy"]["multi_document_supported"],
        "partial_multi_document_unsupported_accuracy": case["category_accuracy"][
            "partial_multi_document_unsupported"
        ],
        "citation_invalid_accuracy": case["category_accuracy"]["citation_invalid"],
        "stale_current_evidence_accuracy": case["category_accuracy"]["stale_current_evidence"],
        "unresolved_conflict_accuracy": case["category_accuracy"]["unresolved_conflict"],
        "false_supported_on_citation_stale_or_conflict_cases": case[
            "false_supported_on_citation_stale_or_conflict_cases"
        ],
        "category_accuracy": case["category_accuracy"],
    }


def _evaluate_requirements(
    metrics: dict[str, Any], requirements: dict[str, Any]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, value in requirements.items():
        if name == "all_requirements_must_pass":
            continue
        if name.endswith("_min"):
            metric_name = name[: -len("_min")]
            checks[name] = float(metrics[metric_name]) >= float(value)
        else:
            checks[name] = metrics[name] == value
    return checks


class NliEngine:
    """Pinned FP32 CPU RoBERTa verifier for A4.4d validation pairs only."""

    def __init__(self, binding: dict[str, Any]) -> None:
        self.binding = binding
        local_dir = CACHE_ROOT / "semantic-verifier"
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(binding["model_id"]),
            revision=str(binding["revision"]),
            local_dir=local_dir,
            allow_patterns=[
                "config.json",
                "merges.txt",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        )
        self.model_path = local_dir / str(binding["weights_file"])
        if not self.model_path.exists():
            raise FileNotFoundError(f"Pinned safetensors model missing: {self.model_path}")
        actual_hash = _sha256(self.model_path)
        expected_hash = str(binding["weights_sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Pinned safetensors SHA256 mismatch: expected {expected_hash}, got {actual_hash}."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        id2label = {
            int(key): str(value).upper() for key, value in self.model.config.id2label.items()
        }
        expected_labels = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        if id2label != expected_labels:
            raise RuntimeError(f"Pinned native label mapping drifted: {id2label}")
        self.model.to(torch.device("cpu"))
        self.model.eval()

    def logits(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        tokenization = self.binding["tokenization"]
        encoded = self.tokenizer(
            premises,
            hypotheses,
            max_length=int(tokenization["max_length"]),
            truncation=bool(tokenization["truncation"]),
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = self.model(**encoded)
        logits = outputs.logits.detach().cpu().numpy().astype(np.float64)
        if logits.ndim != 2 or logits.shape[1] != 3:
            raise RuntimeError(f"Expected three logits per pair, got shape {logits.shape}.")
        return logits


def execute(output_dir: Path) -> dict[str, Any]:
    """Score the sealed validation split once and evaluate frozen A4.4a requirements."""
    a44a = _load(A44A_CONFIG_PATH)
    a44b = _load(A44B_CONFIG_PATH)
    a44c = _load(A44C_CONFIG_PATH)
    a44d = _load(A44D_CONFIG_PATH)
    closure = _load(A44C_CLOSURE_PATH)
    binding = a44b["semantic_verifier"]

    if a44d["source_main_sha"] != "57bb2c81ab2cc2d5b8c1a4928c2600b4a770d110":
        raise RuntimeError("A4.4d predecessor main SHA drifted.")
    if a44d["binding_id"] != a44b["binding_id"]:
        raise RuntimeError("A4.4d binding id does not match frozen A4.4b.")
    if a44d["protocol_id"] != a44a["protocol_id"]:
        raise RuntimeError("A4.4d protocol id does not match frozen A4.4a.")
    if a44d["calibration_execution_id"] != a44c["execution_id"]:
        raise RuntimeError("A4.4d calibration execution id does not match A4.4c.")
    if str(a44a["validation_suite"]["sha256"]) != FROZEN_A44A_SUITE_SHA256:
        raise RuntimeError("Frozen A4.4a suite registration hash drifted.")
    if closure["status"] != "CLOSED_CALIBRATION_TEMPERATURE_FROZEN":
        raise RuntimeError("A4.4c closure status is not frozen.")
    if closure["scientific_disposition"]["global_temperature_is_frozen_at_3_67"] is not True:
        raise RuntimeError("A4.4c closure does not freeze T=3.67.")
    if float(a44d["frozen_calibration"]["selected_temperature"]) != FROZEN_TEMPERATURE:
        raise RuntimeError("A4.4d frozen temperature drifted.")
    if a44d["validation_requirements"] != a44a["future_validation_requirements"]:
        raise RuntimeError("A4.4d validation requirements drifted from A4.4a.")

    bundle = generate_bundle()
    validation_cases = generate_validation_cases(bundle)
    if len(validation_cases) != int(a44d["scope"]["validation_case_rows"]):
        raise RuntimeError("A4.4d validation case count mismatch.")
    if any(row["split"] != "validation" for row in validation_cases):
        raise RuntimeError("A4.4d materialized a non-validation row.")
    validation_case_sha = hashlib.sha256(
        canonical_validation_jsonl_bytes(validation_cases)
    ).hexdigest()

    documents = {str(row["document_id"]): dict(row) for row in bundle.documents}
    pairs = _semantic_pairs(validation_cases, documents)
    expected_pairs = int(a44d["scope"]["validation_semantic_pair_rows"])
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"A4.4d validation semantic-pair count mismatch: expected {expected_pairs}, "
            f"got {len(pairs)}."
        )

    gold = np.asarray([int(row["gold_label"]) for row in pairs], dtype=np.int64)
    expected_gold = {
        str(key): int(value)
        for key, value in a44d["scope"]["validation_gold_relation_counts"].items()
    }
    if _class_counts(gold) != expected_gold:
        raise RuntimeError("A4.4d validation semantic-pair gold counts drifted.")

    engine = NliEngine(binding)
    all_logits: list[np.ndarray] = []
    batch_size = int(binding["batch_size"])
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        all_logits.append(
            engine.logits(
                [str(row["premise"]) for row in chunk],
                [str(row["hypothesis"]) for row in chunk],
            )
        )
    logits = np.concatenate(all_logits, axis=0)
    raw_argmax = np.argmax(logits, axis=1).astype(np.int64)
    calibrated_argmax = np.argmax(logits / FROZEN_TEMPERATURE, axis=1).astype(np.int64)
    if not np.array_equal(raw_argmax, calibrated_argmax):
        raise RuntimeError("Frozen positive temperature unexpectedly changed an argmax class.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []
    predicted_relations: dict[str, str] = {}
    for index, (pair, row_logits) in enumerate(zip(pairs, logits, strict=True)):
        predicted_relation = LABEL_TO_RELATION[int(raw_argmax[index])]
        predicted_relations[str(pair["pair_id"])] = predicted_relation
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "category": pair["category"],
                "intent": pair["intent"],
                "atom_id": pair["atom_id"],
                "document_id": pair["document_id"],
                "split": "validation",
                "gold_relation": pair["gold_relation"],
                "gold_label": int(gold[index]),
                "logits": [float(value) for value in row_logits.tolist()],
                "raw_argmax_label": int(raw_argmax[index]),
                "raw_argmax_relation": predicted_relation,
                "raw_correct": bool(raw_argmax[index] == gold[index]),
            }
        )

    case_rows: list[dict[str, Any]] = []
    for case in validation_cases:
        predicted_verdict = _predict_case_verdict(case, documents, predicted_relations)
        case_rows.append(
            {
                "case_id": str(case["case_id"]),
                "intent": str(case["intent"]),
                "category": str(case["category"]),
                "split": "validation",
                "gold_verdict": str(case["expected_verdict"]),
                "predicted_verdict": predicted_verdict,
                "correct": predicted_verdict == str(case["expected_verdict"]),
            }
        )

    metrics = _registered_metrics(gold, raw_argmax, case_rows)
    requirements = dict(a44d["validation_requirements"])
    requirement_checks = _evaluate_requirements(metrics, requirements)
    all_pass = all(requirement_checks.values())
    expected_all = bool(requirements["all_requirements_must_pass"])
    scientific_pass = all_pass if expected_all else any(requirement_checks.values())
    status = (
        a44d["result_policy"]["pass_status"]
        if scientific_pass
        else a44d["result_policy"]["fail_status"]
    )

    (output_dir / "validation_pair_logits.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    (output_dir / "validation_case_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in case_rows),
        encoding="utf-8",
    )

    result = {
        "execution_id": a44d["execution_id"],
        "checkpoint": "A4.4d",
        "status": status,
        "scientific_pass": scientific_pass,
        "source_main_sha": a44d["source_main_sha"],
        "protocol_id": a44d["protocol_id"],
        "binding_id": a44d["binding_id"],
        "a44a_suite_sha256": FROZEN_A44A_SUITE_SHA256,
        "validation_case_sha256": validation_case_sha,
        "frozen_temperature": FROZEN_TEMPERATURE,
        "validation": {
            "case_rows": len(validation_cases),
            "semantic_pairs": len(pairs),
            "gold_relation_counts": _class_counts(gold),
            "raw_argmax_relation_counts": _class_counts(raw_argmax),
            "raw_argmax_accuracy": float(np.mean(raw_argmax == gold)),
            "registered_metrics": metrics,
            "requirement_checks": requirement_checks,
            "requirements_passed": sum(requirement_checks.values()),
            "requirements_total": len(requirement_checks),
            "all_requirements_pass": scientific_pass,
            "argmax_preserved_after_frozen_temperature": True,
        },
        "semantic_verifier": {
            "model_id": binding["model_id"],
            "revision": binding["revision"],
            "weights_file": binding["weights_file"],
            "weights_sha256": _sha256(engine.model_path),
            "device": "CPU",
            "dtype": "float32",
            "batch_size": batch_size,
            "class_decision": "argmax_raw_logits",
        },
        "sealed_boundaries": {
            "calibration_case_rows_materialized": 0,
            "candidate_rows_scored": 0,
            "confirmatory_query_records_inspected": 0,
            "confirmatory_queries_scored": 0,
            "temperature_refits": 0,
            "threshold_searches": 0,
            "model_substitutions": 0,
            "post_result_rescues": 0,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# A4.4d frozen validation result",
        "",
        f"- Status: `{status}`",
        f"- Scientific pass: `{scientific_pass}`",
        f"- Validation cases: `{len(validation_cases)}`",
        f"- Semantic pairs: `{len(pairs)}`",
        f"- Frozen temperature: `{FROZEN_TEMPERATURE}`",
        f"- Raw argmax accuracy: `{float(np.mean(raw_argmax == gold)):.12f}`",
        "",
        "## Registered validation metrics",
        "",
    ]
    for name, value in metrics.items():
        if name == "category_accuracy":
            continue
        report_lines.append(f"- {name}: `{value}`")
    report_lines.extend(["", "## Requirement checks", ""])
    for name, passed in requirement_checks.items():
        report_lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    report_lines.extend(
        [
            "",
            "No refit, threshold search, model substitution, candidate scoring, "
            "confirmatory inspection, or post-result rescue is authorized by this result.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"status": status, "scientific_pass": scientific_pass}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.output_dir)


if __name__ == "__main__":
    main()

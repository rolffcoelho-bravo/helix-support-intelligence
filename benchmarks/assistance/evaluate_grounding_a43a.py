# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.28,<1",
#   "huggingface-hub>=0.34,<1",
#   "numpy>=2.1,<3",
#   "onnxruntime>=1.22,<2",
#   "sentencepiece>=0.2,<1",
#   "transformers>=4.55,<5",
# ]
# ///
"""Execute the frozen A4.3a candidate-independent grounding validity test."""

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
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "assistance"))

from grounding_anchors_a43a import canonical_jsonl_bytes, generate_anchors  # noqa: E402

from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "models" / "assistance_validity_a43a_v1.json"
BINDING_PATH = ROOT / "configs" / "models" / "assistance_binding_a41_v1.json"
CACHE_ROOT = ROOT / ".cache" / "phase4-assistance-a43a"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positive = [row for row in rows if bool(row["expected_entailment"])]
    negative = [row for row in rows if not bool(row["expected_entailment"])]
    true_positive = sum(float(row["entailment_probability"]) >= threshold for row in positive)
    true_negative = sum(float(row["entailment_probability"]) < threshold for row in negative)
    sensitivity = true_positive / len(positive) if positive else None
    specificity = true_negative / len(negative) if negative else None
    balanced = (
        (float(sensitivity) + float(specificity)) / 2.0
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "rows": len(rows),
        "positive_rows": len(positive),
        "negative_rows": len(negative),
        "positive_sensitivity": sensitivity,
        "negative_specificity": specificity,
        "balanced_accuracy": balanced,
    }


def _category_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    categories = sorted({str(row["category"]) for row in rows})
    return {
        category: _metric_rows([row for row in rows if row["category"] == category], threshold)
        for category in categories
    }


def _select_threshold(calibration: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    spec = config["evaluator_validation"]["threshold_calibration"]
    minimum = round(float(spec["grid_min"]) * 100)
    maximum = round(float(spec["grid_max"]) * 100)
    step = round(float(spec["grid_step"]) * 100)
    candidates: list[dict[str, Any]] = []
    for integer in range(minimum, maximum + 1, step):
        threshold = integer / 100.0
        metrics = _metric_rows(calibration, threshold)
        sensitivity = float(metrics["positive_sensitivity"])
        specificity = float(metrics["negative_specificity"])
        candidates.append(
            {
                "threshold": threshold,
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "positive_sensitivity": sensitivity,
                "negative_specificity": specificity,
                "minimum_class_rate": min(sensitivity, specificity),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            row["minimum_class_rate"],
            row["threshold"],
        ),
    )
    return {"selected": selected, "grid": candidates}


def _validation_checks(
    validation: list[dict[str, Any]],
    threshold: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    overall = _metric_rows(validation, threshold)
    categories = _category_metrics(validation, threshold)
    requirements = config["evaluator_validation"]["validation_pass_requirements"]

    checks = {
        "overall_positive_sensitivity": float(overall["positive_sensitivity"])
        >= float(requirements["overall_positive_sensitivity_min"]),
        "overall_negative_specificity": float(overall["negative_specificity"])
        >= float(requirements["overall_negative_specificity_min"]),
        "balanced_accuracy": float(overall["balanced_accuracy"])
        >= float(requirements["balanced_accuracy_min"]),
        "literal_policy_sensitivity": float(categories["literal_policy"]["positive_sensitivity"])
        >= float(requirements["literal_policy_sensitivity_min"]),
        "paraphrase_queue_sensitivity": float(
            categories["paraphrase_queue"]["positive_sensitivity"]
        )
        >= float(requirements["paraphrase_queue_sensitivity_min"]),
        "multi_document_conjunction_sensitivity": float(
            categories["multi_document_conjunction"]["positive_sensitivity"]
        )
        >= float(requirements["multi_document_conjunction_sensitivity_min"]),
        "contradiction_queue_specificity": float(
            categories["contradiction_queue"]["negative_specificity"]
        )
        >= float(requirements["contradiction_queue_specificity_min"]),
        "unsupported_approval_specificity": float(
            categories["unsupported_approval"]["negative_specificity"]
        )
        >= float(requirements["unsupported_approval_specificity_min"]),
        "citation_mismatch_specificity": float(
            categories["citation_mismatch"]["negative_specificity"]
        )
        >= float(requirements["citation_mismatch_specificity_min"]),
        "stale_current_claim_specificity": float(
            categories["stale_current_claim"]["negative_specificity"]
        )
        >= float(requirements["stale_current_claim_specificity_min"]),
        "conflict_union_claim_specificity": float(
            categories["conflict_union_claim"]["negative_specificity"]
        )
        >= float(requirements["conflict_union_claim_specificity_min"]),
    }
    return {
        "overall": overall,
        "categories": categories,
        "requirements": requirements,
        "checks": checks,
        "passed": all(checks.values()),
    }


class NliEngine:
    """Pinned local ONNX evaluator used only on A4.3a anchor pairs."""

    def __init__(self, binding: dict[str, Any]) -> None:
        self.binding = binding
        local_dir = CACHE_ROOT / "evaluation"
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(binding["model_id"]),
            revision=str(binding["revision"]),
            local_dir=local_dir,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "spm.model",
                "sentencepiece.bpe.model",
                str(binding["onnx_path"]),
            ],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        self.model_path = local_dir / str(binding["onnx_path"])
        if not self.model_path.exists():
            raise FileNotFoundError(f"Pinned ONNX model missing: {self.model_path}")
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def probabilities(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        encoded = self.tokenizer(
            premises,
            hypotheses,
            max_length=int(self.binding["max_length"]),
            truncation=True,
            padding=True,
            return_tensors="np",
        )
        feed = {key: np.asarray(value) for key, value in encoded.items() if key in self.input_names}
        logits = np.asarray(self.session.run(None, feed)[0]).astype(float)
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probabilities = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
        label = int(self.binding["entailment_label"])
        if label < 0 or label >= probabilities.shape[1]:
            raise RuntimeError("Frozen entailment label is outside model logits.")
        return [float(row[label]) for row in probabilities]


def execute(output_dir: Path) -> dict[str, Any]:
    """Run calibration and one untouched validation pass on candidate-independent anchors."""
    config = _load(CONFIG_PATH)
    binding = _load(BINDING_PATH)["evaluation_verifier"]
    validation_binding = config["evaluator_validation"]
    for key in ("model_id", "revision", "architecture_family", "entailment_label"):
        if binding[key] != validation_binding[key]:
            raise RuntimeError(f"A4.3a evaluator binding mismatch for {key}.")

    bundle = generate_bundle()
    documents = {str(row["document_id"]): row for row in bundle.documents}
    anchors = generate_anchors(bundle)
    engine = NliEngine(binding)
    scored: list[dict[str, Any]] = []
    batch_size = int(binding["batch_size"])
    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start : start + batch_size]
        premises = [
            "\n".join(
                f"{documents[document_id]['title']}\n{documents[document_id]['body']}"
                for document_id in row["document_ids"]
            )
            for row in chunk
        ]
        hypotheses = [str(row["hypothesis"]) for row in chunk]
        probabilities = engine.probabilities(premises, hypotheses)
        for row, probability in zip(chunk, probabilities, strict=True):
            scored.append({**row, "entailment_probability": probability})

    calibration = [row for row in scored if row["split"] == "calibration"]
    validation = [row for row in scored if row["split"] == "validation"]
    existing_threshold = float(validation_binding["existing_threshold"])
    existing_calibration = {
        "threshold": existing_threshold,
        "overall": _metric_rows(calibration, existing_threshold),
        "categories": _category_metrics(calibration, existing_threshold),
    }
    selection = _select_threshold(calibration, config)
    selected_threshold = float(selection["selected"]["threshold"])
    holdout = _validation_checks(validation, selected_threshold, config)
    status = "PASSED_EVALUATOR_VALIDITY" if holdout["passed"] else "FAILED_EVALUATOR_VALIDITY"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "anchor_scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )
    result = {
        "validity_id": config["validity_id"],
        "status": status,
        "evaluator": {
            "model_id": binding["model_id"],
            "revision": binding["revision"],
            "architecture_family": binding["architecture_family"],
            "onnx_path": binding["onnx_path"],
            "onnx_sha256": _sha256(engine.model_path),
            "entailment_label": binding["entailment_label"],
            "max_length": binding["max_length"],
            "batch_size": binding["batch_size"],
        },
        "anchor_suite": {
            "rows": len(anchors),
            "calibration_rows": len(calibration),
            "validation_rows": len(validation),
            "sha256": hashlib.sha256(canonical_jsonl_bytes(anchors)).hexdigest(),
        },
        "existing_threshold_calibration_diagnostic": existing_calibration,
        "threshold_selection": selection,
        "selected_threshold": selected_threshold,
        "validation": holdout,
        "candidate_calls": 0,
        "openai_calls": 0,
        "candidate_scores_computed": 0,
        "confirmatory_queries_scored": 0,
        "post_validation_threshold_adjustment": False,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# A4.3a grounding evaluator validity",
                "",
                f"**Status: {status}**",
                "",
                f"Selected calibration threshold: **{selected_threshold:.2f}**.",
                (
                    "Validation positive sensitivity: "
                    f"**{float(holdout['overall']['positive_sensitivity']):.4f}**."
                ),
                (
                    "Validation negative specificity: "
                    f"**{float(holdout['overall']['negative_specificity']):.4f}**."
                ),
                (
                    "Validation balanced accuracy: "
                    f"**{float(holdout['overall']['balanced_accuracy']):.4f}**."
                ),
                "",
                "No assistance candidate, OpenAI call, or confirmatory query was used.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.output_dir)
    print(
        json.dumps({"status": result["status"], "selected_threshold": result["selected_threshold"]})
    )


if __name__ == "__main__":
    main()

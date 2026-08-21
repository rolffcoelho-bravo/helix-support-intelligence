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
"""Execute frozen A4.4c calibration-only semantic-verifier inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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

from compositional_cases_a44a import canonical_jsonl_bytes, generate_cases  # noqa: E402

from helix_support_intelligence.data.helixbank import generate_bundle  # noqa: E402

A44A_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44a_v1.json"
A44B_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44b_v1.json"
A44C_CONFIG_PATH = ROOT / "configs" / "models" / "assistance_grounding_a44c_v1.json"
CACHE_ROOT = ROOT / ".cache" / "phase4-assistance-a44c"
RELATION_TO_LABEL = {"CONTRADICTED": 0, "UNKNOWN": 1, "ENTAILED": 2}
LABEL_TO_RELATION = {value: key for key, value in RELATION_TO_LABEL.items()}


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
    calibration_cases: list[dict[str, Any]], documents: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for case in calibration_cases:
        if case["split"] != "calibration":
            raise RuntimeError("A4.4c received a non-calibration case.")
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
        raise RuntimeError("A4.4c semantic pair ids must be unique.")
    return pairs


def _nll(logits: np.ndarray, gold: np.ndarray, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    scaled = logits / temperature
    maxima = np.max(scaled, axis=1, keepdims=True)
    stabilized = scaled - maxima
    logsumexp = np.log(np.exp(stabilized).sum(axis=1)) + maxima[:, 0]
    selected = scaled[np.arange(len(gold)), gold]
    return float(np.mean(logsumexp - selected))


def _grid() -> list[float]:
    return [integer / 100.0 for integer in range(25, 401)]


def _fit_temperature(logits: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    candidates = [
        {"temperature": temperature, "nll": _nll(logits, gold, temperature)}
        for temperature in _grid()
    ]
    selected = min(candidates, key=lambda row: (float(row["nll"]), float(row["temperature"])))
    return {"selected": selected, "grid": candidates}


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return {
        LABEL_TO_RELATION[label]: int(np.sum(values == label))
        for label in sorted(LABEL_TO_RELATION)
    }


class NliEngine:
    """Pinned FP32 CPU RoBERTa verifier for A4.4c calibration pairs only."""

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
    """Score calibration semantic pairs only and freeze one global temperature."""
    a44a = _load(A44A_CONFIG_PATH)
    a44b = _load(A44B_CONFIG_PATH)
    a44c = _load(A44C_CONFIG_PATH)
    binding = a44b["semantic_verifier"]

    if a44c["source_main_sha"] != "d94b834f61c4208028907c602b54792c643ac074":
        raise RuntimeError("A4.4c predecessor main SHA drifted.")
    if a44c["binding_id"] != a44b["binding_id"]:
        raise RuntimeError("A4.4c binding id does not match frozen A4.4b.")
    if a44c["protocol_id"] != a44a["protocol_id"]:
        raise RuntimeError("A4.4c protocol id does not match frozen A4.4a.")

    cases = generate_cases()
    suite_hash = hashlib.sha256(canonical_jsonl_bytes(cases)).hexdigest()
    expected_suite_hash = str(a44a["validation_suite"]["sha256"])
    if suite_hash != expected_suite_hash:
        raise RuntimeError("Frozen A4.4a suite hash mismatch.")
    calibration_cases = [row for row in cases if row["split"] == "calibration"]
    if len(calibration_cases) != int(a44c["scope"]["calibration_case_rows"]):
        raise RuntimeError("A4.4c calibration case count mismatch.")

    bundle = generate_bundle()
    documents = {str(row["document_id"]): dict(row) for row in bundle.documents}
    pairs = _semantic_pairs(calibration_cases, documents)
    expected_pairs = int(a44c["scope"]["calibration_semantic_pair_rows"])
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"A4.4c calibration semantic-pair count mismatch: expected {expected_pairs}, "
            f"got {len(pairs)}."
        )

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
    gold = np.asarray([int(row["gold_label"]) for row in pairs], dtype=np.int64)
    raw_argmax = np.argmax(logits, axis=1).astype(np.int64)

    temperature_fit = _fit_temperature(logits, gold)
    selected_temperature = float(temperature_fit["selected"]["temperature"])
    calibrated_argmax = np.argmax(logits / selected_temperature, axis=1).astype(np.int64)
    if not np.array_equal(raw_argmax, calibrated_argmax):
        raise RuntimeError("Positive global temperature unexpectedly changed an argmax class.")

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_rows: list[dict[str, Any]] = []
    for index, (pair, row_logits) in enumerate(zip(pairs, logits, strict=True)):
        scored_rows.append(
            {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "category": pair["category"],
                "intent": pair["intent"],
                "atom_id": pair["atom_id"],
                "document_id": pair["document_id"],
                "split": "calibration",
                "gold_relation": pair["gold_relation"],
                "gold_label": int(gold[index]),
                "logits": [float(value) for value in row_logits.tolist()],
                "raw_argmax_label": int(raw_argmax[index]),
                "raw_argmax_relation": LABEL_TO_RELATION[int(raw_argmax[index])],
                "raw_correct": bool(raw_argmax[index] == gold[index]),
            }
        )
    (output_dir / "calibration_pair_logits.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    (output_dir / "temperature_grid.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in temperature_fit["grid"]),
        encoding="utf-8",
    )

    raw_nll = _nll(logits, gold, 1.0)
    calibrated_nll = _nll(logits, gold, selected_temperature)
    result = {
        "execution_id": a44c["execution_id"],
        "checkpoint": "A4.4c",
        "status": "CALIBRATION_TEMPERATURE_FROZEN",
        "source_main_sha": a44c["source_main_sha"],
        "protocol_id": a44c["protocol_id"],
        "binding_id": a44c["binding_id"],
        "a44a_suite_sha256": suite_hash,
        "calibration": {
            "case_rows": len(calibration_cases),
            "semantic_pairs": len(pairs),
            "gold_relation_counts": _class_counts(gold),
            "raw_argmax_relation_counts": _class_counts(raw_argmax),
            "raw_argmax_accuracy": float(np.mean(raw_argmax == gold)),
            "raw_nll_at_temperature_1": raw_nll,
            "selected_temperature": selected_temperature,
            "selected_temperature_nll": calibrated_nll,
            "nll_improvement": raw_nll - calibrated_nll,
            "grid_start": 0.25,
            "grid_stop": 4.0,
            "grid_step": 0.01,
            "grid_points": len(temperature_fit["grid"]),
            "tie_break": "smallest_temperature",
            "argmax_preserved_after_temperature": True,
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
            "validation_case_rows_scored": 0,
            "validation_semantic_pairs_scored": 0,
            "validation_metrics_computed": 0,
            "g0_g1_g2_candidates_scored": 0,
            "confirmatory_query_records_inspected": 0,
            "confirmatory_queries_scored": 0,
            "model_family_comparisons": 0,
            "post_result_model_substitutions": 0,
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
    if not math.isclose(
        float(temperature_fit["selected"]["nll"]), calibrated_nll, rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError("Selected-grid NLL does not match reconstructed calibrated NLL.")

    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# A4.4c calibration-only execution",
                "",
                "**Status: CALIBRATION_TEMPERATURE_FROZEN**",
                "",
                f"Calibration cases scored: **{len(calibration_cases)}**.",
                f"Eligible semantic pairs scored: **{len(pairs)}**.",
                f"Frozen global temperature: **{selected_temperature:.2f}**.",
                f"Raw three-class NLL: **{raw_nll:.6f}**.",
                f"Temperature-scaled NLL: **{calibrated_nll:.6f}**.",
                (
                    "Raw argmax accuracy on calibration pairs: "
                    f"**{float(np.mean(raw_argmax == gold)):.4f}**."
                ),
                "",
                "The positive global temperature preserved every raw-logit argmax class.",
                "No A4.4a validation case, G0/G1/G2 candidate, or confirmatory query was scored.",
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
        json.dumps(
            {
                "status": result["status"],
                "selected_temperature": result["calibration"]["selected_temperature"],
                "semantic_pairs": result["calibration"]["semantic_pairs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

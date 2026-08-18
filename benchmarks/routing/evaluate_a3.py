# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5",
#   "scikit-learn==1.8.0",
#   "scipy==1.17.0",
#   "torch==2.13.0",
#   "transformers==4.57.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Run the frozen Phase 2 A3 compact-transformer benchmark on validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as functional
import transformers
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ECE_BINS = 15
RISK_COVERAGE_POINTS = tuple(step / 10 for step in range(1, 11))
RUN_ID = "phase2-development-a3-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "banking77.json"
A3_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "routing_a3.json"
A2_CHECKPOINT_PATH = (
    REPO_ROOT / "benchmarks" / "routing" / "results" / "a2_validation_v2.json"
)


class MeanPoolClassifier(nn.Module):
    """Fine-tunable MiniLM encoder with A2-compatible pooling geometry."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        hidden_dimension: int,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        )
        observed_hidden = int(self.encoder.config.hidden_size)
        if observed_hidden != hidden_dimension:
            raise ValueError(
                f"A3 hidden dimension drifted: expected {hidden_dimension}, got {observed_hidden}"
            )
        self.classifier = nn.Linear(hidden_dimension, num_labels)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
            torch.sum(mask, dim=1),
            min=1e-9,
        )
        normalized = functional.normalize(pooled, p=2, dim=1)
        return self.classifier(normalized)


def _data_module() -> Any:
    source_root = str(REPO_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return importlib.import_module("helix_support_intelligence.data.banking77")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-support-intelligence-phase2/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def _top_k_recall(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    top_k = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k == y_true[:, None], axis=1)))


def _multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(np.sum(mask))
        if count == 0:
            continue
        accuracy = float(np.mean(correct[mask]))
        mean_confidence = float(np.mean(confidence[mask]))
        ece += (count / total) * abs(accuracy - mean_confidence)
    return float(ece)


def _risk_coverage(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    confidence = np.max(probabilities, axis=1)
    order = np.argsort(-confidence, kind="stable")
    rows: list[dict[str, float | int]] = []
    for coverage in RISK_COVERAGE_POINTS:
        accepted = max(1, min(len(order), round(coverage * len(order))))
        selected = order[:accepted]
        accuracy = float(np.mean(predicted[selected] == y_true[selected]))
        rows.append(
            {
                "coverage": coverage,
                "accepted": accepted,
                "confidence_threshold": float(confidence[selected[-1]]),
                "selective_risk": 1.0 - accuracy,
            }
        )
    return rows


def _confusion_pairs(
    y_true: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
    limit: int = 20,
) -> list[dict[str, str | int]]:
    errors = Counter(
        (labels[int(true_index)], labels[int(pred_index)])
        for true_index, pred_index in zip(y_true, predicted, strict=True)
        if true_index != pred_index
    )
    return [
        {"true_intent": pair[0], "predicted_intent": pair[1], "count": count}
        for pair, count in errors.most_common(limit)
    ]


def _metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "top3_recall": _top_k_recall(y_true, probabilities, 3),
        "expected_calibration_error_15bin": _expected_calibration_error(
            y_true,
            predicted,
            probabilities,
        ),
        "multiclass_brier_score": _multiclass_brier(y_true, probabilities),
        "mean_max_probability": float(np.mean(np.max(probabilities, axis=1))),
        "risk_coverage": _risk_coverage(y_true, predicted, probabilities),
        "top_confusion_pairs": _confusion_pairs(y_true, predicted, labels),
    }


def _risk_coverage_delta(
    a2_rows: list[object],
    a3_rows: list[object],
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for a2_row, a3_row in zip(a2_rows, a3_rows, strict=True):
        assert isinstance(a2_row, dict)
        assert isinstance(a3_row, dict)
        coverage = float(a2_row["coverage"])
        if coverage != float(a3_row["coverage"]):
            raise ValueError("A2/A3 risk-coverage grid drifted")
        a2_risk = float(a2_row["a2_selective_risk"])
        a3_risk = float(a3_row["selective_risk"])
        output.append(
            {
                "coverage": coverage,
                "a2_selective_risk": a2_risk,
                "a3_selective_risk": a3_risk,
                "a3_minus_a2_selective_risk": a3_risk - a2_risk,
            }
        )
    return output


def _a2_confusion_changes(
    a2_rows: list[object],
    y_true: np.ndarray,
    a3_predicted: np.ndarray,
    labels: list[str],
) -> list[dict[str, str | int]]:
    a3_errors = Counter(
        (labels[int(true_index)], labels[int(pred_index)])
        for true_index, pred_index in zip(y_true, a3_predicted, strict=True)
        if true_index != pred_index
    )
    output: list[dict[str, str | int]] = []
    for row in a2_rows:
        assert isinstance(row, dict)
        key = (str(row["true_intent"]), str(row["predicted_intent"]))
        a2_count = int(row["count"])
        a3_count = a3_errors.get(key, 0)
        output.append(
            {
                "true_intent": key[0],
                "predicted_intent": key[1],
                "a2_count": a2_count,
                "a3_count": a3_count,
                "a3_minus_a2": a3_count - a2_count,
            }
        )
    return output


def _sha256_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _prediction_hashes(
    validation: list[Any],
    predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    source_revision: str,
    data_api: Any,
) -> dict[str, str]:
    decision_lines: list[str] = []
    top3_lines: list[str] = []
    rounded_confidence_lines: list[str] = []
    for example, pred_index, row in zip(validation, predicted, probabilities, strict=True):
        sample = data_api.sample_id(example, source_revision)
        decision = labels[int(pred_index)]
        order = np.argsort(-row, kind="stable")[:3]
        top3 = "|".join(labels[int(index)] for index in order)
        confidence = float(row[int(pred_index)])
        decision_lines.append(f"{sample}\t{decision}")
        top3_lines.append(f"{sample}\t{top3}")
        rounded_confidence_lines.append(f"{sample}\t{confidence:.6f}")
    return {
        "predicted_intents_sha256": _sha256_lines(decision_lines),
        "top3_intents_sha256": _sha256_lines(top3_lines),
        "confidence_rounded_6_sha256": _sha256_lines(rounded_confidence_lines),
    }


def _write_predictions(
    path: Path,
    validation: list[Any],
    predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    source_revision: str,
    data_api: Any,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "true_intent",
                "predicted_intent",
                "confidence",
                "top3_intents",
            ],
        )
        writer.writeheader()
        for example, pred_index, row in zip(validation, predicted, probabilities, strict=True):
            order = np.argsort(-row, kind="stable")[:3]
            writer.writerow(
                {
                    "sample_id": data_api.sample_id(example, source_revision),
                    "true_intent": example.intent,
                    "predicted_intent": labels[int(pred_index)],
                    "confidence": f"{float(row[int(pred_index)]):.12f}",
                    "top3_intents": "|".join(labels[int(index)] for index in order),
                }
            )


def _token_length_audit(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, object]:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    input_ids = encoded["input_ids"]
    lengths = np.asarray([len(row) for row in input_ids], dtype=np.int64)
    truncated = int(np.sum(lengths > max_length))
    return {
        "max_observed_tokens": int(np.max(lengths)),
        "p99_tokens": float(np.quantile(lengths, 0.99)),
        "max_length": max_length,
        "truncated_examples": truncated,
        "truncation_fraction": float(truncated / len(lengths)),
    }


def _tensor_dataset(
    tokenizer: Any,
    texts: list[str],
    labels: np.ndarray,
    max_length: int,
) -> TensorDataset:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.from_numpy(labels),
    )


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or "LayerNorm.weight" in name:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def _evaluate_model(
    model: MeanPoolClassifier,
    loader: DataLoader[tuple[Tensor, ...]],
    y_true: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, float]:
    model.eval()
    logits_rows: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            logits = model(input_ids, attention_mask)
            logits_rows.append(logits.detach().cpu().numpy())
    elapsed = time.perf_counter() - started
    logits_array = np.concatenate(logits_rows, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    predicted = np.argmax(probabilities, axis=1)
    return _metrics(y_true, predicted, probabilities, labels), predicted, probabilities, elapsed


def _markdown_report(result: dict[str, object]) -> str:
    metrics = result["metrics"]
    comparison = result["comparison_to_a2"]
    assert isinstance(metrics, dict)
    assert isinstance(comparison, dict)
    return "\n".join(
        [
            "# Phase 2 A3 Development Benchmark",
            "",
            (
                "> Validation evidence only. The confirmatory BANKING77 test split was not "
                "downloaded or opened."
            ),
            "",
            "| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |",
            "|---|---:|---:|---:|---:|---:|",
            ("| A3 | {f1:.4f} | {bal:.4f} | {top3:.4f} | {ece:.4f} | {brier:.4f} |").format(
                f1=float(metrics["macro_f1"]),
                bal=float(metrics["balanced_accuracy"]),
                top3=float(metrics["top3_recall"]),
                ece=float(metrics["expected_calibration_error_15bin"]),
                brier=float(metrics["multiclass_brier_score"]),
            ),
            "",
            "## Delta versus audited A2",
            "",
            f"- Macro-F1: {float(comparison['macro_f1_delta']):+.4f}",
            f"- Balanced accuracy: {float(comparison['balanced_accuracy_delta']):+.4f}",
            f"- Top-3 recall: {float(comparison['top3_recall_delta']):+.4f}",
            f"- ECE: {float(comparison['ece_delta']):+.4f}",
            f"- Brier: {float(comparison['brier_delta']):+.4f}",
            "",
            "The fixed three-epoch budget is reported without best-epoch selection.",
            "Wall-clock timing is descriptive and is not release latency evidence.",
            "",
        ]
    )


def run(output_dir: Path) -> dict[str, object]:
    config = json.loads(A3_CONFIG_PATH.read_text(encoding="utf-8"))
    representation = config["representation"]
    training = config["training"]
    seed = int(training["seed"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    data_api = _data_module()
    spec = data_api.Banking77Spec.from_json(DATA_CONFIG_PATH)
    with tempfile.TemporaryDirectory(prefix="helix-phase2-a3-") as temp:
        train_csv = Path(temp) / "train.csv"
        _download(spec.train_url, train_csv)
        if data_api.sha256_file(train_csv) != spec.train_sha256:
            raise ValueError("BANKING77 train checksum drifted")
        source_train = data_api.load_csv(train_csv, "train")

    train, validation, quarantined = data_api.split_training_examples(source_train, spec)
    if len(train) != spec.expected_counts["train"]:
        raise ValueError("derived train count drifted")
    if len(validation) != spec.expected_counts["validation"]:
        raise ValueError("derived validation count drifted")
    if len(quarantined) != spec.expected_counts["quarantine"]:
        raise ValueError("quarantine count drifted")

    train_hash = data_api.sha256_bytes(data_api.canonical_jsonl_bytes(train, spec.source_revision))
    validation_hash = data_api.sha256_bytes(
        data_api.canonical_jsonl_bytes(validation, spec.source_revision)
    )
    if train_hash != spec.expected_hashes["train"]:
        raise ValueError("derived train hash drifted")
    if validation_hash != spec.expected_hashes["validation"]:
        raise ValueError("derived validation hash drifted")

    labels = sorted({example.intent for example in train})
    if len(labels) != spec.intent_count:
        raise ValueError("A3 training labels no longer cover the frozen intent vocabulary")
    label_to_index = {label: index for index, label in enumerate(labels)}
    x_train = [example.text for example in train]
    x_validation = [example.text for example in validation]
    y_train = np.asarray([label_to_index[item.intent] for item in train], dtype=np.int64)
    y_validation = np.asarray(
        [label_to_index[item.intent] for item in validation],
        dtype=np.int64,
    )

    model_id = str(representation["model_id"])
    revision = str(representation["revision"])
    max_length = int(representation["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
    )
    truncation = {
        "train": _token_length_audit(tokenizer, x_train, max_length),
        "validation": _token_length_audit(tokenizer, x_validation, max_length),
    }

    train_dataset = _tensor_dataset(tokenizer, x_train, y_train, max_length)
    validation_dataset = _tensor_dataset(tokenizer, x_validation, y_validation, max_length)
    validation_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
        validation_dataset,
        batch_size=int(training["validation_batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model_load_started = time.perf_counter()
    model = MeanPoolClassifier(
        model_id=model_id,
        revision=revision,
        hidden_dimension=int(representation["hidden_dimension"]),
        num_labels=len(labels),
    )
    model_load_seconds = time.perf_counter() - model_load_started

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    max_epochs = int(training["max_epochs"])
    train_batch_size = int(training["train_batch_size"])
    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = round(float(training["warmup_ratio"]) * total_steps)

    optimizer = torch.optim.AdamW(
        _parameter_groups(model, float(training["weight_decay"])),
        lr=float(training["learning_rate"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_epsilon"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_history: list[dict[str, object]] = []
    final_metrics: dict[str, object] | None = None
    final_predicted: np.ndarray | None = None
    final_probabilities: np.ndarray | None = None
    final_validation_seconds = 0.0
    training_started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_generator = torch.Generator().manual_seed(seed + epoch)
        train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            generator=epoch_generator,
            num_workers=0,
        )
        model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        epoch_started = time.perf_counter()
        for input_ids, attention_mask, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = functional.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            scheduler.step()
            batch_size = int(target.shape[0])
            epoch_loss_sum += float(loss.detach()) * batch_size
            epoch_examples += batch_size

        epoch_training_seconds = time.perf_counter() - epoch_started
        metrics, predicted, probabilities, validation_seconds = _evaluate_model(
            model,
            validation_loader,
            y_validation,
            labels,
        )
        epoch_history.append(
            {
                "epoch": epoch,
                "mean_training_loss": epoch_loss_sum / epoch_examples,
                "validation_macro_f1": metrics["macro_f1"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_top3_recall": metrics["top3_recall"],
                "training_seconds": epoch_training_seconds,
                "validation_seconds": validation_seconds,
            }
        )
        print(
            f"A3 epoch {epoch}/{max_epochs}: "
            f"loss={epoch_loss_sum / epoch_examples:.6f}, "
            f"macro-F1={float(metrics['macro_f1']):.4f}"
        )
        final_metrics = metrics
        final_predicted = predicted
        final_probabilities = probabilities
        final_validation_seconds = validation_seconds

    training_seconds = time.perf_counter() - training_started
    if final_metrics is None or final_predicted is None or final_probabilities is None:
        raise RuntimeError("A3 training produced no final validation result")

    a2_checkpoint = json.loads(A2_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    a2_metrics = a2_checkpoint["metrics"]
    a2_risk_rows = a2_checkpoint["risk_coverage"]
    a2_confusions = a2_checkpoint["principal_a2_confusions"]
    assert isinstance(a2_metrics, dict)
    assert isinstance(a2_risk_rows, list)
    assert isinstance(a2_confusions, list)

    comparison = {
        "macro_f1_delta": float(final_metrics["macro_f1"]) - float(a2_metrics["macro_f1"]),
        "balanced_accuracy_delta": (
            float(final_metrics["balanced_accuracy"])
            - float(a2_metrics["balanced_accuracy"])
        ),
        "top3_recall_delta": (
            float(final_metrics["top3_recall"]) - float(a2_metrics["top3_recall"])
        ),
        "ece_delta": (
            float(final_metrics["expected_calibration_error_15bin"])
            - float(a2_metrics["expected_calibration_error_15bin"])
        ),
        "brier_delta": (
            float(final_metrics["multiclass_brier_score"])
            - float(a2_metrics["multiclass_brier_score"])
        ),
    }
    final_risk_rows = final_metrics["risk_coverage"]
    assert isinstance(final_risk_rows, list)
    risk_comparison = _risk_coverage_delta(a2_risk_rows, final_risk_rows)
    confusion_comparison = _a2_confusion_changes(
        a2_confusions,
        y_validation,
        final_predicted,
        labels,
    )
    hashes = _prediction_hashes(
        validation,
        final_predicted,
        final_probabilities,
        labels,
        spec.source_revision,
        data_api,
    )

    result: dict[str, object] = {
        "run_id": RUN_ID,
        "phase": 2,
        "status": "development_validation_only",
        "test_set_opened": False,
        "selection_partition": "validation",
        "epoch_selection": "none_fixed_final_epoch",
        "data": {
            "dataset_contract": spec.version,
            "source_revision": spec.source_revision,
            "source_train_sha256": spec.train_sha256,
            "derived_train_rows": len(train),
            "derived_validation_rows": len(validation),
            "quarantine_rows": len(quarantined),
            "derived_train_sha256": train_hash,
            "derived_validation_sha256": validation_hash,
        },
        "specification": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "transformers": transformers.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "model": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "model_load_seconds": model_load_seconds,
        },
        "tokenization_audit": truncation,
        "training": {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "training_seconds": training_seconds,
            "epoch_history": epoch_history,
        },
        "metrics": final_metrics,
        "comparison_to_a2": comparison,
        "risk_coverage_comparison_to_a2": risk_comparison,
        "a2_principal_confusion_changes": confusion_comparison,
        "prediction_hashes": hashes,
        "timing_observation": {
            "scope": "GitHub-hosted CPU runner; descriptive only",
            "final_validation_seconds": final_validation_seconds,
            "final_validation_ms_per_example": (
                1000.0 * final_validation_seconds / len(validation)
            ),
        },
        "pending_phase2_evidence": [
            "calibration_comparison",
            "frozen_oos_benchmark",
            "expected_routing_cost",
            "final_risk_coverage_operating_point",
            "router_model_card",
            "route_contract_tests",
            "confirmatory_test",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    _write_predictions(
        output_dir / "a3_validation_predictions.csv",
        validation,
        final_predicted,
        final_probabilities,
        labels,
        spec.source_revision,
        data_api,
    )

    compact = {
        "run_id": RUN_ID,
        "test_set_opened": False,
        "metrics": {
            key: final_metrics[key]
            for key in (
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "top3_recall",
                "expected_calibration_error_15bin",
                "multiclass_brier_score",
                "mean_max_probability",
            )
        },
        "comparison_to_a2": comparison,
        "risk_at_50": risk_comparison[4],
        "risk_at_70": risk_comparison[6],
        "tokenization_audit": truncation,
        "prediction_hashes": hashes,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "total_parameters": total_parameters,
        "training_seconds": training_seconds,
        "final_validation_ms_per_example": 1000.0 * final_validation_seconds / len(validation),
    }
    print("A3_SUMMARY_JSON=" + json.dumps(compact, sort_keys=True))
    print("Confirmatory test opened: false")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "phase2-routing" / "a3",
    )
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()

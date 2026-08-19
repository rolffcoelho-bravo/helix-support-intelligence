from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
A2_CONFIG_PATH = ROOT / "configs" / "models" / "routing_a2.json"
A3_CONFIG_PATH = ROOT / "configs" / "models" / "routing_a3.json"
A3_SCRIPT_PATH = ROOT / "benchmarks" / "routing" / "evaluate_a3.py"


def _load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def test_a3_reuses_a2_transformer_checkpoint() -> None:
    a2 = _load_json(A2_CONFIG_PATH)
    a3 = _load_json(A3_CONFIG_PATH)
    a2_representation = a2["representation"]
    a3_representation = a3["representation"]
    assert isinstance(a2_representation, dict)
    assert isinstance(a3_representation, dict)

    assert a3_representation["model_id"] == a2_representation["model_id"]
    assert a3_representation["revision"] == a2_representation["revision"]
    assert a3_representation["hidden_dimension"] == 384
    assert a3_representation["num_hidden_layers"] == 6
    assert a3_representation["pooling"] == "attention_masked_mean"
    assert a3_representation["normalize_pooled_representation"] is True
    assert a3_representation["fine_tuned"] is True


def test_a3_training_budget_is_fixed_before_evaluation() -> None:
    a3 = _load_json(A3_CONFIG_PATH)
    training = a3["training"]
    anti_shopping = a3["anti_shopping"]
    assert isinstance(training, dict)
    assert isinstance(anti_shopping, dict)

    early_stopping = training["early_stopping"]
    assert isinstance(early_stopping, dict)
    assert training["seed"] == 20260818
    assert training["max_epochs"] == 3
    assert early_stopping["enabled"] is False
    assert training["learning_rate"] == 0.00002
    assert training["train_batch_size"] == 32
    assert training["validation_batch_size"] == 64
    assert training["hyperparameter_search_allowed"] is False
    assert anti_shopping["alternate_transformer_allowed"] is False
    assert anti_shopping["alternate_pooling_allowed"] is False
    assert anti_shopping["alternate_epoch_budget_allowed_after_result"] is False
    assert anti_shopping["hyperparameter_search_allowed"] is False


def test_a3_keeps_confirmatory_test_sealed() -> None:
    a3 = _load_json(A3_CONFIG_PATH)
    evaluation = a3["evaluation"]
    assert isinstance(evaluation, dict)

    assert evaluation["fit_partition"] == "train"
    assert evaluation["selection_partition"] == "validation"
    assert evaluation["confirmatory_partition"] == "test"
    assert evaluation["test_set_may_select_model"] is False
    assert evaluation["test_set_may_select_epoch"] is False
    assert evaluation["test_set_may_select_threshold"] is False

    script = A3_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "spec.test_url" not in script
    assert "Confirmatory test opened: false" in script


def test_a3_declares_cpu_only_locked_environment() -> None:
    a3 = _load_json(A3_CONFIG_PATH)
    dependency_policy = a3["dependency_policy"]
    representation = a3["representation"]
    assert isinstance(dependency_policy, dict)
    assert isinstance(representation, dict)

    assert representation["device"] == "cpu"
    assert dependency_policy["torch"] == "2.13.0"
    assert dependency_policy["torch_variant"] == "cpu_only"
    assert dependency_policy["torch_index"] == "https://download.pytorch.org/whl/cpu"
    assert dependency_policy["transformers"] == "4.57.1"
    assert dependency_policy["script_lock_required_before_benchmark"] is True


def test_a3_primary_comparator_is_audited_a2() -> None:
    a3 = _load_json(A3_CONFIG_PATH)
    evaluation = a3["evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["primary_comparator"] == "phase2-a2-validation-v2"

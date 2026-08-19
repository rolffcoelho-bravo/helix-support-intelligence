from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "retrieval" / "b1_dense_v1.json"
SCRIPT = ROOT / "benchmarks" / "retrieval" / "evaluate_b1.py"
A2_LOCK = ROOT / "benchmarks" / "routing" / "evaluate_a2.py.lock"


def _config() -> dict[str, object]:
    payload: object = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("B1 config must be a JSON object")
    return cast(dict[str, object], payload)


def test_b1_configuration_is_frozen_before_first_score() -> None:
    config = _config()
    model = config["model"]
    environment = config["environment"]
    assert isinstance(model, dict)
    assert isinstance(environment, dict)

    assert config["status"] == "frozen_before_first_score"
    assert config["benchmark_version"] == "retrieval-benchmark-v1"
    assert model["model_id"] == "BAAI/bge-small-en-v1.5"
    assert model["revision"] == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert model["license"] == "MIT"
    assert model["embedding_dimension"] == 384
    assert model["max_sequence_length"] == 512
    assert model["normalize_embeddings"] is True
    assert model["device"] == "cpu"
    assert model["fine_tuning"] is False
    assert model["query_instruction"] == (
        "Represent this sentence for searching relevant passages: "
    )
    assert model["passage_instruction"] == ""
    assert environment["reuse_lock"] == "benchmarks/routing/evaluate_a2.py.lock"
    assert environment["cpu_only"] is True


def test_b1_development_script_refuses_sealed_inputs_and_pins_weight_hash() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'filename="model.safetensors"' in source
    assert "model_safetensors_sha256" in source
    assert '"confirmatory_queries.jsonl"' in source
    assert '"confirmatory_qrels.jsonl"' in source
    assert '"confirmatory_partition_opened": False' in source
    assert '"official_banking77_test_accessed": False' in source
    assert "banking.test_url" not in source
    assert "test.csv" not in source


def test_b1_reused_scientific_lock_is_cpu_only() -> None:
    lock = A2_LOCK.read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in lock
    assert 'name = "torch"' in lock
    lowered = lock.casefold()
    assert 'name = "nvidia-' not in lowered
    assert 'name = "triton"' not in lowered

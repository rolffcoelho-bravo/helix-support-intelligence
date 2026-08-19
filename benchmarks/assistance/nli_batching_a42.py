"""Batched sentence-support evaluation for the A4.2 ONNX verifier/evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np


def batched_support(
    output: dict[str, Any],
    documents: list[dict[str, object]],
    engine: Any,
) -> list[dict[str, Any]]:
    """Evaluate factual sentence pairs in batches of up to the frozen size eight."""
    import evaluate_a42 as runtime

    by_id = {str(row["document_id"]): row for row in documents}
    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, str, str]] = []
    decision = str(output.get("decision", ""))

    for index, sentence in enumerate(
        runtime.sentences(str(output.get("answer", "")))
    ):
        cited = list(
            dict.fromkeys(
                match.group(1)
                for match in runtime.CITATION_RE.finditer(sentence)
            )
        )
        is_factual = runtime.factual(sentence, decision)
        row = {
            "sentence_index": index,
            "text": sentence,
            "citations": cited,
            "factual": is_factual,
            "support_verdict": "NOT_APPLICABLE",
            "entailment_probability": None,
        }
        rows.append(row)
        if not is_factual:
            continue
        usable = [
            by_id[citation]
            for citation in cited
            if citation in by_id and runtime.eligible(by_id[citation])
        ]
        if not cited or len(usable) != len(cited):
            row["support_verdict"] = "UNSUPPORTED"
            continue
        premise = "\n".join(
            f"{document['title']}\n{document['body']}"
            for document in usable
        )
        hypothesis = runtime.CITATION_RE.sub("", sentence).strip()
        pending.append((index, premise, hypothesis))

    batch_size = int(engine.binding.batch_size)
    if batch_size != 8:
        raise RuntimeError("A4.2 requires the frozen A4.1 NLI batch size of eight.")
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        premises = [item[1] for item in chunk]
        hypotheses = [item[2] for item in chunk]
        encoded = engine.tokenizer(
            premises,
            hypotheses,
            max_length=engine.binding.max_length,
            truncation=True,
            padding=True,
            return_tensors="np",
        )
        feed = {
            key: np.asarray(value)
            for key, value in encoded.items()
            if key in engine.input_names
        }
        logits = np.asarray(engine.session.run(None, feed)[0]).astype(float)
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probabilities = exponentiated / np.sum(
            exponentiated,
            axis=1,
            keepdims=True,
        )
        label = int(engine.binding.entailment_label)
        if label < 0 or label >= probabilities.shape[1]:
            raise RuntimeError("Frozen entailment label is outside model logits.")
        for item, probability_row in zip(chunk, probabilities, strict=True):
            sentence_index = item[0]
            probability = float(probability_row[label])
            rows[sentence_index]["entailment_probability"] = probability
            rows[sentence_index]["support_verdict"] = (
                "SUPPORTED"
                if probability >= float(engine.binding.threshold)
                else "UNSUPPORTED"
            )
    return rows

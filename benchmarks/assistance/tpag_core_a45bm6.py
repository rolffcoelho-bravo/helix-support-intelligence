"""Deterministic TPAG core and registered A4.5b-M6 calibration metrics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import combinations
from typing import Any

ALL_SLOTS = (
    "entity_or_subject",
    "predicate_or_event",
    "target_slot_identity",
    "target_value",
    "temporal_scope",
    "location_scope",
    "organizational_scope",
    "conditional_scope",
    "modality_or_quantification",
)
SCOPE_SLOTS = (
    "entity_or_subject",
    "predicate_or_event",
    "target_slot_identity",
    "temporal_scope",
    "location_scope",
    "organizational_scope",
    "conditional_scope",
    "modality_or_quantification",
)
PREDICATE_TO_SLOT = {
    "resolve": "resolution_deadline",
    "acknowledge": "acknowledgement_deadline",
    "review": "review_deadline",
    "release": "release_deadline",
    "verify": "verification_deadline",
    "escalate": "escalation_deadline",
    "record": "recording_deadline",
    "finalize": "finalization_deadline",
}
KNOWN_PREDICATES = frozenset(PREDICATE_TO_SLOT)
MODALITY_EQUIVALENTS = frozenset(
    {
        "must process every",
        "must process all",
        "is required to process every",
        "is required to process all",
    }
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SUBJECT_RE = re.compile(r"\b(Orchid request \d{3}|OR-\d{3})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\bDuring\s+(20\d{2}),", re.IGNORECASE)
ORG_RE = re.compile(r"\b([A-Z][a-z]+ desk)\b")
LOCATION_RE = re.compile(r"\b(?:in the|location is)\s+([a-z]+ district)\b", re.IGNORECASE)
VALUE_RE = re.compile(r"\b(?:within|is)\s+(\d+)\s+business days\b", re.IGNORECASE)
EXPLICIT_SLOT_RE = re.compile(
    r"\b([a-z]+_(?:deadline|retention))\b",
    re.IGNORECASE,
)
ACTIVE_RECORD_RE = re.compile(r"The active record is (Orchid request \d{3})\.", re.IGNORECASE)
ALIAS_DECL_RE = re.compile(
    r"(Orchid request \d{3}) is referenced internally as (OR-\d{3})\.",
    re.IGNORECASE,
)


def _normal(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _clean_text(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _canonical_subject(value: str, aliases: dict[str, str]) -> str:
    normalized = _normal(value)
    canonical = aliases.get(normalized, value)
    match = re.fullmatch(r"orchid request (\d{3})", _normal(canonical))
    if match:
        return f"Orchid request {match.group(1)}"
    return str(canonical)


def make_alias_map(units: list[dict[str, Any]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for unit in units:
        subject = str(unit["subject"])
        alias = str(unit["subject_alias"])
        aliases[_normal(subject)] = subject
        aliases[_normal(alias)] = subject
    return aliases


def _document_aliases(document_text: str, aliases: dict[str, str]) -> dict[str, str]:
    result = dict(aliases)
    for match in ALIAS_DECL_RE.finditer(document_text):
        canonical = _canonical_subject(match.group(1), aliases)
        result[_normal(match.group(2))] = canonical
    return result


def _active_subject_before(sentences: list[str], index: int, aliases: dict[str, str]) -> str | None:
    for sentence in reversed(sentences[:index]):
        match = ACTIVE_RECORD_RE.search(sentence)
        if match:
            return _canonical_subject(match.group(1), aliases)
    return None


def decontextualize_sentence(
    sentence: str,
    aliases: dict[str, str],
    active_subject: str | None,
) -> str:
    text = _clean_text(sentence)
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if alias.startswith("or-"):
            text = re.sub(
                rf"\b{re.escape(alias)}\b",
                canonical.lower(),
                text,
                flags=re.IGNORECASE,
            )
    if active_subject is not None:
        text = re.sub(r"\bit records\b", f"{active_subject.lower()} records", text, count=1)
    return _clean_text(text)


def _extract_predicate(text: str) -> tuple[str | None, str | None]:
    patterns = (
        r"\bgoverning action is\s+([a-z]+(?: [a-z]+ handling)?)\b",
        r"\band\s+([a-z]+(?: [a-z]+ handling)?)\s+them within\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw = _normal(match.group(1))
            inner = None
            paraphrase = re.fullmatch(r"complete ([a-z]+) handling", raw)
            if paraphrase:
                inner = paraphrase.group(1)
            return raw, inner
    archive = re.search(r"\bmust archive every\b", text, flags=re.IGNORECASE)
    if archive:
        return "archive", None
    return None, None


def parse_frame(text: str, aliases: dict[str, str]) -> dict[str, Any]:
    clean = _clean_text(text)
    frame: dict[str, Any] = {slot: None for slot in ALL_SLOTS}
    explicit_slots: set[str] = set()

    subject_match = SUBJECT_RE.search(clean)
    if subject_match:
        frame["entity_or_subject"] = _canonical_subject(subject_match.group(1), aliases)
        explicit_slots.add("entity_or_subject")

    year_match = YEAR_RE.search(clean)
    if year_match:
        frame["temporal_scope"] = int(year_match.group(1))
        explicit_slots.add("temporal_scope")

    org_match = ORG_RE.search(clean)
    if org_match:
        frame["organizational_scope"] = org_match.group(1)
        explicit_slots.add("organizational_scope")

    location_match = LOCATION_RE.search(clean)
    if location_match:
        frame["location_scope"] = _normal(location_match.group(1))
        explicit_slots.add("location_scope")

    condition_match = re.search(r"\bwhen\s+([^.]+)", clean, flags=re.IGNORECASE)
    if condition_match:
        frame["conditional_scope"] = _normal(condition_match.group(1))
        explicit_slots.add("conditional_scope")

    for modality in sorted(MODALITY_EQUIVALENTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(modality)}\b", clean, flags=re.IGNORECASE):
            frame["modality_or_quantification"] = modality
            explicit_slots.add("modality_or_quantification")
            break
    if frame["modality_or_quantification"] is None and re.search(
        r"\bmay process some\b", clean, flags=re.IGNORECASE
    ):
        frame["modality_or_quantification"] = "may process some"
        explicit_slots.add("modality_or_quantification")

    value_match = VALUE_RE.search(clean)
    if value_match:
        frame["target_value"] = int(value_match.group(1))
        explicit_slots.add("target_value")

    raw_predicate, inner_predicate = _extract_predicate(clean)
    if raw_predicate is not None:
        frame["predicate_or_event"] = raw_predicate
        explicit_slots.add("predicate_or_event")

    slot_match = EXPLICIT_SLOT_RE.search(clean)
    if slot_match:
        frame["target_slot_identity"] = _normal(slot_match.group(1))
        explicit_slots.add("target_slot_identity")
    elif raw_predicate in PREDICATE_TO_SLOT:
        frame["target_slot_identity"] = PREDICATE_TO_SLOT[str(raw_predicate)]
        explicit_slots.add("target_slot_identity")
    elif inner_predicate in PREDICATE_TO_SLOT:
        frame["target_slot_identity"] = PREDICATE_TO_SLOT[str(inner_predicate)]
        explicit_slots.add("target_slot_identity")
    elif re.search(r"\baudit note\b", clean, flags=re.IGNORECASE):
        frame["target_slot_identity"] = "audit_note_retention"
        explicit_slots.add("target_slot_identity")

    frame["_raw_predicate"] = raw_predicate
    frame["_inner_predicate"] = inner_predicate
    frame["_explicit_slots"] = sorted(explicit_slots)
    return frame


def parse_span(text: str, aliases: dict[str, str]) -> dict[str, Any]:
    parts = SENTENCE_SPLIT_RE.split(text.strip()) if text.strip() else []
    frames = [parse_frame(part, aliases) for part in parts]
    merged: dict[str, Any] = {slot: None for slot in ALL_SLOTS}
    explicit_slots: set[str] = set()
    raw_predicates: list[str] = []
    for frame in frames:
        for slot in ALL_SLOTS:
            value = frame[slot]
            if value is not None and merged[slot] is None:
                merged[slot] = value
        explicit_slots.update(str(slot) for slot in frame["_explicit_slots"])
        raw = frame.get("_raw_predicate")
        if raw is not None:
            raw_predicates.append(str(raw))
    merged["_raw_predicate"] = raw_predicates[0] if raw_predicates else None
    merged["_inner_predicate"] = None
    merged["_explicit_slots"] = sorted(explicit_slots)
    return merged


def _frame_public(frame: dict[str, Any]) -> dict[str, Any]:
    return {slot: frame.get(slot) for slot in ALL_SLOTS}


def predict_proposition_case(row: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    document_text = str(row["document_text"])
    sentences = SENTENCE_SPLIT_RE.split(document_text.strip()) if document_text.strip() else []
    local_aliases = _document_aliases(document_text, aliases)
    claim_frame = parse_frame(str(row["claim"]), aliases)
    targets: list[int] = []
    decontextualized: list[str] = []
    target_frames: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        active_subject = _active_subject_before(sentences, index, local_aliases)
        candidate = decontextualize_sentence(sentence, local_aliases, active_subject)
        frame = parse_frame(candidate, local_aliases)
        if frame["entity_or_subject"] != claim_frame["entity_or_subject"]:
            continue
        informative = sum(
            frame[slot] is not None
            for slot in (
                "predicate_or_event",
                "target_slot_identity",
                "target_value",
                "temporal_scope",
                "location_scope",
                "organizational_scope",
                "conditional_scope",
                "modality_or_quantification",
            )
        )
        if informative < 4:
            continue
        targets.append(index)
        decontextualized.append(candidate)
        target_frames.append(_frame_public(frame))
    return {
        "proposition_case_id": str(row["proposition_case_id"]),
        "surface_propositions": sentences,
        "target_proposition_indices": targets,
        "decontextualized_target_propositions": decontextualized,
        "target_frames": target_frames,
    }


def residual_request_key(claim_value: str, evidence_value: str, evidence_text: str) -> str:
    payload = json.dumps(
        [claim_value, evidence_value, evidence_text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def residual_request(
    claim_frame: dict[str, Any],
    evidence_frame: dict[str, Any],
    evidence_text: str,
) -> dict[str, str] | None:
    claim_value = claim_frame.get("predicate_or_event")
    evidence_value = evidence_frame.get("predicate_or_event")
    if claim_value is None or evidence_value is None:
        return None
    claim_text = str(claim_value)
    evidence_text_value = str(evidence_value)
    if _normal(claim_text) == _normal(evidence_text_value):
        return None
    if _normal(claim_text) in KNOWN_PREDICATES and _normal(evidence_text_value) in KNOWN_PREDICATES:
        return None
    key = residual_request_key(claim_text, evidence_text_value, evidence_text)
    return {
        "request_id": key,
        "claim_value": claim_text,
        "evidence_value": evidence_text_value,
        "evidence_text": evidence_text,
    }


def collect_residual_requests(suite: dict[str, Any], aliases: dict[str, str]) -> list[dict[str, str]]:
    requests: dict[str, dict[str, str]] = {}
    for row in suite["alignment_rows"]:
        claim_frame = parse_frame(str(row["claim"]), aliases)
        evidence_text = str(row["evidence_proposition"])
        evidence_frame = parse_span(evidence_text, aliases)
        request = residual_request(claim_frame, evidence_frame, evidence_text)
        if request is not None:
            requests[request["request_id"]] = request
    for row in suite["evidence_group_rows"]:
        claim_frame = parse_frame(str(row["claim"]), aliases)
        for evidence in row["evidence_propositions"]:
            evidence_text = str(evidence)
            evidence_frame = parse_span(evidence_text, aliases)
            request = residual_request(claim_frame, evidence_frame, evidence_text)
            if request is not None:
                requests[request["request_id"]] = request
    return [requests[key] for key in sorted(requests)]


def _residual_relation(
    claim_frame: dict[str, Any],
    evidence_frame: dict[str, Any],
    evidence_text: str,
    raw_scores: dict[str, dict[str, float]],
    threshold: float,
) -> str:
    claim_value = claim_frame.get("predicate_or_event")
    evidence_value = evidence_frame.get("predicate_or_event")
    if claim_value is None or evidence_value is None:
        return "UNSPECIFIED"
    claim_text = _normal(str(claim_value))
    evidence_text_value = _normal(str(evidence_value))
    if claim_text == evidence_text_value:
        return "MATCH"
    if claim_text in KNOWN_PREDICATES and evidence_text_value in KNOWN_PREDICATES:
        return "MISMATCH"
    key = residual_request_key(str(claim_value), str(evidence_value), evidence_text)
    scores = raw_scores.get(key)
    if scores is None:
        return "UNSPECIFIED"
    if float(scores["entailment"]) >= threshold:
        return "MATCH"
    if float(scores["contradiction"]) >= threshold:
        return "MISMATCH"
    return "UNSPECIFIED"


def _simple_relation(claim: Any, evidence: Any) -> str:
    if evidence is None:
        return "UNSPECIFIED"
    if claim is None:
        return "UNSPECIFIED"
    return "MATCH" if _normal(str(claim)) == _normal(str(evidence)) else "MISMATCH"


def _modality_relation(claim: Any, evidence: Any) -> str:
    if evidence is None:
        return "UNSPECIFIED"
    if claim is None:
        return "UNSPECIFIED"
    claim_text = _normal(str(claim))
    evidence_text = _normal(str(evidence))
    if claim_text in MODALITY_EQUIVALENTS and evidence_text in MODALITY_EQUIVALENTS:
        return "MATCH"
    return "MATCH" if claim_text == evidence_text else "MISMATCH"


def align_frames(
    claim_frame: dict[str, Any],
    evidence_frame: dict[str, Any],
    evidence_text: str,
    raw_scores: dict[str, dict[str, float]],
    threshold: float,
) -> dict[str, Any]:
    relations: dict[str, str] = {}
    relations["entity_or_subject"] = _simple_relation(
        claim_frame.get("entity_or_subject"), evidence_frame.get("entity_or_subject")
    )
    relations["predicate_or_event"] = _residual_relation(
        claim_frame, evidence_frame, evidence_text, raw_scores, threshold
    )
    relations["target_slot_identity"] = _simple_relation(
        claim_frame.get("target_slot_identity"), evidence_frame.get("target_slot_identity")
    )
    relations["target_value"] = _simple_relation(
        claim_frame.get("target_value"), evidence_frame.get("target_value")
    )
    relations["temporal_scope"] = _simple_relation(
        claim_frame.get("temporal_scope"), evidence_frame.get("temporal_scope")
    )
    relations["location_scope"] = _simple_relation(
        claim_frame.get("location_scope"), evidence_frame.get("location_scope")
    )
    relations["organizational_scope"] = _simple_relation(
        claim_frame.get("organizational_scope"), evidence_frame.get("organizational_scope")
    )
    relations["conditional_scope"] = _simple_relation(
        claim_frame.get("conditional_scope"), evidence_frame.get("conditional_scope")
    )
    relations["modality_or_quantification"] = _modality_relation(
        claim_frame.get("modality_or_quantification"),
        evidence_frame.get("modality_or_quantification"),
    )
    incompatible = any(relations[slot] == "MISMATCH" for slot in SCOPE_SLOTS)
    compatibility = "INCOMPATIBLE" if incompatible else "COMPATIBLE"
    covered = {
        slot
        for slot, relation in relations.items()
        if relation == "MATCH" or (slot == "target_value" and relation == "MISMATCH")
    }
    missing = set(ALL_SLOTS) - covered
    if compatibility == "INCOMPATIBLE":
        coverage = "NOT_APPLICABLE"
        polarity = "NOT_APPLICABLE"
        relation = "UNKNOWN"
    elif missing:
        coverage = "INCOMPLETE"
        polarity = "UNRESOLVED"
        relation = "UNKNOWN"
    else:
        coverage = "COMPLETE"
        if relations["target_value"] == "MISMATCH":
            polarity = "REFUTES"
            relation = "CONTRADICTED"
        else:
            polarity = "SUPPORTS"
            relation = "ENTAILED"
    return {
        "slot_relations": relations,
        "scope_compatibility": compatibility,
        "coverage_status": coverage,
        "covered_slots": sorted(covered),
        "missing_slots": sorted(missing),
        "polarity": polarity,
        "final_relation": relation,
    }


def predict_alignment_case(
    row: dict[str, Any],
    aliases: dict[str, str],
    raw_scores: dict[str, dict[str, float]],
    threshold: float,
) -> dict[str, Any]:
    claim_frame = parse_frame(str(row["claim"]), aliases)
    evidence_text = str(row["evidence_proposition"])
    evidence_frame = parse_span(evidence_text, aliases)
    prediction = align_frames(claim_frame, evidence_frame, evidence_text, raw_scores, threshold)
    prediction["alignment_id"] = str(row["alignment_id"])
    return prediction


def _subset_prediction(
    indices: tuple[int, ...],
    span_predictions: list[dict[str, Any]],
    incoherent_slots: set[str],
) -> tuple[bool, str]:
    covered: set[str] = set()
    target_value_mismatch = False
    for index in indices:
        prediction = span_predictions[index]
        if prediction["scope_compatibility"] != "COMPATIBLE":
            return False, "UNRESOLVED"
        covered.update(str(slot) for slot in prediction["covered_slots"])
        if prediction["slot_relations"]["target_value"] == "MISMATCH":
            target_value_mismatch = True
    covered.difference_update(incoherent_slots)
    if set(ALL_SLOTS) - covered:
        return False, "UNRESOLVED"
    return True, "REFUTES" if target_value_mismatch else "SUPPORTS"


def predict_group_case(
    row: dict[str, Any],
    aliases: dict[str, str],
    raw_scores: dict[str, dict[str, float]],
    threshold: float,
) -> dict[str, Any]:
    claim_frame = parse_frame(str(row["claim"]), aliases)
    evidence_texts = [str(value) for value in row["evidence_propositions"]]
    span_frames = [parse_span(text, aliases) for text in evidence_texts]
    span_predictions = [
        align_frames(claim_frame, frame, text, raw_scores, threshold)
        for frame, text in zip(span_frames, evidence_texts, strict=True)
    ]

    incoherent_slots: set[str] = set()
    for frame, prediction in zip(span_frames, span_predictions, strict=True):
        explicit = set(str(slot) for slot in frame["_explicit_slots"] if slot in SCOPE_SLOTS)
        mismatches = {
            slot
            for slot in SCOPE_SLOTS
            if prediction["slot_relations"].get(slot) == "MISMATCH"
        }
        if len(explicit) == 1 and len(mismatches) == 1 and explicit == mismatches:
            incoherent_slots.update(mismatches)

    compatible_indices = [
        index
        for index, prediction in enumerate(span_predictions)
        if prediction["scope_compatibility"] == "COMPATIBLE"
    ]
    covered: set[str] = set()
    for index in compatible_indices:
        covered.update(str(slot) for slot in span_predictions[index]["covered_slots"])
    covered.difference_update(incoherent_slots)
    missing = set(ALL_SLOTS) - covered

    sufficient: list[tuple[tuple[int, ...], str]] = []
    for size in range(1, len(compatible_indices) + 1):
        for subset in combinations(compatible_indices, size):
            is_sufficient, polarity = _subset_prediction(
                subset, span_predictions, incoherent_slots
            )
            if is_sufficient:
                sufficient.append((subset, polarity))

    minimal: list[tuple[tuple[int, ...], str]] = []
    for subset, polarity in sufficient:
        subset_set = set(subset)
        if any(set(other).issubset(subset_set) and len(other) < len(subset) for other, _ in sufficient):
            continue
        minimal.append((subset, polarity))
    minimal.sort(key=lambda item: (len(item[0]), item[0], item[1]))

    has_support = any(polarity == "SUPPORTS" for _, polarity in minimal)
    has_refute = any(polarity == "REFUTES" for _, polarity in minimal)
    coherence = "INCOHERENT" if incoherent_slots else "COHERENT"
    if has_support and has_refute:
        sufficiency = "CONFLICTING"
        polarity = "CONFLICTING"
        final_relation = "CONFLICTING_EVIDENCE"
    elif minimal:
        sufficiency = "SUFFICIENT"
        polarity = "REFUTES" if has_refute else "SUPPORTS"
        final_relation = "CONTRADICTED" if has_refute else "ENTAILED"
    else:
        sufficiency = "INSUFFICIENT"
        polarity = "UNRESOLVED"
        final_relation = "UNKNOWN"

    return {
        "group_id": str(row["group_id"]),
        "covered_decisive_slots": sorted(covered),
        "missing_decisive_slots": sorted(missing),
        "minimal_sufficient_groups_zero_based": [list(subset) for subset, _ in minimal],
        "cross_proposition_scope_coherence": coherence,
        "sufficiency": sufficiency,
        "polarity": polarity,
        "final_relation": final_relation,
        "span_predictions": span_predictions,
    }


def predict_claim_case(row: dict[str, Any]) -> dict[str, Any]:
    gate = str(row["deterministic_gate"])
    relations = [str(value) for value in row["group_relations"]]
    if gate == "CITATION_INVALID":
        verdict = "CITATION_INVALID"
    elif gate == "STALE_EVIDENCE":
        verdict = "STALE_EVIDENCE"
    elif gate == "REGISTERED_CONFLICT":
        verdict = "CONFLICTING_EVIDENCE"
    elif "CONFLICTING_EVIDENCE" in relations:
        verdict = "CONFLICTING_EVIDENCE"
    elif "ENTAILED" in relations:
        verdict = "SUPPORTED"
    else:
        verdict = "UNSUPPORTED"
    return {"case_id": str(row["case_id"]), "verdict": verdict}


def predict_all(
    suite: dict[str, Any],
    raw_scores: dict[str, dict[str, float]],
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    aliases = make_alias_map(suite["units"])
    return {
        "propositions": [predict_proposition_case(row, aliases) for row in suite["proposition_rows"]],
        "alignments": [
            predict_alignment_case(row, aliases, raw_scores, threshold)
            for row in suite["alignment_rows"]
        ],
        "groups": [
            predict_group_case(row, aliases, raw_scores, threshold)
            for row in suite["evidence_group_rows"]
        ],
        "claims": [predict_claim_case(row) for row in suite["claim_rows"]],
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _recall(gold: list[str], predicted: list[str], label: str) -> float:
    positives = sum(value == label for value in gold)
    true_positive = sum(
        gold_value == label and predicted_value == label
        for gold_value, predicted_value in zip(gold, predicted, strict=True)
    )
    return _safe_div(float(true_positive), float(positives))


def _precision(gold: list[str], predicted: list[str], label: str) -> float:
    predicted_positive = sum(value == label for value in predicted)
    true_positive = sum(
        gold_value == label and predicted_value == label
        for gold_value, predicted_value in zip(gold, predicted, strict=True)
    )
    return _safe_div(float(true_positive), float(predicted_positive))


def _f1(gold: list[str], predicted: list[str], label: str) -> float:
    precision = _precision(gold, predicted, label)
    recall = _recall(gold, predicted, label)
    return _safe_div(2.0 * precision * recall, precision + recall)


def _macro_f1(gold: list[str], predicted: list[str], labels: tuple[str, ...]) -> float:
    return sum(_f1(gold, predicted, label) for label in labels) / len(labels)


def _accuracy(gold: list[str], predicted: list[str]) -> float:
    return _safe_div(
        float(sum(a == b for a, b in zip(gold, predicted, strict=True))),
        float(len(gold)),
    )


def _subtype_rows(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]], subtype: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (row, prediction)
        for row, prediction in zip(rows, predictions, strict=True)
        if str(row["subtype"]) == subtype
    ]


def _rate(pairs: list[tuple[dict[str, Any], dict[str, Any]]], predicate: Any) -> float:
    return _safe_div(float(sum(bool(predicate(row, prediction)) for row, prediction in pairs)), float(len(pairs)))


def _proposition_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    surface_tp = 0
    surface_gold = 0
    surface_pred = 0
    target_tp = 0
    target_gold = 0
    target_pred = 0
    frame_correct = 0
    frame_total = 0
    no_target_total = 0
    no_target_false = 0
    for row, prediction in zip(rows, predictions, strict=True):
        gold = row["gold"]
        gold_surface = list(gold["surface_propositions"])
        pred_surface = list(prediction["surface_propositions"])
        surface_tp += sum(value in gold_surface for value in pred_surface)
        surface_gold += len(gold_surface)
        surface_pred += len(pred_surface)
        gold_targets = [int(value) for value in gold["target_proposition_indices"]]
        pred_targets = [int(value) for value in prediction["target_proposition_indices"]]
        target_tp += len(set(gold_targets) & set(pred_targets))
        target_gold += len(gold_targets)
        target_pred += len(pred_targets)
        gold_frames = list(gold["target_frames"])
        pred_frames = list(prediction["target_frames"])
        count = max(len(gold_frames), len(pred_frames))
        for index in range(count):
            for slot in ALL_SLOTS:
                frame_total += 1
                gold_value = gold_frames[index].get(slot) if index < len(gold_frames) else None
                pred_value = pred_frames[index].get(slot) if index < len(pred_frames) else None
                if gold_value == pred_value:
                    frame_correct += 1
        if not gold_targets:
            no_target_total += 1
            if pred_targets:
                no_target_false += 1
    proposition_precision = _safe_div(float(surface_tp), float(surface_pred))
    proposition_recall = _safe_div(float(surface_tp), float(surface_gold))
    target_precision = _safe_div(float(target_tp), float(target_pred))
    target_recall = _safe_div(float(target_tp), float(target_gold))
    target_f1 = _safe_div(
        2.0 * target_precision * target_recall,
        target_precision + target_recall,
    )
    return {
        "proposition_precision": proposition_precision,
        "proposition_recall": proposition_recall,
        "target_proposition_f1": target_f1,
        "decontextualized_target_frame_slot_accuracy": _safe_div(
            float(frame_correct), float(frame_total)
        ),
        "no_target_false_positive_rate": _safe_div(float(no_target_false), float(no_target_total)),
    }


def _alignment_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    gold_slots: list[str] = []
    pred_slots: list[str] = []
    for row, prediction in zip(rows, predictions, strict=True):
        for slot in ALL_SLOTS:
            gold_slots.append(str(row["gold"]["slot_relations"][slot]))
            pred_slots.append(str(prediction["slot_relations"][slot]))
    gold_compat = [str(row["gold"]["scope_compatibility"]) for row in rows]
    pred_compat = [str(value["scope_compatibility"]) for value in predictions]
    gold_polarity: list[str] = []
    pred_polarity: list[str] = []
    for row, prediction in zip(rows, predictions, strict=True):
        if str(row["gold"]["scope_compatibility"]) == "COMPATIBLE":
            gold_polarity.append(str(row["gold"]["polarity"]))
            pred_polarity.append(str(prediction["polarity"]))
    gold_relation = [str(row["gold"]["final_relation"]) for row in rows]
    pred_relation = [str(value["final_relation"]) for value in predictions]

    metrics = {
        "slot_relation_macro_f1": _macro_f1(
            gold_slots, pred_slots, ("MATCH", "MISMATCH", "UNSPECIFIED")
        ),
        "scope_compatibility_macro_f1": _macro_f1(
            gold_compat, pred_compat, ("COMPATIBLE", "INCOMPATIBLE")
        ),
        "compatible_recall": _recall(gold_compat, pred_compat, "COMPATIBLE"),
        "incompatible_recall": _recall(gold_compat, pred_compat, "INCOMPATIBLE"),
        "polarity_macro_f1": _macro_f1(
            gold_polarity, pred_polarity, ("SUPPORTS", "REFUTES", "UNRESOLVED")
        ),
        "supports_recall": _recall(gold_polarity, pred_polarity, "SUPPORTS"),
        "refutes_recall": _recall(gold_polarity, pred_polarity, "REFUTES"),
        "final_relation_macro_f1": _macro_f1(
            gold_relation, pred_relation, ("ENTAILED", "CONTRADICTED", "UNKNOWN")
        ),
        "entailed_recall": _recall(gold_relation, pred_relation, "ENTAILED"),
        "contradicted_recall": _recall(gold_relation, pred_relation, "CONTRADICTED"),
        "unknown_recall": _recall(gold_relation, pred_relation, "UNKNOWN"),
    }

    rejection_subtypes = {
        "entity_mismatch_rejection": "entity_mismatch",
        "predicate_mismatch_rejection": "predicate_mismatch",
        "target_slot_identity_mismatch_rejection": "target_slot_identity_mismatch",
        "temporal_scope_mismatch_rejection": "temporal_scope_mismatch",
        "location_scope_mismatch_rejection": "location_scope_mismatch",
        "organizational_scope_mismatch_rejection": "organizational_scope_mismatch",
        "conditional_scope_mismatch_rejection": "conditional_scope_mismatch",
        "modality_quantification_mismatch_rejection": "modality_quantification_mismatch",
        "same_domain_near_miss_rejection": "same_domain_near_miss",
        "cross_unit_distractor_rejection": "cross_unit_distractor",
    }
    for metric_name, subtype in rejection_subtypes.items():
        metrics[metric_name] = _rate(
            _subtype_rows(rows, predictions, subtype),
            lambda _row, prediction: prediction["scope_compatibility"] == "INCOMPATIBLE",
        )

    preservation_subtypes = {
        "missing_target_value_stays_compatible": "missing_target_value",
        "missing_temporal_scope_stays_compatible": "missing_temporal_scope",
        "missing_conditional_scope_stays_compatible": "missing_conditional_scope",
        "missing_location_scope_stays_compatible": "missing_location_scope",
        "missing_organizational_scope_stays_compatible": "missing_organizational_scope",
        "missing_modality_quantification_stays_compatible": "missing_modality_quantification",
    }
    for metric_name, subtype in preservation_subtypes.items():
        metrics[metric_name] = _rate(
            _subtype_rows(rows, predictions, subtype),
            lambda _row, prediction: prediction["scope_compatibility"] == "COMPATIBLE",
        )

    alias_pairs = _subtype_rows(rows, predictions, "explicit_entity_alias_match")
    metrics["explicit_entity_alias_match_recall"] = _rate(
        alias_pairs,
        lambda _row, prediction: prediction["slot_relations"]["entity_or_subject"] == "MATCH",
    )
    paraphrase_pairs = _subtype_rows(rows, predictions, "predicate_paraphrase_match")
    metrics["predicate_paraphrase_match_recall"] = _rate(
        paraphrase_pairs,
        lambda _row, prediction: prediction["slot_relations"]["predicate_or_event"] == "MATCH",
    )
    value_conflicts = _subtype_rows(rows, predictions, "direct_refutation_value_conflict")
    metrics["target_value_conflict_stays_scope_compatible"] = _rate(
        value_conflicts,
        lambda _row, prediction: prediction["scope_compatibility"] == "COMPATIBLE",
    )
    metrics["target_value_conflict_refutation_recall"] = _rate(
        value_conflicts,
        lambda _row, prediction: prediction["final_relation"] == "CONTRADICTED",
    )
    return metrics


def _group_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    gold_sufficiency = [str(row["gold"]["sufficiency"]) for row in rows]
    pred_sufficiency = [str(value["sufficiency"]) for value in predictions]
    covered_exact = []
    missing_exact = []
    minimal_exact = []
    for row, prediction in zip(rows, predictions, strict=True):
        covered_exact.append(
            sorted(str(value) for value in row["gold"]["covered_decisive_slots"])
            == sorted(str(value) for value in prediction["covered_decisive_slots"])
        )
        missing_exact.append(
            sorted(str(value) for value in row["gold"]["missing_decisive_slots"])
            == sorted(str(value) for value in prediction["missing_decisive_slots"])
        )
        minimal_exact.append(
            sorted([list(value) for value in row["gold"]["minimal_sufficient_groups_zero_based"]])
            == sorted([list(value) for value in prediction["minimal_sufficient_groups_zero_based"]])
        )
    metrics = {
        "sufficiency_macro_f1": _macro_f1(
            gold_sufficiency,
            pred_sufficiency,
            ("SUFFICIENT", "INSUFFICIENT", "CONFLICTING"),
        ),
        "sufficient_recall": _recall(gold_sufficiency, pred_sufficiency, "SUFFICIENT"),
        "insufficient_recall": _recall(gold_sufficiency, pred_sufficiency, "INSUFFICIENT"),
        "covered_slot_exact_set_accuracy": _safe_div(
            float(sum(covered_exact)), float(len(covered_exact))
        ),
        "missing_slot_exact_set_accuracy": _safe_div(
            float(sum(missing_exact)), float(len(missing_exact))
        ),
        "minimal_sufficient_group_exact_match": _safe_div(
            float(sum(minimal_exact)), float(len(minimal_exact))
        ),
    }
    subtype_metrics = {
        "complementary_two_span_support_recall": (
            "complementary_two_span_support",
            lambda prediction: prediction["final_relation"] == "ENTAILED",
        ),
        "complementary_three_span_support_recall": (
            "complementary_three_span_support",
            lambda prediction: prediction["final_relation"] == "ENTAILED",
        ),
        "irrelevant_distractor_robustness": (
            "support_with_cross_unit_distractor",
            lambda prediction: prediction["final_relation"] == "ENTAILED",
        ),
        "cross_span_scope_incoherence_insufficiency_recall": (
            "cross_span_scope_incoherence",
            lambda prediction: prediction["sufficiency"] == "INSUFFICIENT",
        ),
        "same_scope_conflict_accuracy": (
            "same_scope_support_refute_conflict",
            lambda prediction: prediction["final_relation"] == "CONFLICTING_EVIDENCE",
        ),
    }
    for metric_name, (subtype, predicate) in subtype_metrics.items():
        pairs = _subtype_rows(rows, predictions, subtype)
        metrics[metric_name] = _rate(
            pairs, lambda _row, prediction, predicate=predicate: predicate(prediction)
        )
    different_condition = _subtype_rows(rows, predictions, "different_condition_not_conflict")
    metrics["different_condition_false_conflict_rate"] = _rate(
        different_condition,
        lambda _row, prediction: prediction["final_relation"] == "CONFLICTING_EVIDENCE",
    )
    gold_relation = [str(row["gold"]["final_relation"]) for row in rows]
    pred_relation = [str(value["final_relation"]) for value in predictions]
    metrics["conflicting_evidence_recall"] = _recall(
        gold_relation, pred_relation, "CONFLICTING_EVIDENCE"
    )
    return metrics


def _claim_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    categories = sorted({str(row["category"]) for row in rows})
    category_accuracy = []
    for category in categories:
        pairs = [
            (row, prediction)
            for row, prediction in zip(rows, predictions, strict=True)
            if str(row["category"]) == category
        ]
        category_accuracy.append(
            _rate(
                pairs,
                lambda row, prediction: prediction["verdict"] == row["expected_verdict"],
            )
        )
    gold = [str(row["expected_verdict"]) for row in rows]
    pred = [str(value["verdict"]) for value in predictions]
    safety_rows = [
        (row, prediction)
        for row, prediction in zip(rows, predictions, strict=True)
        if str(row["expected_verdict"]) != "SUPPORTED"
    ]
    return {
        "claim_category_macro_accuracy": sum(category_accuracy) / len(category_accuracy),
        "supported_precision": _precision(gold, pred, "SUPPORTED"),
        "supported_recall": _recall(gold, pred, "SUPPORTED"),
        "false_supported_safety": _rate(
            safety_rows,
            lambda _row, prediction: prediction["verdict"] == "SUPPORTED",
        ),
        "citation_invalid_accuracy": _rate(
            [
                pair
                for pair in zip(rows, predictions, strict=True)
                if str(pair[0]["category"]) == "citation_invalid"
            ],
            lambda row, prediction: prediction["verdict"] == row["expected_verdict"],
        ),
        "stale_evidence_accuracy": _rate(
            [
                pair
                for pair in zip(rows, predictions, strict=True)
                if str(pair[0]["category"]) == "stale_evidence"
            ],
            lambda row, prediction: prediction["verdict"] == row["expected_verdict"],
        ),
        "registered_conflict_accuracy": _rate(
            [
                pair
                for pair in zip(rows, predictions, strict=True)
                if str(pair[0]["category"]) == "registered_conflict"
            ],
            lambda row, prediction: prediction["verdict"] == row["expected_verdict"],
        ),
    }


def evaluate_predictions(
    suite: dict[str, Any], predictions: dict[str, list[dict[str, Any]]]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(_proposition_metrics(suite["proposition_rows"], predictions["propositions"]))
    metrics.update(_alignment_metrics(suite["alignment_rows"], predictions["alignments"]))
    metrics.update(_group_metrics(suite["evidence_group_rows"], predictions["groups"]))
    metrics.update(_claim_metrics(suite["claim_rows"], predictions["claims"]))
    return metrics


def requirement_results(
    metrics: dict[str, float], requirements: dict[str, float]
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for name, floor in requirements.items():
        metric_name = name.removesuffix("_min").removesuffix("_max")
        value = float(metrics[metric_name])
        if name.endswith("_max"):
            results[name] = value <= float(floor) + 1e-12
        else:
            results[name] = value + 1e-12 >= float(floor)
    return results


def candidate_record(
    suite: dict[str, Any],
    raw_scores: dict[str, dict[str, float]],
    threshold: float,
    requirements: dict[str, float],
) -> dict[str, Any]:
    predictions = predict_all(suite, raw_scores, threshold)
    metrics = evaluate_predictions(suite, predictions)
    checks = requirement_results(metrics, requirements)
    critical_names = (
        "entity_mismatch_rejection",
        "predicate_mismatch_rejection",
        "target_slot_identity_mismatch_rejection",
        "temporal_scope_mismatch_rejection",
        "location_scope_mismatch_rejection",
        "organizational_scope_mismatch_rejection",
        "conditional_scope_mismatch_rejection",
        "modality_quantification_mismatch_rejection",
        "insufficient_recall",
        "refutes_recall",
        "conflicting_evidence_recall",
    )
    minimum_critical = min(float(metrics[name]) for name in critical_names)
    return {
        "alignment_confidence_min": threshold,
        "metrics": metrics,
        "requirements": checks,
        "requirements_passed": sum(checks.values()),
        "requirements_total": len(checks),
        "feasible": all(checks.values()),
        "minimum_safety_critical_recall": minimum_critical,
    }


def _selection_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    return (
        int(bool(candidate["feasible"])),
        int(candidate["requirements_passed"]),
        float(metrics["final_relation_macro_f1"]),
        float(candidate["minimum_safety_critical_recall"]),
        float(metrics["claim_category_macro_accuracy"]),
        float(metrics["minimal_sufficient_group_exact_match"]),
        float(metrics["scope_compatibility_macro_f1"]),
        float(metrics["slot_relation_macro_f1"]),
        float(metrics["target_proposition_f1"]),
        float(candidate["alignment_confidence_min"]),
    )


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("A4.5b-M6 candidate set is empty")
    return max(candidates, key=_selection_key)


def probability_triplet(logits: list[float]) -> dict[str, float]:
    if len(logits) != 3:
        raise RuntimeError("A4.5b-M6 residual model must emit exactly three logits")
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return {
        "contradiction": exps[0] / total,
        "entailment": exps[1] / total,
        "neutral": exps[2] / total,
    }

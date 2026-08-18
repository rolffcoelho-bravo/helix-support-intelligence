"""Framework-neutral routing endpoint contract for the frozen Phase 2 router."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

from helix_support_intelligence.domain.decisions import TerminalDecision

ROUTE_ENDPOINT_PATH = "/v1/tickets/route"
ROUTE_ENDPOINT_METHOD = "POST"


class RoutingContractError(ValueError):
    """Raised when a request or frozen routing configuration violates its contract."""


class Urgency(StrEnum):
    """Urgency values permitted by the public routing response contract."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class IntentScorer(Protocol):
    """Replaceable ML boundary supplying uncalibrated intent probabilities."""

    @property
    def model_id(self) -> str:
        """Return the model-ladder identifier implemented by this scorer."""

    def predict_proba(self, text: str) -> Mapping[str, float]:
        """Return one probability for every frozen routing intent."""


@dataclass(frozen=True, slots=True)
class RouteRequest:
    """Validated request accepted by the framework-neutral route endpoint."""

    request_id: UUID
    text: str
    urgency: Urgency = Urgency.NORMAL

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RouteRequest:
        """Parse the public route-request payload and reject undeclared fields."""
        allowed = {"request_id", "text", "urgency"}
        unknown = set(payload) - allowed
        if unknown:
            raise RoutingContractError(f"unknown route-request fields: {sorted(unknown)}")

        request_id_value = payload.get("request_id")
        if not isinstance(request_id_value, str):
            raise RoutingContractError("request_id must be a UUID string")
        try:
            request_id = UUID(request_id_value)
        except ValueError as exc:
            raise RoutingContractError("request_id must be a valid UUID") from exc

        text_value = payload.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise RoutingContractError("text must be a non-empty string")

        urgency_value = payload.get("urgency", Urgency.NORMAL.value)
        if not isinstance(urgency_value, str):
            raise RoutingContractError("urgency must be a string")
        try:
            urgency = Urgency(urgency_value)
        except ValueError as exc:
            raise RoutingContractError("urgency is outside the public contract") from exc

        return cls(request_id=request_id, text=text_value.strip(), urgency=urgency)


@dataclass(frozen=True, slots=True)
class RouteAlternative:
    """One calibrated alternative intent exposed by the routing contract."""

    intent: str
    queue: str
    probability: float

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible representation declared by routing.schema.json."""
        return {
            "intent": self.intent,
            "queue": self.queue,
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """JSON-compatible routing decision matching the public routing schema."""

    request_id: UUID
    intent: str | None
    queue: str | None
    urgency: Urgency
    out_of_scope_score: float
    confidence: float
    top_alternatives: tuple[RouteAlternative, ...]
    decision: TerminalDecision
    reason_code: str | None
    model_version: str

    def as_dict(self) -> dict[str, object]:
        """Return the public response payload with no undeclared fields."""
        return {
            "request_id": str(self.request_id),
            "intent": self.intent,
            "queue": self.queue,
            "urgency": self.urgency.value,
            "out_of_scope_score": self.out_of_scope_score,
            "confidence": self.confidence,
            "top_alternatives": [item.as_dict() for item in self.top_alternatives],
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "model_version": self.model_version,
        }


@dataclass(frozen=True, slots=True)
class RoutingPolicyConfig:
    """Frozen application-facing values required to apply routing-selected-v1."""

    model_id: str
    model_version: str
    temperature: float
    threshold: float
    queue_by_intent: Mapping[str, str]


class RouteEndpoint:
    """Apply the frozen selective-routing policy to one validated ticket."""

    path = ROUTE_ENDPOINT_PATH
    method = ROUTE_ENDPOINT_METHOD

    def __init__(self, scorer: IntentScorer, config: RoutingPolicyConfig) -> None:
        self._scorer = scorer
        self._config = config
        if scorer.model_id != config.model_id:
            raise RoutingContractError(
                f"scorer model_id {scorer.model_id!r} does not match {config.model_id!r}"
            )

    @classmethod
    def from_config_files(
        cls,
        scorer: IntentScorer,
        selected_config_path: Path,
        operations_config_path: Path,
    ) -> RouteEndpoint:
        """Build the endpoint from the frozen public Phase 2 configuration files."""
        selected = _read_json_object(selected_config_path)
        operations = _read_json_object(operations_config_path)

        model = _require_object(selected, "model")
        calibration = _require_object(selected, "calibration")
        policy = _require_object(selected, "operating_policy")
        intent_rows = _require_object(operations, "intents")

        model_id = _require_string(model, "id")
        model_version = _require_string(selected, "version")
        calibration_method = _require_string(calibration, "method")
        if calibration_method != "temperature_scaling":
            raise RoutingContractError("selected router must use temperature_scaling")

        temperature = _require_positive_number(calibration, "temperature")
        threshold = _require_probability(policy, "threshold")

        queues: dict[str, str] = {}
        for intent, row in intent_rows.items():
            if not isinstance(intent, str):
                raise RoutingContractError("intent names must be strings")
            if not isinstance(row, dict):
                raise RoutingContractError(f"operation row for {intent!r} must be an object")
            typed_row = cast(dict[str, object], row)
            queues[intent] = _require_string(typed_row, "queue")
        if not queues:
            raise RoutingContractError("routing operations must declare at least one intent")

        config = RoutingPolicyConfig(
            model_id=model_id,
            model_version=model_version,
            temperature=temperature,
            threshold=threshold,
            queue_by_intent=queues,
        )
        return cls(scorer, config)

    def handle(self, request: RouteRequest) -> RoutingDecision:
        """Return AUTO_ROUTE or an explicit safe escalation for one request."""
        try:
            raw_probabilities = self._scorer.predict_proba(request.text)
        except Exception:
            return self._system_failure(request, "routing_scorer_failure")

        try:
            ranked = _calibrated_ranking(
                raw_probabilities,
                self._config.queue_by_intent,
                self._config.temperature,
            )
        except RoutingContractError:
            return self._system_failure(request, "routing_invalid_model_output")

        best = ranked[0]
        confidence = best.probability
        out_of_scope_score = 1.0 - confidence

        if confidence >= self._config.threshold:
            return RoutingDecision(
                request_id=request.request_id,
                intent=best.intent,
                queue=best.queue,
                urgency=request.urgency,
                out_of_scope_score=out_of_scope_score,
                confidence=confidence,
                top_alternatives=tuple(ranked[1:4]),
                decision=TerminalDecision.AUTO_ROUTE,
                reason_code=None,
                model_version=self._config.model_version,
            )

        return RoutingDecision(
            request_id=request.request_id,
            intent=None,
            queue=None,
            urgency=request.urgency,
            out_of_scope_score=out_of_scope_score,
            confidence=confidence,
            top_alternatives=tuple(ranked[:3]),
            decision=TerminalDecision.ESCALATE_LOW_CONFIDENCE,
            reason_code="routing_confidence_below_frozen_threshold",
            model_version=self._config.model_version,
        )

    def _system_failure(self, request: RouteRequest, reason_code: str) -> RoutingDecision:
        return RoutingDecision(
            request_id=request.request_id,
            intent=None,
            queue=None,
            urgency=request.urgency,
            out_of_scope_score=1.0,
            confidence=0.0,
            top_alternatives=(),
            decision=TerminalDecision.ESCALATE_SYSTEM_FAILURE,
            reason_code=reason_code,
            model_version=self._config.model_version,
        )


def _calibrated_ranking(
    raw_probabilities: Mapping[str, float],
    queue_by_intent: Mapping[str, str],
    temperature: float,
) -> list[RouteAlternative]:
    expected_intents = set(queue_by_intent)
    supplied_intents = set(raw_probabilities)
    if supplied_intents != expected_intents:
        missing = sorted(expected_intents - supplied_intents)
        extra = sorted(supplied_intents - expected_intents)
        raise RoutingContractError(
            f"model output intent set drifted; missing={missing}, extra={extra}"
        )

    validated: dict[str, float] = {}
    for intent, value in raw_probabilities.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RoutingContractError(f"probability for {intent!r} must be numeric")
        probability = float(value)
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise RoutingContractError(f"probability for {intent!r} is outside [0, 1]")
        validated[intent] = probability

    total = sum(validated.values())
    if not math.isfinite(total) or total <= 0.0:
        raise RoutingContractError("model probabilities must have positive finite mass")

    normalized = {intent: value / total for intent, value in validated.items()}
    exponent = 1.0 / temperature
    powered = {intent: value**exponent for intent, value in normalized.items()}
    powered_total = sum(powered.values())
    if not math.isfinite(powered_total) or powered_total <= 0.0:
        raise RoutingContractError("temperature scaling produced invalid probability mass")

    calibrated = {intent: value / powered_total for intent, value in powered.items()}
    ordered = sorted(calibrated.items(), key=lambda item: (-item[1], item[0]))
    return [
        RouteAlternative(intent=intent, queue=queue_by_intent[intent], probability=probability)
        for intent, probability in ordered
    ]


def _read_json_object(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RoutingContractError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def _require_object(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RoutingContractError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RoutingContractError(f"{key} must be a non-empty string")
    return value


def _require_positive_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RoutingContractError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RoutingContractError(f"{key} must be positive and finite")
    return number


def _require_probability(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RoutingContractError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise RoutingContractError(f"{key} must be in [0, 1]")
    return number

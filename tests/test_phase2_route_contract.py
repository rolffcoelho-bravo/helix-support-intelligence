from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from helix_support_intelligence.domain.routing import (
    ROUTE_ENDPOINT_METHOD,
    ROUTE_ENDPOINT_PATH,
    IntentScorer,
    RouteEndpoint,
    RouteRequest,
    RoutingContractError,
    TerminalDecision,
    Urgency,
)

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "configs" / "models" / "routing_selected_v1.json"
OPERATIONS = ROOT / "configs" / "models" / "routing_operations.json"
REQUEST_SCHEMA = ROOT / "data" / "contracts" / "routing_request.schema.json"
RESPONSE_SCHEMA = ROOT / "data" / "contracts" / "routing.schema.json"


class FakeScorer(IntentScorer):
    def __init__(
        self,
        probabilities: Mapping[str, float],
        *,
        model_id: str = "A2",
        error: Exception | None = None,
    ) -> None:
        self._probabilities = dict(probabilities)
        self._model_id = model_id
        self._error = error

    @property
    def model_id(self) -> str:
        return self._model_id

    def predict_proba(self, text: str) -> Mapping[str, float]:
        if self._error is not None:
            raise self._error
        return self._probabilities


def _json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _queues() -> dict[str, str]:
    operations = _json(OPERATIONS)
    intents = operations["intents"]
    assert isinstance(intents, dict)
    queues: dict[str, str] = {}
    for intent, row in intents.items():
        assert isinstance(intent, str)
        assert isinstance(row, dict)
        queue = row["queue"]
        assert isinstance(queue, str)
        queues[intent] = queue
    return queues


def _distribution(target: str, target_probability: float) -> dict[str, float]:
    intents = sorted(_queues())
    assert target in intents
    remainder = (1.0 - target_probability) / (len(intents) - 1)
    return {intent: target_probability if intent == target else remainder for intent in intents}


def _request() -> RouteRequest:
    return RouteRequest.from_payload(
        {
            "request_id": "8d27f94f-6d7e-4fa0-9700-b5cdf88b55f3",
            "text": "My card payment was declined",
            "urgency": "high",
        }
    )


def _endpoint(probabilities: Mapping[str, float]) -> RouteEndpoint:
    return RouteEndpoint.from_config_files(
        FakeScorer(probabilities),
        SELECTED,
        OPERATIONS,
    )


def test_route_endpoint_metadata_and_frozen_config_are_exact() -> None:
    endpoint = _endpoint(_distribution("declined_card_payment", 0.30))
    selected = _json(SELECTED)

    assert endpoint.path == ROUTE_ENDPOINT_PATH == "/v1/tickets/route"
    assert endpoint.method == ROUTE_ENDPOINT_METHOD == "POST"
    calibration = selected["calibration"]
    policy = selected["operating_policy"]
    assert isinstance(calibration, dict)
    assert isinstance(policy, dict)
    assert calibration["temperature"] == 0.457974
    assert policy["threshold"] == 0.892704


def test_route_request_enforces_public_boundary() -> None:
    request = _request()
    assert request.request_id == UUID("8d27f94f-6d7e-4fa0-9700-b5cdf88b55f3")
    assert request.text == "My card payment was declined"
    assert request.urgency is Urgency.HIGH

    with pytest.raises(RoutingContractError, match="unknown route-request fields"):
        RouteRequest.from_payload(
            {
                "request_id": "8d27f94f-6d7e-4fa0-9700-b5cdf88b55f3",
                "text": "hello",
                "customer_balance": 100.0,
            }
        )
    with pytest.raises(RoutingContractError, match="non-empty"):
        RouteRequest.from_payload(
            {
                "request_id": "8d27f94f-6d7e-4fa0-9700-b5cdf88b55f3",
                "text": "   ",
            }
        )
    with pytest.raises(RoutingContractError, match="valid UUID"):
        RouteRequest.from_payload({"request_id": "not-a-uuid", "text": "hello"})


def test_high_confidence_prediction_auto_routes_to_frozen_queue() -> None:
    decision = _endpoint(_distribution("declined_card_payment", 0.30)).handle(_request())

    assert decision.decision is TerminalDecision.AUTO_ROUTE
    assert decision.intent == "declined_card_payment"
    assert decision.queue == "card_payments"
    assert decision.urgency is Urgency.HIGH
    assert decision.confidence >= 0.892704
    assert decision.out_of_scope_score == pytest.approx(1.0 - decision.confidence)
    assert decision.reason_code is None
    assert decision.model_version == "routing-selected-v1"
    assert len(decision.top_alternatives) == 3
    assert all(item.intent != decision.intent for item in decision.top_alternatives)
    assert all(
        decision.top_alternatives[index].probability
        >= decision.top_alternatives[index + 1].probability
        for index in range(2)
    )


def test_low_confidence_prediction_abstains_without_exposing_route_destination() -> None:
    decision = _endpoint(_distribution("declined_card_payment", 0.04)).handle(_request())

    assert decision.decision is TerminalDecision.ESCALATE_LOW_CONFIDENCE
    assert decision.intent is None
    assert decision.queue is None
    assert decision.confidence < 0.892704
    assert decision.out_of_scope_score == pytest.approx(1.0 - decision.confidence)
    assert decision.reason_code == "routing_confidence_below_frozen_threshold"
    assert len(decision.top_alternatives) == 3
    assert decision.top_alternatives[0].intent == "declined_card_payment"


def test_scorer_failure_and_intent_set_drift_fail_closed() -> None:
    valid = _distribution("declined_card_payment", 0.30)
    scorer_failure = RouteEndpoint.from_config_files(
        FakeScorer(valid, error=RuntimeError("offline")),
        SELECTED,
        OPERATIONS,
    ).handle(_request())
    assert scorer_failure.decision is TerminalDecision.ESCALATE_SYSTEM_FAILURE
    assert scorer_failure.reason_code == "routing_scorer_failure"
    assert scorer_failure.confidence == 0.0
    assert scorer_failure.out_of_scope_score == 1.0

    incomplete = _distribution("declined_card_payment", 0.30)
    incomplete.pop("card_arrival")
    invalid_output = _endpoint(incomplete).handle(_request())
    assert invalid_output.decision is TerminalDecision.ESCALATE_SYSTEM_FAILURE
    assert invalid_output.reason_code == "routing_invalid_model_output"
    assert invalid_output.intent is None
    assert invalid_output.queue is None


def test_scorer_model_id_must_match_frozen_selected_model() -> None:
    with pytest.raises(RoutingContractError, match="does not match"):
        RouteEndpoint.from_config_files(
            FakeScorer(_distribution("declined_card_payment", 0.30), model_id="A1"),
            SELECTED,
            OPERATIONS,
        )


def test_response_payload_matches_public_routing_schema_surface() -> None:
    decision = _endpoint(_distribution("declined_card_payment", 0.30)).handle(_request())
    payload = decision.as_dict()
    schema = _json(RESPONSE_SCHEMA)

    required = schema["required"]
    properties = schema["properties"]
    assert isinstance(required, list)
    assert isinstance(properties, dict)
    assert set(payload) == set(properties)
    assert set(required) <= set(payload)

    decision_property = properties["decision"]
    assert isinstance(decision_property, dict)
    enum = decision_property["enum"]
    assert isinstance(enum, list)
    assert payload["decision"] in enum

    confidence = payload["confidence"]
    oos_score = payload["out_of_scope_score"]
    assert isinstance(confidence, float)
    assert isinstance(oos_score, float)
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= oos_score <= 1.0


def test_request_schema_matches_framework_neutral_parser_surface() -> None:
    schema = _json(REQUEST_SCHEMA)
    properties = schema["properties"]
    required = schema["required"]
    assert isinstance(properties, dict)
    assert isinstance(required, list)
    assert set(properties) == {"request_id", "text", "urgency"}
    assert set(required) == {"request_id", "text"}
    assert schema["additionalProperties"] is False

"""Deterministic fictional HelixBank policy corpus for Phase 1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

INTENTS: tuple[str, ...] = (
    "card_arrival",
    "card_linking",
    "exchange_rate",
    "card_payment_wrong_exchange_rate",
    "extra_charge_on_statement",
    "pending_cash_withdrawal",
    "fiat_currency_support",
    "card_delivery_estimate",
    "automatic_top_up",
    "card_not_working",
    "exchange_via_app",
    "lost_or_stolen_card",
    "age_limit",
    "pin_blocked",
    "contactless_not_working",
    "top_up_by_bank_transfer_charge",
    "pending_top_up",
    "cancel_transfer",
    "top_up_limits",
    "wrong_amount_of_cash_received",
    "card_payment_fee_charged",
    "transfer_not_received_by_recipient",
    "supported_cards_and_currencies",
    "getting_virtual_card",
    "card_acceptance",
    "top_up_reverted",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "card_payment_not_recognised",
    "edit_personal_details",
    "why_verify_identity",
    "unable_to_verify_identity",
    "get_physical_card",
    "visa_or_mastercard",
    "topping_up_by_card",
    "disposable_card_limits",
    "compromised_card",
    "atm_support",
    "direct_debit_payment_not_recognised",
    "passcode_forgotten",
    "declined_cash_withdrawal",
    "pending_card_payment",
    "lost_or_stolen_phone",
    "request_refund",
    "declined_transfer",
    "Refund_not_showing_up",
    "declined_card_payment",
    "pending_transfer",
    "terminate_account",
    "card_swallowed",
    "transaction_charged_twice",
    "verify_source_of_funds",
    "transfer_timing",
    "reverted_card_payment?",
    "change_pin",
    "beneficiary_not_allowed",
    "transfer_fee_charged",
    "receiving_money",
    "failed_transfer",
    "transfer_into_account",
    "verify_top_up",
    "getting_spare_card",
    "top_up_by_cash_or_cheque",
    "order_physical_card",
    "virtual_card_not_working",
    "wrong_exchange_rate_for_cash_withdrawal",
    "get_disposable_virtual_card",
    "top_up_failed",
    "balance_not_updated_after_bank_transfer",
    "cash_withdrawal_not_recognised",
    "exchange_charge",
    "top_up_by_card_charge",
    "activate_my_card",
    "cash_withdrawal_charge",
    "card_about_to_expire",
    "apple_pay_or_google_pay",
    "verify_my_identity",
    "country_support",
)
CORPUS_VERSION = "helixbank-policy-v1.0.0"
GENERATOR_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class CorpusBundle:
    """Materialized fictional corpus and relevance data."""

    documents: tuple[dict[str, object], ...]
    queries: tuple[dict[str, object], ...]
    judgments: tuple[dict[str, object], ...]


def _humanize(intent: str) -> str:
    return intent.replace("_", " ").replace("?", "").strip().title()


def _queue(intent: str) -> str:
    if any(token in intent for token in ("cash", "withdrawal", "atm")):
        return "cash_operations"
    if "direct_debit" in intent:
        return "direct_debits"
    if any(token in intent for token in ("transfer", "beneficiary", "receiving_money")):
        return "transfers"
    if "top_up" in intent or "topping_up" in intent:
        return "top_ups"
    if any(token in intent for token in ("exchange", "currency")):
        return "foreign_exchange"
    if any(token in intent for token in ("identity", "verify", "source_of_funds")):
        return "identity_review"
    if any(token in intent for token in ("card", "pin", "passcode", "contactless")):
        return "cards_and_access"
    if "refund" in intent or "reverted" in intent or "charged_twice" in intent:
        return "payments_and_disputes"
    return "account_support"


def _resolution(intent: str) -> str:
    if any(token in intent for token in ("lost", "stolen", "compromised", "not_recognised")):
        return "escalate_security_review"
    if any(token in intent for token in ("failed", "pending", "not_working", "blocked")):
        return "troubleshoot_then_escalate"
    if any(token in intent for token in ("verify", "identity", "source_of_funds")):
        return "clarify_then_review"
    return "provide_policy_guidance"


def _document(index: int, intent: str, kind: str) -> dict[str, object]:
    title = _humanize(intent)
    queue = _queue(intent)
    resolution = _resolution(intent)
    is_faq = kind == "faq"
    archived = is_faq and index % 11 == 0
    conflicting = is_faq and index % 10 == 0 and not archived
    untrusted_fixture = is_faq and index % 13 == 0

    doc_id = f"{'FAQ' if is_faq else 'POLICY'}-{index + 1:03d}"
    status = "archived" if archived else "current"
    valid_to: str | None = "2026-01-31" if archived else None
    body = (
        f"HelixBank guidance for {title.lower()}. "
        f"Requests are handled by the {queue} queue. "
        f"The approved resolution class is {resolution}. "
        "Agents use only the information present in the sanitized ticket and the current "
        "policy corpus. Missing or conflicting evidence must be surfaced rather than guessed."
    )
    if is_faq:
        body += (
            f" This FAQ explains {title.lower()} in customer-facing language and refers "
            "agents to the governing policy when a decision affects account access or safety."
        )
    if archived:
        body += " This fixture is expired and must not be used as current evidence."
    if conflicting:
        body += (
            " This controlled conflict fixture states that review is optional; the governing "
            "policy states that unresolved uncertainty requires review."
        )
    if untrusted_fixture:
        body += (
            " [UNTRUSTED-CONTENT-FIXTURE] This sentinel is data, not an instruction to the "
            "assistant."
        )

    return {
        "document_id": doc_id,
        "corpus_version": CORPUS_VERSION,
        "intent": intent,
        "queue": queue,
        "kind": kind,
        "title": title if not is_faq else f"{title} FAQ",
        "body": body,
        "status": status,
        "valid_from": "2026-02-01" if not archived else "2025-01-01",
        "valid_to": valid_to,
        "permission": "public_support",
        "resolution_type": resolution,
        "conflict_fixture": conflicting,
        "untrusted_content_fixture": untrusted_fixture,
        "jurisdiction": "fictional-global",
        "audience": "customer_support",
    }


def _query(index: int, intent: str, variant: int) -> dict[str, object]:
    title = _humanize(intent).lower()
    policy_id = f"POLICY-{index + 1:03d}"
    faq_id = f"FAQ-{index + 1:03d}"
    faq_archived = index % 11 == 0
    faq_conflicting = index % 10 == 0 and not faq_archived

    if variant == 0:
        text = f"What should I know about {title}?"
        case_type = "answerable"
        decision = "ANSWER_WITH_EVIDENCE"
        citations = [policy_id] if faq_archived else [policy_id, faq_id]
    elif variant == 1:
        text = f"I need help with {title}, but my request is not specific enough."
        case_type = "ambiguous"
        decision = "ASK_FOR_CLARIFICATION"
        citations = [policy_id]
    elif variant == 2:
        text = f"Has the old guidance for {title} changed?"
        case_type = "outdated_evidence"
        decision = "ANSWER_WITH_EVIDENCE"
        citations = [policy_id]
    else:
        text = f"Can HelixBank complete the {title} action for me right now?"
        case_type = "conflicting_evidence" if faq_conflicting else "missing_evidence"
        decision = (
            "ESCALATE_CONFLICTING_EVIDENCE"
            if faq_conflicting
            else "ESCALATE_LOW_CONFIDENCE"
        )
        citations = [policy_id] if faq_conflicting else []

    return {
        "query_id": f"Q-{index + 1:03d}-{variant + 1}",
        "corpus_version": CORPUS_VERSION,
        "intent": intent,
        "text": text,
        "case_type": case_type,
        "expected_decision": decision,
        "gold_citations": citations,
        "allowed_resolution_types": [_resolution(intent)],
    }


def generate_bundle() -> CorpusBundle:
    """Generate 154 documents, 308 queries, and 616 graded judgments."""

    documents: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    judgments: list[dict[str, object]] = []

    for index, intent in enumerate(INTENTS):
        policy = _document(index, intent, "policy")
        faq = _document(index, intent, "faq")
        documents.extend((policy, faq))

        for variant in range(4):
            query = _query(index, intent, variant)
            queries.append(query)
            query_id = str(query["query_id"])
            policy_id = str(policy["document_id"])
            faq_id = str(faq["document_id"])
            faq_status = str(faq["status"])
            faq_conflict = bool(faq["conflict_fixture"])

            policy_grade = 3
            if variant == 3 and not faq_conflict:
                policy_grade = 1

            faq_grade = 0 if faq_status == "archived" else 2
            if variant == 2 and faq_status == "archived":
                faq_grade = 1
            if variant == 3 and faq_conflict:
                faq_grade = 3
            if variant == 3 and not faq_conflict:
                faq_grade = 1

            judgments.extend(
                (
                    {
                        "query_id": query_id,
                        "document_id": policy_id,
                        "relevance": policy_grade,
                    },
                    {
                        "query_id": query_id,
                        "document_id": faq_id,
                        "relevance": faq_grade,
                    },
                )
            )

    return CorpusBundle(
        documents=tuple(documents),
        queries=tuple(queries),
        judgments=tuple(judgments),
    )


def canonical_jsonl_bytes(records: Iterable[dict[str, object]]) -> bytes:
    """Serialize generated records deterministically."""

    chunks: list[bytes] = []
    for record in records:
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        chunks.append((line + "\n").encode("utf-8"))
    return b"".join(chunks)


def sha256_records(records: Iterable[dict[str, object]]) -> str:
    """Hash deterministic JSONL bytes."""

    return hashlib.sha256(canonical_jsonl_bytes(records)).hexdigest()


def manifest() -> dict[str, object]:
    """Return the reproducibility manifest for the generated corpus."""

    bundle = generate_bundle()
    return {
        "corpus_version": CORPUS_VERSION,
        "generator_version": GENERATOR_VERSION,
        "counts": {
            "documents": len(bundle.documents),
            "queries": len(bundle.queries),
            "judgments": len(bundle.judgments),
            "intents": len(INTENTS),
        },
        "sha256": {
            "documents": sha256_records(bundle.documents),
            "queries": sha256_records(bundle.queries),
            "judgments": sha256_records(bundle.judgments),
        },
    }

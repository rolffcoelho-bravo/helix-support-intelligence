"""Terminal decisions permitted by the public product contract."""

from enum import StrEnum


class TerminalDecision(StrEnum):
    """A request must finish in exactly one of these mutually exclusive states."""

    ANSWER_WITH_EVIDENCE = "ANSWER_WITH_EVIDENCE"
    AUTO_ROUTE = "AUTO_ROUTE"
    RECOMMEND_TO_AGENT = "RECOMMEND_TO_AGENT"
    ASK_FOR_CLARIFICATION = "ASK_FOR_CLARIFICATION"
    ESCALATE_LOW_CONFIDENCE = "ESCALATE_LOW_CONFIDENCE"
    ESCALATE_OUT_OF_SCOPE = "ESCALATE_OUT_OF_SCOPE"
    ESCALATE_CONFLICTING_EVIDENCE = "ESCALATE_CONFLICTING_EVIDENCE"
    ESCALATE_SAFETY_RISK = "ESCALATE_SAFETY_RISK"
    ESCALATE_SYSTEM_FAILURE = "ESCALATE_SYSTEM_FAILURE"

    @property
    def requires_human_review(self) -> bool:
        """Return whether the state transfers control to a human operator."""
        return self.value.startswith("ESCALATE_")

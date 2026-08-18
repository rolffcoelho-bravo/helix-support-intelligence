"""Foundation contract tests."""

from helix_support_intelligence import __version__
from helix_support_intelligence.domain import TerminalDecision


def test_installed_package_exposes_version() -> None:
    assert __version__ != "0+unknown"


def test_terminal_decisions_are_unique_and_complete() -> None:
    values = {decision.value for decision in TerminalDecision}

    assert len(values) == 9
    assert "ANSWER_WITH_EVIDENCE" in values
    assert "ESCALATE_SYSTEM_FAILURE" in values


def test_only_escalation_states_require_human_review() -> None:
    for decision in TerminalDecision:
        assert decision.requires_human_review is decision.value.startswith("ESCALATE_")

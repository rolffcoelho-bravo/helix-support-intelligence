"""Minimal command surface for the repository-foundation release."""

import json

from helix_support_intelligence import __version__
from helix_support_intelligence.domain.decisions import TerminalDecision


def main() -> None:
    """Print machine-readable foundation metadata."""
    payload = {
        "name": "Helix Support Intelligence",
        "phase": 0,
        "status": "foundation-complete",
        "terminal_decisions": [decision.value for decision in TerminalDecision],
        "version": __version__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

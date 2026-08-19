"""Machine-readable repository status for the current public checkpoint."""

import json

from helix_support_intelligence import __version__
from helix_support_intelligence.api.search import RETRIEVAL_VERSION
from helix_support_intelligence.domain.decisions import TerminalDecision


def main() -> None:
    """Print machine-readable public checkpoint metadata."""
    payload = {
        "name": "Helix Support Intelligence",
        "phase": 3,
        "status": "retrieval-integrated",
        "retrieval_version": RETRIEVAL_VERSION,
        "search_endpoint": "POST /v1/search",
        "terminal_decisions": [decision.value for decision in TerminalDecision],
        "version": __version__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

"""Fail closed when public files resemble private material or contain common secrets."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

PROHIBITED_PATH_PATTERNS = (
    "*blueprint*",
    "citations.md",
    "findings.md",
    "docs/phase-reports/*",
    "notes*.txt",
    "*.private.*",
    "*.internal.*",
    "*_private.*",
    "*_internal.*",
    ".private/*",
    ".internal/*",
    "internal/*",
)

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{30,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Windows workstation path": re.compile(r"\b[A-Za-z]:\\+(?:Users|Claude AI)\\+"),
}

PRIVATE_PROCESS_PATTERNS = {
    "private blueprint reference": re.compile(
        r"\b(?:next |internal )?blueprint(?: action| objective)?\b",
        re.I,
    ),
    "approval-gate language": re.compile(
        r"\b(?:approval gate|explicit authorization|authorized next phase)\b",
        re.I,
    ),
    "internal phase-lock language": re.compile(
        r"\b(?:phase lock|next locked action)\b",
        re.I,
    ),
    "internal audit phrasing": re.compile(r"\bhostile audit\b", re.I),
    "private research repository": re.compile(r"\bhelix-support-intelligence-core\b", re.I),
    "private workspace": re.compile(r"\bproject_helix_support_intelligence_private\b", re.I),
    "private findings register": re.compile(r"\bFINDINGS\.md\b"),
}


def tracked_files() -> list[Path]:
    """Return Git-tracked files, or all repository files outside transient directories."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0 and result.stdout:
        return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]

    excluded = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
    }
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not excluded & set(path.parts)
    ]


def audit() -> list[str]:
    """Return human-readable publication-boundary violations."""
    violations: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        lowered = relative.lower()
        if any(fnmatch.fnmatch(lowered, pattern) for pattern in PROHIBITED_PATH_PATTERNS):
            violations.append(f"prohibited public path: {relative}")

        if path.resolve() == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"possible {label} in {relative}")
        for label, pattern in PRIVATE_PROCESS_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"possible {label} in {relative}")
    return violations


def main() -> None:
    """Run the audit and exit non-zero on any violation."""
    violations = audit()
    if violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise SystemExit(f"Publication audit failed:\n{details}")
    print("Publication audit passed.")


if __name__ == "__main__":
    main()

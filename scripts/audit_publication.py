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
    "notes*.txt",
    "*.private.*",
    "*.internal.*",
    "*_private.*",
    "*_internal.*",
    ".private/*",
    ".internal/*",
)

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{30,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Windows workstation path": re.compile(r"\b[A-Za-z]:\\+(?:Users|Claude AI)\\+"),
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
    local_reference_files = {"HELIX_INTERNAL_BLUEPRINT.md"}
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in local_reference_files
        and not excluded & set(path.parts)
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

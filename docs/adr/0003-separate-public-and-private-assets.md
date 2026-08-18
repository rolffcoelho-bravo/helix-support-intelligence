# ADR 0003: Separate Public and Private Assets

- Status: Accepted
- Date: 2026-08-18

## Context

Helix is a public reference implementation with private research, prompts, security detail, commercial integrations, data, and future proprietary modules maintained separately.

## Decision

The public repository uses an allow-by-review publication model. Only material with clear public value and no confidential or security-sensitive disclosure is committed. CI checks common secret patterns and prohibited private filenames. Uncertain material remains outside the public repository.

## Consequences

The public project remains reproducible and credible without treating private intellectual property as unfinished public work. Automated checks reduce risk but do not replace human review.

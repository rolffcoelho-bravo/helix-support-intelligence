# ADR 0002: Use a `src` Layout and Stable Domain Boundaries

- Status: Accepted
- Date: 2026-08-18

## Context

The project will add models, retrieval systems, APIs, and infrastructure over several phases. Import behaviour must remain independent of the working directory, and provider code must not control product decisions.

## Decision

Use a typed Python 3.12 package under `src/helix_support_intelligence`. Stable domain contracts live in provider-independent modules. Tests import the installed package, and notebooks never become the authoritative implementation.

## Consequences

Clean-checkout imports are testable, accidental local imports are reduced, and replaceable components remain behind explicit boundaries.

# Phase 4 A4.5d: TPAG fresh-validation registration and pre-execution contract audit

## Status

A4.5d is a zero-result registration and pre-execution audit checkpoint. It does not open, materialize, inspect, or score the 20-unit A4.5a fresh-validation partition and it does not inspect or score the 68-query confirmatory partition.

The checkpoint is closed as:

`REGISTERED_ZERO_RESULT_VALIDATION_BLOCKED_INPUT_CONTRACT_MISMATCH`

This is not a negative validation result. No fresh-validation result exists.

## Frozen scientific state inherited from M6

A4.5b-M6 remains closed as `PASSED_TPAG_CALIBRATION_READINESS_PARAMETERS_FROZEN` at main SHA `40d6bdb417e798a7c0ead7709bdcec5d8241a989`.

The authoritative TPAG implementation remains unchanged:

- implementation: `tpag-deterministic-typed-parser-plus-qc-nli-residual-v1`;
- residual model: `cross-encoder/nli-deberta-v3-base`;
- revision: `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- model SHA-256: `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`;
- tokenizer SHA-256: `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd`;
- selected `alignment_confidence_min`: `0.60`.

A4.5d authorizes no model substitution, parser modification, prompt search, threshold search, parameter refit, or post-result rescue.

## Frozen A4.5a validity contract

The original A4.5a validity surface remains the only sealed fresh-validation surface currently registered for the AERF lineage:

- 20 validation units;
- 180 component-pair rows;
- 180 claim rows;
- 60 ENTAILED, 40 CONTRADICTED, and 80 UNKNOWN component relations;
- validation pair SHA-256 `5f6f0294230de5da3af8baaee2403c9497bd42308c96f9d1041f4f88667d1da7`;
- validation claim SHA-256 `116040d37035e4a43a3bee17ea2d29fe87d85c6148adade770e8c224456e43d6`.

All 29 original A4.5a hard validity requirements are inherited without alteration: 19 component requirements and 10 claim/safety requirements.

## Pre-execution contract finding

The static audit found that the frozen M6 TPAG implementation cannot be applied directly to the registered A4.5a support-text grammar without changing the scientific implementation.

The mismatch is concrete and deterministic. A4.5a constructs subjects as `Orchid case NNN` and ordinary support prose such as handled-by, requires, and review-window statements. The frozen M6 TPAG parser instead recognizes the later TPAG-native `Orchid request NNN` / `OR-NNN` subject grammar and the M5/M6 typed operation/frame patterns.

Therefore, directly opening the A4.5a validation partition with the frozen M6 parser would not be a clean test of the calibrated TPAG method. Conversely, silently adding an A4.5a-specific parser or deterministic adapter now would change the frozen scientific implementation after calibration and before validation.

A4.5d resolves this by failing closed before any validation exposure.

## What was not done

A4.5d performs zero semantic inference and zero model calls. It performs zero threshold evaluations and zero parameter fitting. It does not construct or enumerate the 20 validation units. It does not read their individual text, gold labels, predictions, or metrics. It does not inspect the confirmatory records.

The audit uses only already-public registration metadata, the deterministic source contracts, and synthetic non-validation probe strings to verify the parser boundary.

## Governance consequence

Fresh validation remains sealed. A4.5c remains permanently ineligible and is not repurposed.

No validation workflow is registered or authorized in A4.5d. Before any fresh-validation execution, a separately approved checkpoint must resolve the input-contract problem without using the sealed validation records for design or tuning. Scientifically defensible options are limited to either:

1. a pre-registered, independently justified compatibility bridge whose design does not inspect the sealed validation records and whose effect is established on non-validation evidence; or
2. a new TPAG-native independent validity construction registered before implementation-specific execution.

Which route to take is a separate methodology decision. A4.5d does not pre-authorize either one.

## Closure boundary

A4.5d is complete when the zero-result registration, blocker record, fail-closed preflight, static tests, and ordinary repository quality gate pass. Its successful closure means only that the scientific boundary was identified and preserved before validation, not that TPAG has passed independent validation.

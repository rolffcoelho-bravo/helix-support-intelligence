# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: Active — protocol frozen, modelling not yet started
- Date opened: 2026-08-18
- Public version: 0.1.0

## Frozen inputs

- Phase 1 status: Passed.
- BANKING77 train: 7,904 rows; SHA-256 `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3`.
- BANKING77 validation: 1,976 rows; SHA-256 `5a6e2bef72257bb3aa33aba4ca4a93a13738e0a487be88e7846b986b33713455`.
- BANKING77 confirmatory test: 3,080 rows; SHA-256 `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468`.
- Model ladder: `routing-ladder-v1`, A0-A3 required; A4 disabled unless bounded remediation is authorized.
- Selection partition: validation only.

## Required deliverables

- [x] Public A0-A3 model-ladder contract.
- [x] Routing evaluation protocol.
- [ ] Reproducible A0-A3 implementations and evaluation artifacts.
- [ ] Calibration comparison.
- [ ] Frozen out-of-scope benchmark and evaluation.
- [ ] Declared routing cost matrix.
- [ ] Risk-coverage report.
- [ ] Router model card.
- [ ] `/v1/tickets/route` contract tests.
- [ ] One frozen routing configuration and operating threshold.
- [ ] Registered confirmatory result for H3 and H4.

## Test-set status

The official BANKING77 test split remains confirmatory and is not authorized for model selection, calibration selection, feature selection, or operating-threshold selection.

## Current decision

Phase 2 is open. The next implementation action is the deterministic routing evaluation harness and A0/A1 baselines against train/validation only. A2 and A3 dependency expansion must remain bounded and reproducible; no new classifier family may be added outside the frozen ladder.

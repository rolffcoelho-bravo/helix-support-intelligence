# Changelog

All notable public changes are documented here. The project follows semantic versioning from its first stable release.

## [Unreleased]

### Added

- Reproducible Python 3.12 package foundation.
- Stable terminal-decision contract.
- Locked development environment and quality command surface.
- CI, contribution, security, architecture, and publication-review controls.
- Citation metadata and the Phase 0 exit report.
- Pinned BANKING77 provenance, checksums, leakage quarantine, and deterministic split contract.
- Frozen HelixBank Policy Corpus v1 with documents, golden queries, and relevance judgments.
- Machine-readable routing, retrieval, generation, safety, event, and evaluation data contracts.
- Phase 1 offline data-quality validation and deterministic fixtures.
- Frozen Phase 2 routing model ladder and validation-only evaluation protocol.
- Deterministic A0/A1 routing benchmark executed in GitHub Actions with reproducible evidence artifacts.
- First public Phase 2 development checkpoint for the A1 TF-IDF + logistic-regression baseline, including calibration diagnostics, risk-coverage evidence, and confusion pairs.
- Frozen A2 sentence-embedding benchmark using a pinned `all-MiniLM-L6-v2` revision plus the same linear classifier specification used for A1 comparison.
- CPU-only PyTorch resolution and committed uv script lock for the A2 scientific environment, preventing silent transitive dependency or hardware-variant drift.
- Audited A2 validation v2 checkpoint with exact decision reproducibility, bounded numerical reproducibility, A1 comparison, calibration diagnostics, full-count confusion analysis, and descriptive CPU timing.
- Frozen A3 end-to-end MiniLM fine-tuning benchmark and audited negative-result checkpoint; the registered A3 recipe underfits and is rejected without post-result rescue tuning.
- Five-fold intent-stratified cross-fitted calibration benchmark for A1 and A2 with audited balanced fold assignment and CPU-only locked environment.
- Temperature scaling selected for both A1 and A2; calibrated A2 reduces validation ECE to approximately 0.0162 and Brier score to approximately 0.1398 without changing macro-F1 or top-3 recall.
- Regression tests protecting calibration fold balance, method selection, guardrail outcomes, bounded reproducibility, read-only workflow permissions, and confirmatory-test isolation.
- Mandatory end-of-checkpoint execution-audit gate covering code correctness, metric interpretation, split integrity, reproducibility, dependency/hardware consistency, CI behavior, and public claim wording.

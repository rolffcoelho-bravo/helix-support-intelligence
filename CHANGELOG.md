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
- Frozen 160-query, 20-category support-like OOS benchmark with cross-fitted evaluation; calibrated A2 reaches approximately 0.8956 OOS AUROC versus 0.8491 for calibrated A1, while high-recall OOS separation remains explicitly imperfect.
- Public synthetic routing-cost matrix with registered cost and OOS-prevalence sensitivities, plus a reproducible selective operating-point benchmark.
- Audited cost-policy checkpoint preserving the negative H3 development result for A2 calibration and the positive H4 selective-routing development result.
- Full-refit threshold-transfer audit separating unbiased cross-fitted development selection from the final calibration probability scale without cost re-optimization.
- Frozen `routing-selected-v1` development configuration: A2, temperature scaling, and a robust full-refit-scale selective-routing threshold.
- Regression tests protecting calibration fold balance, method selection, guardrail outcomes, bounded reproducibility, frozen routing selection, read-only workflow permissions, and confirmatory-test isolation.
- Mandatory end-of-checkpoint execution-audit gate covering code correctness, metric interpretation, split integrity, reproducibility, dependency/hardware consistency, CI behavior, and public claim wording.

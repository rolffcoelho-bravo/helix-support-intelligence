# Changelog

All notable public changes are documented here. The project follows semantic versioning from its first stable release.

## [Unreleased]

### Added

- Reproducible Python 3.12 package foundation.
- Stable terminal-decision contract.
- Locked development environment and quality command surface.
- CI, contribution, security, architecture, and publication-boundary controls.
- Machine-readable citation metadata.
- Pinned BANKING77 provenance, checksums, leakage quarantine, and deterministic split contract.
- Frozen HelixBank Policy Corpus v1 with documents, golden queries, and relevance judgments.
- Machine-readable routing, retrieval, generation, safety, event, and evaluation data contracts.
- Offline data-quality validation and deterministic fixtures.
- Reproducible routing model ladder covering A0 through A3.
- Deterministic A0/A1 routing benchmark with public evidence artifacts.
- A1 TF-IDF + logistic-regression baseline with calibration diagnostics, risk-coverage evidence, and confusion analysis.
- Frozen A2 sentence-embedding benchmark using a pinned `all-MiniLM-L6-v2` revision plus the same linear classifier specification used for A1 comparison.
- CPU-only PyTorch resolution and committed uv script lock for the A2 scientific environment.
- Audited A2 validation checkpoint with exact decision reproducibility, bounded numerical reproducibility, A1 comparison, calibration diagnostics, full-count confusion analysis, and descriptive CPU timing.
- Frozen A3 end-to-end MiniLM fine-tuning benchmark and preserved negative result; the registered three-epoch recipe underfits and was not rescued through post-result tuning.
- Five-fold intent-stratified cross-fitted calibration benchmark for A1 and A2.
- Temperature scaling selected for both A1 and A2; calibrated A2 reduces validation ECE to approximately 0.0162 and Brier score to approximately 0.1398 without changing macro-F1 or top-3 recall.
- Frozen 160-query, 20-category support-like OOS benchmark with cross-fitted evaluation; calibrated A2 reaches approximately 0.8956 OOS AUROC versus 0.8491 for calibrated A1, while high-recall OOS separation remains explicitly imperfect.
- Public synthetic routing-cost matrix with cost and OOS-prevalence sensitivities, plus a reproducible selective operating-point benchmark.
- Cost-policy evaluation preserving the negative development result for the calibration-cost hypothesis and the positive development result for selective routing.
- Full-refit threshold-transfer verification separating cross-fitted development selection from the final calibration probability scale without cost re-optimization.
- Frozen `routing-selected-v1` configuration: A2, temperature scaling, and a fixed selective-routing threshold.
- Public router model card and framework-neutral `POST /v1/tickets/route` domain contract tied to `routing-selected-v1`.
- Phase-specific route request schema under `data/contracts/phase2/`, preserving the previously frozen root contract suite.
- Fail-closed route-contract tests covering low-confidence abstention, scorer failures, malformed runtime outputs, direct configuration validation, queue mapping, schema surface, and exact inclusive threshold semantics.
- Regression tests protecting calibration fold balance, method selection, guardrail outcomes, bounded reproducibility, frozen routing selection, read-only workflow permissions, and confirmatory-test isolation.
- Execution-integrity checks covering code correctness, metric interpretation, split integrity, reproducibility, dependency/hardware consistency, CI behavior, and public claim wording.
- One registered BANKING77 confirmatory evaluation for the frozen A2 router, with official-test macro-F1 0.9016, top-3 recall 0.9744, ECE 0.0169, Brier 0.1456, and immutable execution provenance.
- Confirmatory selective-routing result showing routing error risk of 1.95% at 75% confidence-ranked coverage versus 9.84% under full automation, with paired-bootstrap 95% CI for the risk difference of [-8.78 pp, -6.96 pp].
- Confirmatory in-domain calibration-cost comparison recorded as inconclusive; the development mixed in-domain/OOS cost result remains unsupported and the reused development OOS set is not relabeled as independent confirmation.
- Independent post-result verification reproducing event counts, routing costs, thresholds, bootstrap intervals, split integrity, and registered verdicts without post-test tuning.
- Public/private boundary audit strengthened so internal research bibliographies, findings notes, planning language, workstation paths, and private artifacts are rejected from the public repository.

### Changed

- Public documentation now focuses on technical methods, reproducibility, measured results, limitations, and product boundaries rather than internal project-management or publication-strategy material.

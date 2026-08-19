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
- Frozen Phase 3 retrieval protocol covering B0 BM25, B1 dense retrieval, B2 hybrid RRF, and B3 hybrid plus cross-encoder reranking, with registered relevance, inference, latency, and complexity-adoption rules.
- Repository-owned deterministic retrieval implementations and typed model-adapter boundaries completed before opening the frozen retrieval benchmark.
- Registered R3.2 retrieval execution over 308 queries and 147 eligible HelixBank documents, with raw rankings, per-query metrics, five-pass latency samples, exact model revisions, input hashes, and permanent GitHub Actions provenance.
- Independent automated reconstruction of aggregate metrics, per-query metrics, bootstrap intervals, latency summaries, diagnostic slices, and the registered complexity-selection decision.
- Additional manual code/result audit confirming ranking shape, deterministic tie behavior, RRF composition, reranker scope, latency stability, and checksum integrity, with an audit-only intent-cluster bootstrap sensitivity check.
- Frozen `retrieval-selected-v1` configuration selecting deterministic B0 BM25 after no more-complex candidate satisfied the predeclared adoption rule.
- FastAPI `POST /v1/search` integration bound to `retrieval-selected-v1`, with strict input validation, pre-ranking evidence eligibility, deterministic response serialization, bounded top-50 retrieval, and a stable non-leaking backend-failure response.
- R3.3 integration tests covering selected-configuration alignment, archived-document exclusion, zero-score tie breaking, deterministic HTTP output, invalid input, OpenAPI surface, and search-backend failure behavior.
- Frozen Phase 4 A4.0 evidence-grounded assistance protocol with intent-clustered development/confirmatory partitions, explicit evidence-sufficiency decisions, sentence-level grounding and citation metrics, a bounded G0-G2 candidate ladder, independent verifier requirements, adversarial prompt-injection/citation-spoof/staleness tests, cluster-bootstrap inference, latency/cost accounting, and no post-result rescue rule.
- Phase 4 assistance evaluation-record schema and protocol guard tests that prevent generation scoring before the separate A4.1 model/prompt/verifier/budget binding checkpoint is merged.
- Frozen A4.1 assistance runtime binding with an immutable generator snapshot, exact prompt hashes, independent pinned NLI verifier families, deterministic development-only diagnostic subsets, dated pricing, fixed latency/cost ceilings, and a benchmark-scoped dependency lock; the binding preflight makes no model calls and produces no assistance performance scores.

### Changed

- Public documentation now focuses on technical methods, reproducibility, measured results, limitations, and product boundaries rather than internal project-management or publication-strategy material.
- Retrieval documentation now distinguishes the evaluated B0-B3 ladder from the selected runtime configuration: B3 improved graded ranking relevance but failed its registered P95 latency budget, so the deployed Phase 3 search path remains deterministic B0 BM25.

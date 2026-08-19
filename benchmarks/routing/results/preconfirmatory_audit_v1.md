# Phase 2 Final Pre-Confirmatory Integrity Verification

> Status: **passed**. The official BANKING77 confirmatory test remained unopened at this checkpoint.

## Results

| Verification surface | Result |
|---|---|
| Release CI | **passed** |
| Ruff lint / format | passed |
| Strict mypy | passed |
| Pytest | **77 passed** |
| Phase 1 offline data contracts | passed |
| Publication boundary check | passed |
| Pre-confirmatory integrity manifest | **35 artifacts verified** |
| Scientific no-test preflight | **passed** |
| Confirmatory execution mode | manual, read-only |
| Selected model | A2 |
| Temperature | `0.457974` |
| Selected threshold | `0.892704` |
| Raw A2 H3 comparator threshold | `0.367217` |
| Registered bootstrap | 5,000 paired replicates, seed `20260819` |
| H3 confirmatory scope | independent BANKING77 in-domain component only |
| Confirmatory test opened | **false** |

Release CI evidence: workflow run `32197909626`. No-test scientific preflight evidence: workflow run `32197909434`.

## What this verification establishes

The Phase 2 development and implementation surface was frozen before confirmatory evaluation. A machine-readable Git-blob manifest pins 35 selection-critical scientific and execution artifacts, including the data contract, A2 configuration, calibration and cost contracts, OOS benchmark, selected router, authoritative development results, route schemas, confirmatory protocol, evaluator, workflow, and relevant tests.

The confirmatory workflow verifies this manifest before executing a no-test preflight. Repository permissions are read-only and there is no automatic pull-request or push test-opening path.

The separate preflight loads the confirmatory evaluator inside the CPU-only A2 scientific environment without opening the test set. Its final successful run reported `preflight_passed` and `test_set_opened=false`.

## Integrity findings

### 1. H3 independence scope required narrowing

Development expected routing cost mixes BANKING77 in-domain cases with the 160-query synthetic OOS benchmark. That OOS benchmark was frozen before development scoring, but it was later inspected and used during operating-policy selection. It therefore cannot be reused as unseen confirmatory evidence.

The correction was made **before the test was opened**. H3's confirmatory estimand is explicitly the independent BANKING77-test **in-domain cost component**: frozen calibrated A2 policy minus frozen raw-A2 comparator. The original mixed in-domain/OOS development endpoint cannot be declared fully confirmed by the BANKING77 test alone because Phase 2 has no unseen OOS confirmatory sample.

This correction does not change the development H3 result. H3 remains unsupported on the registered development mixed-cost endpoint.

### 2. Confirmatory execution required an executable freeze

A narrative freeze is weaker than an executable guarantee. `routing-preconfirmatory-freeze-v1` therefore records Git-blob identities for the frozen scientific surface, and a standard-library verifier blocks confirmatory execution if a listed artifact drifts.

The final preflight verified all **35** registered artifacts successfully.

### 3. Root-package and scientific-runtime boundaries were separated

The root Helix package intentionally has no NumPy, scikit-learn, PyTorch, or sentence-transformers runtime dependency. Early protocol tests attempted to import the scientific evaluator from generic root pytest, which would have blurred that architecture boundary.

Generic tests remain dependency-free and executable scientific preflight runs in its own locked environment. This preserves the production-shaped domain boundary while still exercising the evaluator under its real scientific dependency graph.

### 4. Preflight found a BANKING77 contract-path defect

The first executable preflight failed before test access because the evaluator looked for derived hashes under `data["split"]["expected"]`. The frozen Phase 1 contract stores `expected` at the top level. The path was corrected to `data["expected"]`.

This was a substantive code correction even though no confirmatory scientific result had yet been generated.

### 5. Mechanical CI defects were corrected

The verification also corrected Ruff formatting/line-length issues and removed a self-referential workflow assertion. None of these repairs changed the model, calibration, OOS benchmark, costs, thresholds, hypotheses, or development results.

## Scientific position before confirmatory evaluation

Nothing in this verification changes the frozen development evidence:

- A2 remains the selected model;
- A3 remains a registered negative result;
- temperature scaling remains the selected calibration method;
- H3 development remains **unsupported** for A2 expected routing cost;
- H4 development remains **supported** for selective abstention;
- frozen application temperature remains `0.457974`;
- frozen application threshold remains `0.892704`;
- A4 remains disabled.

The confirmatory test cannot alter any of those selections.

## Reproducibility value

This verification strengthens the credibility of the final result by preventing three common applied-ML weaknesses: changing selection-critical artifacts between development and test, treating reused benchmark data as independent confirmation, and running confirmatory evaluation through code that has not executed in its declared dependency environment.

The explicit H3 scope limitation is intentionally conservative. A narrower independently estimable claim is more defensible than a broader mixed endpoint whose OOS component has already participated in development selection.

The integrity manifest also gives external reviewers a concrete reproducibility object: the confirmatory result can be tied to an exact frozen scientific surface.

## Remaining limitations

Phase 2 lacks an independent unseen OOS confirmatory sample, so the original mixed in-domain/OOS H3 endpoint cannot receive a fully independent confirmatory verdict in this version. The synthetic routing costs remain scenario assumptions rather than estimates of real-bank economics or customer harm.

The confirmatory test evaluates the frozen BANKING77 domain. It does not establish live serving latency, production drift resistance, real traffic OOS prevalence, or deployment impact.

## Decision

**Final pre-confirmatory integrity verification: PASSED.**

The frozen routing configuration was technically and methodologically ready for its registered BANKING77 confirmatory evaluation. The test remained sealed at this checkpoint.

## Subsequent evaluation

The registered confirmatory evaluation was executed against this frozen configuration and is reported separately in `confirmatory_test_v1.md`, with an independent post-result verification in `confirmatory_post_audit_v1.md`.

# Phase 2 Final Pre-Confirmatory Audit

> Gate status: **passed**. The official BANKING77 confirmatory test remains unopened. Phase 3 retrieval remains forbidden.

## Results

| Audit surface | Result |
|---|---|
| Release CI | **passed** |
| Ruff lint / format | passed |
| Strict mypy | passed |
| Pytest | **77 passed** |
| Phase 1 offline data contracts | passed |
| Publication audit | passed |
| Pre-confirmatory integrity manifest | **35 artifacts verified** |
| Locked scientific no-test preflight | **passed** |
| Confirmatory workflow | manual-only, read-only |
| Selected model | A2 |
| Temperature | `0.457974` |
| Selected threshold | `0.892704` |
| Raw A2 H3 comparator threshold | `0.367217` |
| Registered bootstrap | 5,000 paired replicates, seed `20260819` |
| H3 confirmatory scope | independent BANKING77 in-domain component only |
| Confirmatory test opened | **false** |
| Phase 3 started | **false** |

Release CI evidence: workflow run `32197909626`. No-test scientific preflight evidence: workflow run `32197909434`.

## What the gate establishes

The complete Phase 2 development and implementation surface is now frozen before the one-shot confirmatory test. A machine-readable Git-blob manifest pins 35 selection-critical scientific and execution artifacts, including the data contract, A2 configuration, calibration and cost contracts, OOS benchmark, selected router, authoritative development results, route schemas, confirmatory protocol, evaluator, workflow, and relevant tests.

The manual confirmatory workflow verifies this manifest before executing a no-test preflight. Only after those checks pass can an explicitly authorized test-access step run. The workflow has read-only repository permissions and no pull-request or push trigger.

The separate preflight workflow loads the confirmatory evaluator inside the already-audited CPU-only A2 scientific environment while remaining incapable of authorizing test access. Its final successful run reported `preflight_passed` and `test_set_opened=false`.

## Hostile-audit findings

### 1. H3 independence scope was too broad

The most important finding was methodological. Development expected routing cost mixes BANKING77 in-domain cases with the 160-query synthetic OOS benchmark. That OOS benchmark was legitimately frozen before development scoring, but it was later inspected and used during operating-policy selection. It therefore cannot be reused as unseen confirmatory evidence.

The repair was made **before the test was opened**. H3's registered confirmatory estimand is now explicitly the independent BANKING77-test **in-domain cost component**: frozen calibrated A2 policy minus frozen raw-A2 comparator. The original mixed in-domain/OOS development endpoint cannot be declared fully confirmed by the BANKING77 test alone because Phase 2 has no unseen OOS confirmatory sample.

This correction does not change the development H3 result. H3 remains unsupported on the registered development mixed-cost endpoint. It prevents a favorable future BANKING77 test result from creating an overstated publication claim.

### 2. Confirmatory execution needed a hard freeze

Prose saying that a configuration is frozen is weaker than an executable guarantee. The audit therefore added `routing-preconfirmatory-freeze-v1`, a Git-blob integrity manifest, plus a stdlib-only verifier. Any listed artifact drift blocks the confirmatory workflow before test access.

The final preflight verified all **35** registered artifacts successfully.

### 3. Root-package and scientific-runtime boundaries were mixed

The root Helix package intentionally has no NumPy, scikit-learn, PyTorch, or sentence-transformers runtime dependency. Early protocol tests attempted to import the scientific evaluator from generic root pytest, which would have blurred that architecture boundary.

The repair keeps generic tests dependency-free and moves executable scientific preflight into its own locked workflow. This preserves the production-shaped domain boundary while still exercising the evaluator under its real scientific dependency graph.

### 4. Preflight found a real BANKING77 contract-path bug

The first executable preflight failed before test access because the evaluator looked for derived hashes under `data["split"]["expected"]`. The frozen Phase 1 contract stores `expected` at the top level. The path was corrected to `data["expected"]`.

This is a substantive code correction even though no scientific result had yet been generated. The preflight exists precisely to catch this class of defect before the one-shot test is consumed.

### 5. Mechanical CI defects were corrected

The audit also corrected Ruff formatting/line-length issues and removed a self-referential workflow assertion that searched the preflight workflow for a forbidden authorization string and then matched the string inside its own search command. None of these repairs changed the model, calibration, OOS benchmark, costs, thresholds, hypotheses, or development results.

## Scientific position before test open

Nothing in this gate changes the frozen development evidence:

- A2 remains the selected model;
- A3 remains a registered negative result;
- temperature scaling remains the selected calibration method;
- H3 development remains **unsupported** for A2 expected routing cost;
- H4 development remains **supported** for selective abstention;
- frozen application temperature remains `0.457974`;
- frozen application threshold remains `0.892704`;
- A4 remains disabled.

The confirmatory test cannot alter any of those selections.

## Methodological and publicability value

This gate materially improves the credibility of the final Phase 2 result. It prevents three common weaknesses in applied-ML research: silently changing artifacts between development and test, treating reused benchmark data as independent confirmation, and running a one-shot test through code that has never executed in its real dependency environment.

The explicit H3 scope limitation is particularly valuable for publication. It makes the paper or technical report slightly less sweeping, but substantially more defensible. A narrower claim with a demonstrably independent estimand is stronger than a broader claim whose OOS component has already participated in selection.

The manifest also gives reviewers a concrete reproducibility object: the eventual confirmatory result can be tied to an exact frozen scientific surface rather than to a narrative description of what supposedly did not change.

## Remaining limitations

Phase 2 still lacks an independent unseen OOS confirmatory sample, so the original mixed in-domain/OOS H3 endpoint cannot receive a fully independent confirmatory verdict in this version. The synthetic routing costs also remain scenario assumptions, not estimates of real-bank economics or customer harm.

The confirmatory test will establish performance only for the frozen BANKING77 domain. It will not prove live serving latency, production drift resistance, real traffic OOS prevalence, or deployment impact.

These are limitations to publish, not reasons to reopen Phase 2 model selection after test access.

## Decision

**Final pre-confirmatory audit: PASSED.**

The one-shot confirmatory evaluation is technically and methodologically ready for an explicit authorization decision. The test remains sealed at this checkpoint.

## Next locked action

After explicit approval, run the **manual one-shot registered BANKING77 confirmatory evaluation** using the exact authorization token. Then permanently record the result, perform the mandatory post-result hostile audit, close the Phase 2 exit report, and make the Phase 2 merge decision.

**Phase 3 retrieval remains forbidden until Phase 2 is formally closed.**

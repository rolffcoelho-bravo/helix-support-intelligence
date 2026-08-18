# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: Active — A0/A1/A2 checkpoints passed; A3 next
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
  - [x] A0/A1 development benchmark and evidence checkpoint.
  - [x] A2 sentence-embedding + linear-classifier benchmark and audited evidence checkpoint.
  - [ ] A3 compact transformer classifier.
- [ ] Calibration comparison.
- [ ] Frozen out-of-scope benchmark and evaluation.
- [ ] Declared routing cost matrix.
- [ ] Final risk-coverage report and operating-point selection.
- [ ] Router model card.
- [ ] `/v1/tickets/route` contract tests.
- [ ] One frozen routing configuration and operating threshold.
- [ ] Registered confirmatory result for H3 and H4.

## A0/A1 checkpoint

The first development benchmark was executed twice in independent GitHub Actions runs against the frozen train/validation partitions only. The generated result JSON, Markdown report, and all A1 validation predictions were byte-identical across both runs.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE |
|---|---:|---:|---:|---:|
| A0 most-frequent | 0.0005 | 0.0130 | 0.0491 | 0.9813 |
| A0 stratified | 0.0163 | 0.0162 | 0.0471 | 0.9833 |
| **A1 TF-IDF + logistic regression** | **0.8422** | **0.8407** | **0.9534** | **0.4895** |

A1 validation accuracy is 0.8563 while mean maximum probability is 0.3667, showing substantial under-confidence. Raw A1 confidence nevertheless orders risk usefully: selective risk is approximately 2.53% at 50% coverage and 6.07% at 70% coverage. No operating threshold is frozen from this checkpoint.

Permanent development evidence is recorded in:

- `benchmarks/routing/results/a0_a1_validation_v1.json`
- `benchmarks/routing/results/a0_a1_validation_v1.md`

## A2 checkpoint

A2 was frozen before evaluation as `sentence-transformers/all-MiniLM-L6-v2` at revision `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`, used only as a 384-dimensional normalized sentence-embedding feature extractor. The linear classifier retained the fixed A1 logistic-regression specification. No alternate encoder or hyperparameter search was allowed.

The audited A2 environment explicitly resolves `torch 2.13.0+cpu` from the PyTorch CPU index, reports `torch_cuda_available=false`, and contains no CUDA, NVIDIA, or Triton packages in the committed lock.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| A1 | 0.8422 | 0.8407 | 0.9534 | 0.4895 | 0.4951 |
| **A2 frozen embeddings + logistic regression** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |
| **A2 − A1** | **+0.0564** | **+0.0556** | **+0.0197** | **−0.1986** | **−0.2450** |

A2 improves selective risk at every registered coverage point. At 50% coverage, risk falls from approximately 2.53% for A1 to 0.40% for A2; at 70%, from approximately 6.07% to 1.59%. These are development observations only and do not select an operating threshold.

Two independent post-audit GitHub Actions runs reproduced all 1,976 predicted intents, all top-3 intent sets, all discrete classification metrics, and every registered selective-risk value exactly. Raw probabilities were not bitwise identical across heterogeneous CPU runners: the maximum absolute confidence difference was approximately `6.74e-7`, ECE differed by approximately `8.78e-9`, and Brier by approximately `5.51e-10`. The checkpoint therefore claims exact decision reproducibility plus bounded numerical reproducibility, not byte-identical ML output.

A2 adds a neural embedding stage. On the two audited GitHub CPU runners, validation embedding ranged from approximately 2.85 to 5.68 ms per example. That spread is too large to use as a stable latency claim, so a standardized A1/A2/A3 latency and cost harness remains required.

Permanent A2 evidence is recorded in:

- `benchmarks/routing/results/a2_validation_v2.json`
- `benchmarks/routing/results/a2_validation_v2.md`
- `benchmarks/routing/evaluate_a2.py.lock`

## Post-execution audit

The A2 checkpoint was not accepted immediately after the first green benchmark. A hostile audit identified and corrected three issues before closure:

1. **Confusion-count bug:** the first comparison treated absence from A2's top-20 confusion list as zero occurrences. The audited implementation now counts A2 occurrences across the full validation prediction set for every leading A1 confusion pair.
2. **Environment mismatch:** the first lock used a CUDA-enabled PyTorch distribution even though inference executed on CPU. The script and config now pin the CPU-only PyTorch index, and the lock was regenerated and verified.
3. **Overstated reproducibility:** the first report used the phrase "byte-identical" for ML outputs. The audited reruns showed exact decisions but sub-micro probability differences across heterogeneous CPU runners, so the public claim was narrowed to exact decision reproducibility plus bounded numerical reproducibility.

The superseded A2 v1 evidence files were removed. The aggregate A2 metrics, model ranking, and risk-coverage conclusions were unchanged by these corrections.

This audit pattern is now a standing execution rule: each implementation checkpoint must end with code, metric, reproducibility, CI, and public-wording review before it is declared complete.

## Test-set status

The official BANKING77 test split remains confirmatory and has not been downloaded or opened by the Phase 2 routing benchmarks. It remains unauthorized for model selection, calibration selection, feature selection, error-driven tuning, or operating-threshold selection.

## Current decision

**A2 survives and is the leading Phase 2 development candidate.** A1 remains the required simpler classical reference. A2's gain is large enough that A3 must now justify its additional training and inference complexity against A2 rather than merely outperforming A1.

Phase 2 remains open. The next locked implementation action is **A3 — one frozen compact transformer classifier** under the identical train/validation contract. Its base checkpoint, tokenizer, training budget, seed, optimization specification, and early-stopping rule must be frozen before the first A3 result. Do not open the confirmatory test and do not introduce a classifier family outside the frozen ladder.

# Phase 2 Exit Report

- Phase: Routing baseline and selective decision policy
- Status: Active — A0/A1/A2/A3 ladder complete; calibration next
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
- [x] Reproducible A0-A3 implementations and evaluation artifacts.
  - [x] A0/A1 development benchmark and evidence checkpoint.
  - [x] A2 sentence-embedding + linear-classifier benchmark and audited evidence checkpoint.
  - [x] A3 compact-transformer benchmark and audited negative-result checkpoint.
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

Permanent A2 evidence is recorded in:

- `benchmarks/routing/results/a2_validation_v2.json`
- `benchmarks/routing/results/a2_validation_v2.md`
- `benchmarks/routing/evaluate_a2.py.lock`

## A3 checkpoint

A3 tested end-to-end task-specific fine-tuning of the same MiniLM transformer used by A2. Its fixed contract used attention-masked mean pooling, a new 384-to-77 linear head, full encoder fine-tuning, AdamW at `2e-5`, and a fixed three-epoch budget with no best-epoch selection and no post-result rescue configuration.

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| **A2** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |
| A3 | 0.6898 | 0.7105 | 0.9226 | 0.7221 | 0.9771 |
| **A3 − A2** | **−0.2088** | **−0.1858** | **−0.0506** | **+0.4311** | **+0.7270** |

A3 is also worse than A2 at every registered risk-coverage point. Its mean maximum probability is only 0.0183, close to a uniform 77-class distribution. Training macro-F1 rose monotonically from 0.5117 to 0.6624 to 0.6898 across the three fixed epochs, while mean training loss declined from 4.2488 to 4.0448.

The hostile audit found no code, split, truncation, dependency, or reproducibility defect that invalidates the result. Two independent GitHub CPU replicas produced exactly the same predictions, top-3 sets, selective-risk curve, ECE, Brier, and confidence values. Only one training example exceeded the 96-token limit and no validation example did.

The correct interpretation is deliberately narrow: **A3 as registered underfits and does not justify its additional complexity.** The result does not establish that task-specific transformer fine-tuning is intrinsically inferior. Changing the learning rate, head-specific optimization, epoch budget, pooling, or backbone after observing this result would violate the frozen anti-shopping rule.

Permanent A3 evidence is recorded in:

- `benchmarks/routing/results/a3_validation_v1.json`
- `benchmarks/routing/results/a3_validation_v1.md`
- `benchmarks/routing/evaluate_a3.py.lock`

**Decision:** A3 is rejected. A2 remains the leading development candidate. A1 remains the required simpler reference. A4 remains disabled.

## Post-execution audit rule

Every checkpoint must close with a hostile audit covering code correctness, metric interpretation, split integrity, reproducibility, dependency/hardware consistency, CI behavior, and public wording. Findings must be corrected before the checkpoint is declared complete.

## Test-set status

The official BANKING77 test split remains confirmatory and has not been downloaded or opened by the Phase 2 routing benchmarks. It remains unauthorized for model selection, calibration selection, feature selection, error-driven tuning, or operating-threshold selection.

## Current decision

The required A0-A3 model ladder is now complete. **A2 is the surviving leading candidate, A1 is the simpler reference, and A3 is rejected as registered.**

Phase 2 remains open. The next locked action is the **calibration comparison**. Calibration-method evaluation must avoid fitting and scoring a calibrator on the same validation observations; the comparison will therefore use deterministic cross-fitted calibration on the frozen validation partition, followed by a single full-validation fit only after the method is selected. Rejected A3 is not eligible for calibration-based resurrection. The confirmatory test remains sealed.

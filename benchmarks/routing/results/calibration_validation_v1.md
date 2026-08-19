# Phase 2 Routing Calibration Checkpoint

> **Validation-only development evidence.** The confirmatory BANKING77 test split was not downloaded, opened, scored, or used for calibration selection.

## Decision

**Temperature scaling is selected for both A1 and A2.** A2 plus temperature scaling becomes the leading calibrated Phase 2 development candidate. No automation threshold is selected here, and this checkpoint does not establish H3 because expected routing cost remains unevaluated.

## Cross-fitting contract

Calibration was evaluated with deterministic five-fold cross-fitting inside the frozen 1,976-row validation partition. For every scored row, the calibrator was fit on the other four folds only. The final fold counts are:

| Fold | Rows |
|---|---:|
| 0 | 390 |
| 1 | 392 |
| 2 | 393 |
| 3 | 402 |
| 4 | 399 |

The assignment is intent-stratified. Within each intent, rows are ordered deterministically from the frozen salt and stable sample ID. A deterministic start offset derived from the same pre-frozen salt and the intent name distributes remainder rows across folds.

## A1 calibration

| Method | Macro-F1 | Top-3 recall | ECE | Brier | NLL | Guardrail |
|---|---:|---:|---:|---:|---:|---|
| Raw | 0.8422 | 0.9534 | 0.4895 | 0.4951 | 1.3666 | reference |
| **Temperature scaling** | **0.8422** | **0.9534** | **0.0263** | **0.2150** | **0.5403** | **pass** |
| Isotonic regression | 0.8319 | 0.9423 | 0.0518 | 0.2461 | 1.6839 | fail |
| Platt scaling | 0.8477 | 0.9555 | 0.0898 | 0.2314 | 0.5920 | pass |

Temperature scaling has the lowest cross-fitted Brier score among guardrail-valid candidates. The full-validation refit temperature, stored for the later frozen router configuration, is approximately **0.346418**.

## A2 calibration

| Method | Macro-F1 | Top-3 recall | ECE | Brier | NLL | Guardrail |
|---|---:|---:|---:|---:|---:|---|
| Raw | 0.8986 | 0.9732 | 0.2910 | 0.2501 | 0.6683 | reference |
| **Temperature scaling** | **0.8986** | **0.9732** | **0.0162** | **0.1398** | **0.3350** | **pass** |
| Isotonic regression | 0.8898 | 0.9615 | 0.0126 | 0.1571 | 1.2764 | fail |
| Platt scaling | 0.8948 | 0.9691 | 0.0305 | 0.1497 | 0.4022 | fail |

A2 temperature scaling reduces ECE from approximately **0.2910 to 0.0162**, Brier from **0.2501 to 0.1398**, and NLL from **0.6683 to 0.3350** while preserving A2 macro-F1 and top-3 recall exactly on the frozen validation partition.

The full-validation refit temperature is approximately **0.457974**.

## Selective-routing observation

Temperature scaling does more than rescale a single scalar confidence uniformly across examples. Although it preserves each example's class ordering, it can change the ranking of maximum confidence across examples because the complete class-probability vector differs from case to case. Accordingly, the risk-coverage ordering is re-evaluated rather than assumed unchanged.

For calibrated A2, selective risk is approximately:

| Coverage | Selective risk |
|---|---:|
| 10% | 0.00% |
| 20% | 0.00% |
| 30% | 0.17% |
| 40% | 0.25% |
| 50% | 0.20% |
| 60% | 0.51% |
| 70% | 1.08% |
| 80% | 2.15% |
| 90% | 4.78% |
| 100% | 9.77% |

These values do **not** define an operating threshold. Threshold selection remains downstream of the OOS and cost-aware evidence.

## Hostile audit

The first successful calibration implementation was not frozen immediately. The end-of-execution audit found that the original intent-stratified round-robin assignment always started each intent at fold zero, producing score-fold sizes of **430 / 411 / 392 / 380 / 363**. Every row was still scored once and every fold retained all intents, so the pre-audit result was methodologically usable, but the training-fold sizes were unnecessarily uneven.

The correction was defined without looking at candidate performance: each intent receives a deterministic start offset derived only from the already-frozen salt and intent name. A second audit then caught that an initial workflow-based source rewrite had failed to modify the script even though the configuration had changed. The workflow was hardened to verify the actual source expression, the corrected source was committed, and the calibration benchmark was rerun.

The final balanced-fold result retained the same winner: temperature scaling for both A1 and A2. The workflow was then restored to read-only permissions.

## Reproducibility

Two independent runs from the corrected source and committed CPU-only lock produced:

- identical sample order;
- identical fold assignments;
- identical raw and calibrated predicted intents for all 1,976 rows;
- identical discrete classification metrics;
- identical registered selective-risk curves;
- the same selected calibration method for A1 and A2.

As with the audited A2 checkpoint, floating-point probabilities are not claimed to be bitwise identical across heterogeneous CPU runners. For A2, the maximum absolute calibrated-confidence difference was approximately **1.41×10⁻⁶**; the cross-fitted ECE, Brier, and NLL differences were on the order of `10^-9`.

## Scientific boundary

This checkpoint supports the statement that temperature scaling substantially improves held-out validation calibration for the current A1 and A2 candidates without degrading their classification metrics. It does **not** establish that calibration reduces damaging automatic routes in operational terms. H3's registered primary endpoint is expected routing cost, and that analysis has not yet been performed.

The README release benchmark remains `pending`.

## Next locked action

Freeze the **Phase 2 out-of-scope benchmark before inspecting any OOS model result**, then evaluate A1 and calibrated A2 on that frozen benchmark. The official BANKING77 test split remains sealed.

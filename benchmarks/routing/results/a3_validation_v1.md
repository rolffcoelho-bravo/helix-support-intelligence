# Phase 2 A3 Validation Checkpoint

> **Development evidence only.** The confirmatory BANKING77 test split was not downloaded, opened, scored, or used for selection.

## Frozen A3

A3 tested whether end-to-end task-specific fine-tuning of the same MiniLM transformer used by A2 would add value over frozen sentence embeddings.

The contract was frozen before the first result:

- `sentence-transformers/all-MiniLM-L6-v2` at revision `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`;
- attention-masked mean pooling with L2 normalization;
- trainable 384-to-77 linear classification head;
- all encoder parameters trainable;
- fixed 3-epoch budget;
- AdamW, learning rate `2e-5`, weight decay `0.01`, 10% warm-up;
- train batch 32, validation batch 64;
- max sequence length 96;
- seed `20260818`;
- no best-epoch selection, no early stopping, no hyperparameter search, and no post-result rescue configuration.

## Result

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| **A2 — frozen MiniLM + logistic regression** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |
| A3 — end-to-end MiniLM fine-tuning | 0.6898 | 0.7105 | 0.9226 | 0.7221 | 0.9771 |
| **A3 − A2** | **−0.2088** | **−0.1858** | **−0.0506** | **+0.4311** | **+0.7270** |

A3 validation accuracy is **0.7404**. Its mean maximum predicted probability is only **0.0183**, close to the uniform 77-class probability of roughly 0.013. Raw probability calibration is therefore extremely poor.

## Training trajectory

A3 improved monotonically across the fixed budget but remained far below A2:

| Epoch | Mean train loss | Validation macro-F1 | Balanced accuracy | Top-3 recall |
|---|---:|---:|---:|---:|
| 1 | 4.2488 | 0.5117 | 0.5507 | 0.8376 |
| 2 | 4.0946 | 0.6624 | 0.6831 | 0.9114 |
| 3 | 4.0448 | 0.6898 | 0.7105 | 0.9226 |

The 77-class uniform cross-entropy scale is approximately `ln(77) ≈ 4.34`. The loss trajectory therefore shows learning, but also substantial residual underfitting at the end of the registered budget.

This distinction matters. The valid conclusion is **not** that transformer fine-tuning is generally inferior. The valid conclusion is that **this frozen A3 configuration does not justify its additional complexity**.

## Selective routing

A3 is worse than A2 at every registered coverage point.

| Coverage | A2 risk | A3 risk |
|---|---:|---:|
| 10% | 0.00% | 3.03% |
| 30% | 0.17% | 6.24% |
| 50% | 0.40% | 10.32% |
| 70% | 1.59% | 15.47% |
| 90% | 5.46% | 21.88% |
| 100% | 9.77% | 25.96% |

No operating threshold is selected here.

## Error structure

A3 improves some A2 confusion pairs, including `declined_transfer → failed_transfer` (6 → 2) and `verify_my_identity → why_verify_identity` (4 → 0), but materially worsens others. Examples include:

- `beneficiary_not_allowed → failed_transfer`: 6 → 10;
- `wrong_exchange_rate_for_cash_withdrawal → card_payment_wrong_exchange_rate`: 4 → 12;
- `pending_top_up → top_up_reverted`: 27 A3 errors;
- `top_up_failed → top_up_reverted`: 23 A3 errors.

The aggregate deterioration dominates the local improvements.

## Audit

The hostile checkpoint audit found no execution defect that invalidates the negative result:

- frozen train and validation hashes were revalidated;
- the confirmatory test was never downloaded;
- all 77 labels were present;
- only one training example exceeded the 96-token limit and no validation example did;
- the committed dependency lock resolves CPU-only PyTorch;
- two independent GitHub-hosted CPU replicas produced exactly identical predictions, top-3 sets, selective-risk values, ECE, Brier, and confidence values;
- the code uses validation only diagnostically because epoch 3 is fixed in advance.

The audit did identify **underfitting as the correct interpretation**. Because the training budget and learning rate were frozen before evaluation, changing them now would be post-result model shopping. A3 is therefore rejected as registered rather than remediated.

## Decision

**A3 does not survive Phase 2 model selection. A2 remains the leading development candidate and A1 remains the required simpler reference.**

The A0–A3 model ladder is now complete. Phase 2 remains open because calibration, OOS evaluation, routing cost, the final risk-coverage operating point, model card, route contract tests, and the confirmatory evaluation are still pending.

## Next locked action

Freeze and execute the **calibration comparison** for A2, with A1 retained as the simpler reference. Calibration-method evaluation must use leakage-safe cross-fitted validation predictions rather than fitting and scoring a calibrator on the same observations. Rejected A3 is not eligible for calibration-based resurrection. The confirmatory test remains sealed.

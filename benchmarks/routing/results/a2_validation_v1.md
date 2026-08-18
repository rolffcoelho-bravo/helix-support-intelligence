# Phase 2 A2 Validation Checkpoint

> **Development evidence only.** The confirmatory BANKING77 test split was not downloaded, opened, scored, or used for selection.

## Frozen A2

A2 was specified before any A2 validation result existed:

- encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- encoder revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`;
- 384-dimensional normalized sentence embeddings;
- no fine-tuning;
- CPU execution, embedding batch size 64;
- logistic regression with the same `C=1.0`, `lbfgs`, `max_iter=1000`, and `tol=1e-4` linear-classifier specification used to keep the A1/A2 comparison representation-focused;
- no model shopping and no hyperparameter search.

The full transitive script environment is committed in `benchmarks/routing/evaluate_a2.py.lock`.

## Result

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE, 15 bins | Multiclass Brier |
|---|---:|---:|---:|---:|---:|
| A1 — TF-IDF + logistic regression | 0.8422 | 0.8407 | 0.9534 | 0.4895 | 0.4951 |
| **A2 — frozen MiniLM embeddings + logistic regression** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |
| **A2 minus A1** | **+0.0564** | **+0.0556** | **+0.0197** | **−0.1986** | **−0.2450** |

A2 validation accuracy is **0.9023**. Mean maximum probability rises from A1's 0.3667 to **0.6114**, while accuracy also rises. A2 therefore remains under-confident on this development partition, but the probability scale is substantially less distorted than A1's raw output.

This does not select a calibration method. Calibration remains a later Phase 2 comparison after A3.

## Selective routing

A2 improves selective risk at every registered coverage point.

| Coverage | A1 risk | A2 risk | Change |
|---|---:|---:|---:|
| 10% | 0.51% | **0.00%** | −0.51 pp |
| 20% | 0.76% | **0.25%** | −0.51 pp |
| 30% | 1.01% | **0.17%** | −0.84 pp |
| 40% | 1.90% | **0.13%** | −1.77 pp |
| 50% | 2.53% | **0.40%** | −2.13 pp |
| 60% | 4.38% | **0.51%** | −3.88 pp |
| 70% | 6.07% | **1.59%** | −4.48 pp |
| 80% | 8.10% | **2.91%** | −5.19 pp |
| 90% | 10.01% | **5.46%** | −4.56 pp |
| 100% | 14.37% | **9.77%** | −4.61 pp |

No automation threshold is selected from this table. It establishes only that A2's raw confidence ordering is materially more useful than A1's on the frozen validation partition.

## Error structure

A2 eliminates several important A1 directional confusions from the top-error set:

- `compromised_card → lost_or_stolen_card`: 4 → 0;
- `top_up_failed → top_up_reverted`: 4 → 0;
- `topping_up_by_card → top_up_by_cash_or_cheque`: 4 → 0;
- `pending_cash_withdrawal → declined_cash_withdrawal`: 3 → 0;
- `card_payment_fee_charged → extra_charge_on_statement`: 3 → 0.

The stronger semantic representation does not solve every adjacent-intent problem. The principal A2 errors now include:

- `declined_transfer → failed_transfer`: 6;
- `beneficiary_not_allowed → failed_transfer`: 6;
- `wrong_exchange_rate_for_cash_withdrawal → cash_withdrawal_charge`: 5;
- `balance_not_updated_after_bank_transfer → transfer_not_received_by_recipient`: 5;
- `extra_charge_on_statement → card_payment_fee_charged`: 4;
- `verify_my_identity → why_verify_identity`: 4.

This is useful for A3 evaluation. A3 must not merely improve aggregate F1; it should demonstrate whether end-to-end contextual fine-tuning resolves the remaining state, authorization, identity, and exchange-rate distinctions.

## Complexity and latency

A2 adds a neural encoder that A1 does not require. On two GitHub-hosted CPU runs, validation encoding took approximately **5.57–5.71 ms per example**, excluding model loading and classifier prediction. Training-set embedding took approximately 41.6–42.8 seconds, while the linear classifier fit took approximately 1.20–1.22 seconds.

These timings are descriptive, not deterministic evidence, and no standardized A1 latency measurement exists yet under the same timing harness. A2 therefore does not win the final complexity/cost criterion at this checkpoint. The final router still requires standardized end-to-end latency and cost evidence.

## Reproducibility

Two independent GitHub Actions executions produced byte-identical deterministic outputs:

- `results.json`: `d55a72ec8e70b83ca6e7f53216a0cb817330171f32bc87cc804609629de0a973`;
- generated report: `b34f69fb74a95f5896a63d4b313af11bc2a2581f290b4281d43676b30279b643`;
- all 1,976 A2 validation predictions: `0249b3d613e780581396b9fba6c4848b868c418de4c84749d3c1a672359b3a04`;
- uv script lock: `59504f7f801b917e83d40fda1010778c9f58157ebac6aabeea5a38ba5b61fe41`.

The committed lock prevents future transitive dependency resolution from silently changing the benchmark environment.

## Decision

**A2 survives and becomes the leading Phase 2 development candidate.** The result is large enough that A3 must now justify its added training complexity against A2 rather than simply outperform A1.

A1 remains a required simpler reference. Nothing in this checkpoint selects the final router.

The README release benchmark remains `pending` because this is validation evidence and the confirmatory test remains unopened.

## Next locked action

Implement **A3 — one frozen compact transformer classifier** under the identical train/validation contract. Freeze the base checkpoint, tokenizer, training budget, seed, early-stopping rule, and optimization specification before the first A3 result. Do not open the confirmatory test and do not introduce another classifier family.

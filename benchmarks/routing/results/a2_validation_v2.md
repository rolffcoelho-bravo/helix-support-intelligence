# Phase 2 A2 Validation Checkpoint — Audited v2

> **Development evidence only.** The confirmatory BANKING77 test split was not downloaded, opened, scored, or used for selection.

## Why this supersedes v1

A post-execution audit found three issues in the first A2 checkpoint presentation:

1. the confusion comparison used only each model's top-20 confusion list, so absence from A2's top 20 was incorrectly interpreted as zero occurrences;
2. the benchmark executed on CPU, but its first lock resolved a CUDA-enabled PyTorch distribution and unnecessary CUDA dependencies;
3. two ML runs on heterogeneous GitHub CPU runners were described as byte-identical, which is stronger than the floating-point evidence supports.

All three were corrected before A2 was allowed to close. **The aggregate classification metrics and the risk-coverage curve did not change.**

## Frozen A2

A2 remains the same pre-specified model:

- encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- encoder revision: `c315f904dfc467d8b9c40ab4ed50b3a8d0866c15`;
- 384-dimensional normalized sentence embeddings;
- no fine-tuning;
- CPU execution, embedding batch size 64;
- logistic regression with `C=1.0`, `lbfgs`, `max_iter=1000`, and `tol=1e-4`;
- no model shopping and no hyperparameter search.

The A2 script now explicitly pins `torch==2.13.0` to the PyTorch CPU index. The committed lock resolves `torch 2.13.0+cpu`, reports `torch_cuda_available=false`, and contains no CUDA, NVIDIA, or Triton packages.

## Result

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE, 15 bins | Multiclass Brier |
|---|---:|---:|---:|---:|---:|
| A1 — TF-IDF + logistic regression | 0.8422 | 0.8407 | 0.9534 | 0.4895 | 0.4951 |
| **A2 — frozen MiniLM embeddings + logistic regression** | **0.8986** | **0.8963** | **0.9732** | **0.2910** | **0.2501** |
| **A2 minus A1** | **+0.0564** | **+0.0556** | **+0.0197** | **−0.1986** | **−0.2450** |

A2 validation accuracy is **0.9023**. Mean maximum probability is approximately **0.6114**, compared with A1's 0.3667. A2 therefore remains under-confident on validation, but materially less so than A1.

No calibration method is selected here. Calibration remains downstream of A3.

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

No automation threshold is selected from this table.

## Corrected confusion analysis

The audited comparison now counts A2 errors across the **full validation prediction set** for each of A1's leading confusion pairs.

Examples:

- `topping_up_by_card → top_up_by_cash_or_cheque`: **4 → 0**;
- `compromised_card → lost_or_stolen_card`: **4 → 1**;
- `top_up_failed → top_up_reverted`: **4 → 0**;
- `pending_cash_withdrawal → declined_cash_withdrawal`: **3 → 1**;
- `card_delivery_estimate → transfer_timing`: **3 → 0**;
- `card_payment_fee_charged → extra_charge_on_statement`: **3 → 1**;
- `declined_transfer → failed_transfer`: **4 → 6**, a regression that remains important for A3.

The previous v1 wording that treated every top-20 disappearance as zero was incorrect and has been superseded.

The principal A2 directional errors include:

- `declined_transfer → failed_transfer`: 6;
- `beneficiary_not_allowed → failed_transfer`: 6;
- `wrong_exchange_rate_for_cash_withdrawal → cash_withdrawal_charge`: 5;
- `balance_not_updated_after_bank_transfer → transfer_not_received_by_recipient`: 5;
- `extra_charge_on_statement → card_payment_fee_charged`: 4;
- `verify_my_identity → why_verify_identity`: 4.

## Reproducibility standard

Two independent post-audit GitHub Actions runs used the CPU-only environment and the same frozen train/validation contract.

The decision-level evidence reproduced exactly:

- all **1,976 sample IDs** and true labels remained aligned;
- all **1,976 predicted intents** were identical;
- all **1,976 top-3 intent sets** were identical;
- accuracy, macro-F1, balanced accuracy, and top-3 recall were identical;
- every registered selective-risk value was identical.

The raw probabilities were **not bitwise identical**, which is expected across heterogeneous CPU runners. The largest confidence difference was only **6.74×10⁻⁷**. ECE differed by **8.78×10⁻⁹**, Brier by **5.51×10⁻¹⁰**, and mean maximum probability by **8.82×10⁻⁹**.

This checkpoint therefore uses the defensible statement **exact decision reproducibility plus bounded numerical reproducibility**, not byte-identical ML output.

The committed CPU lock SHA-256 is:

`e46b877897e03fbf67fe35b90e4907fd812c81c092486ddf16c9dc7f1948cd8d`

## Complexity and timing

A2 adds a neural encoder that A1 does not require. Timing on the two audited GitHub CPU runners varied materially: validation embedding ranged from approximately **2.85 to 5.68 ms per example**.

That runner-to-runner spread is itself useful evidence: these measurements are not stable enough to support a public latency claim. A standardized A1/A2/A3 latency and cost harness remains required before the final router is selected.

## Decision

**A2 survives and remains the leading Phase 2 development candidate.** The audit changed the accuracy of the evidence presentation, not the model ranking.

A1 remains the required simpler reference. A3 remains mandatory under the frozen ladder.

The README release benchmark remains `pending`; the confirmatory test remains unopened.

## Next evaluation step

Implement **A3 — one frozen compact transformer classifier** under the identical train/validation contract. Freeze the base checkpoint, tokenizer, training budget, seed, early-stopping rule, and optimization specification before the first A3 result. The confirmatory test remains separate and no additional classifier family is introduced.

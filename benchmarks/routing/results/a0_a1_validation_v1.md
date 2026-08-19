# Phase 2 A0/A1 Validation Checkpoint

> **Development evidence only.** The confirmatory BANKING77 test split was not downloaded, opened, scored, or used for selection.

## Frozen inputs

- Dataset contract: `banking77-helix-v1`
- Train: 7,904 rows — `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3`
- Validation: 1,976 rows — `5a6e2bef72257bb3aa33aba4ca4a93a13738e0a487be88e7846b986b33713455`
- Quarantine: 123 source-training rows
- Upstream revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Benchmark environment: Python 3.12.3, scikit-learn 1.8.0, NumPy 2.3.5, SciPy 1.17.0
- Thread counts fixed to one for the benchmark job

## Results

| Model | Macro-F1 | Balanced accuracy | Top-3 recall | ECE, 15 bins | Multiclass Brier |
|---|---:|---:|---:|---:|---:|
| A0 — most frequent | 0.0005 | 0.0130 | 0.0491 | 0.9813 | 1.9626 |
| A0 — stratified | 0.0163 | 0.0162 | 0.0471 | 0.9833 | 1.9666 |
| **A1 — TF-IDF + logistic regression** | **0.8422** | **0.8407** | **0.9534** | **0.4895** | **0.4951** |

A1 uses a fixed, untuned specification: lower-cased word unigrams and bigrams, `min_df=2`, sublinear term frequency, L2-normalized TF-IDF, and logistic regression with `C=1.0`, `lbfgs`, tolerance `1e-4`, and at most 1,000 iterations. The fit converged in 33 iterations and produced 8,754 TF-IDF features.

## Finding 1 — the classical baseline is already substantial

A1 reaches validation macro-F1 **0.8422** and balanced accuracy **0.8407** without a hyperparameter search. Its top-three recall is **0.9534**. This is now the first serious routing baseline that A2 and A3 must beat or justify through better calibration, selective risk, out-of-scope behavior, or operational cost.

The result changes the burden of proof for later models. A transformer is not automatically preferable simply because it is more sophisticated.

## Finding 2 — raw probability calibration is poor

A1 validation accuracy is **0.8563**, but its mean maximum predicted probability is only **0.3667**. The 15-bin expected calibration error is **0.4895**.

The dominant problem is therefore under-confidence rather than inflated certainty. This matters because the Helix router is selective: a confidence threshold controls whether a case is automatically routed or escalated. Using the raw A1 probabilities as if they were calibrated probabilities would make the operating threshold difficult to interpret.

This is evidence that Phase 2 calibration is necessary. It is not yet evidence for H3, whose confirmatory comparison remains downstream.

## Finding 3 — confidence ranking is nevertheless useful

Although the raw probability scale is miscalibrated, it orders examples by difficulty surprisingly well.

| Coverage | Accepted | Raw confidence boundary | Selective risk |
|---|---:|---:|---:|
| 10% | 198 | 0.7209 | 0.51% |
| 30% | 593 | 0.4892 | 1.01% |
| 50% | 988 | 0.3263 | 2.53% |
| 70% | 1,383 | 0.1916 | 6.07% |
| 90% | 1,778 | 0.0951 | 10.01% |
| 100% | 1,976 | 0.0307 | 14.37% |

This is an important distinction: **probability calibration and ranking by confidence are not the same property**. A1's numerical confidence values are not trustworthy as probabilities yet, but the ordering already creates a meaningful risk–coverage frontier.

No final automation threshold is selected at this checkpoint.

## Principal confusion pairs

The largest directional A1 confusions were:

1. `balance_not_updated_after_bank_transfer` → `transfer_not_received_by_recipient` — 5 cases
2. `topping_up_by_card` → `top_up_by_cash_or_cheque` — 4
3. `compromised_card` → `lost_or_stolen_card` — 4
4. `declined_transfer` → `failed_transfer` — 4
5. `top_up_failed` → `top_up_reverted` — 4
6. `pending_cash_withdrawal` → `declined_cash_withdrawal` — 3
7. `card_delivery_estimate` → `transfer_timing` — 3
8. `pending_top_up` → `top_up_failed` — 3

These pairs are operationally more informative than aggregate accuracy because several distinguish a process state — pending, failed, declined, reverted, or not received — rather than a broad banking topic. Later models should be judged partly on whether they reduce these semantically adjacent errors without becoming less reliable elsewhere.

## Reproducibility check

The benchmark was executed twice in independent GitHub Actions runs. The second run reproduced the first run byte-for-byte for all three generated evidence files:

- `results.json`: `10f7e922f9e0fa69908d632439dfc8d8383303fef061257a76b0c009f34a2520`
- generated Markdown report: `84c94d55d50edf6a6e2c3c2f919bac55f9102f937fd837ed89bc5b4fb0119959`
- 1,976-row A1 prediction file: `59bd49532cc904ce89cc66f5caabddb3ddd887111b26bc6012721c6249e6a2b4`

The public checkpoint JSON stores the two workflow run identifiers and these evidence hashes.

## What this checkpoint does not establish

It does **not** establish:

- out-of-scope performance;
- calibrated routing probabilities;
- a routing cost advantage;
- a final abstention threshold;
- superiority over A2 or A3;
- H3 or H4 confirmatory conclusions;
- performance on the official test split;
- real-bank or real-customer impact.

The README release benchmark therefore remains `pending`.

## Decision

**A1 survives as the classical Phase 2 baseline.** Its classification quality is high enough that A2 and A3 must earn their additional complexity. Its raw probability calibration is poor enough that calibration and selective routing remain essential parts of the phase.

**Next evaluation step:** implement the bounded **A2 sentence-embedding + linear-classifier candidate** in the same train/validation protocol, while preserving the A1 probability outputs for the later calibration comparison. The confirmatory test split remains separate and no additional classifier family is introduced.

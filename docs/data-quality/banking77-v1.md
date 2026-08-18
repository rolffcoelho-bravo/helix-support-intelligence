# BANKING77 Phase 1 Provenance and Leakage Audit

- Audit date: 2026-08-18
- Upstream revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Upstream train rows: 10,003
- Upstream test rows: 3,080
- Intents: 77
- Status: frozen Phase 1 contract

## Source integrity

Pinned CSV digests:

| Split | SHA-256 |
|---|---|
| train | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| test | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

The materialization script refuses a source whose bytes, row counts, or intent cardinality differ from the contract.

## Leakage audit

The audit compared source-training and official-test text after Unicode NFKC normalization, case-folding, and whitespace collapse. Candidate pairs were restricted to the same intent and a minimum normalized-length ratio of 0.90. `difflib.SequenceMatcher` similarity of at least 0.95 was treated as a quarantine trigger.

The frozen audit identified:

- 125 high-similarity train/test pairs;
- 112 unique official-test rows involved;
- 123 unique source-training rows quarantined.

The official test set itself was not edited.

The quarantine indices are committed in `configs/data/banking77.json`. They are data-quality metadata, not a tunable hyperparameter.

## Derived split

After quarantine, the 9,880-row source-training pool is split deterministically within each intent. Validation membership is chosen by a SHA-256 ordering over a fixed split salt and stable sample identifier.

| Partition | Rows | SHA-256 of canonical generated JSONL |
|---|---:|---|
| train | 7,904 | `bfea6d5e5144b22d2eb67c770ba4891bb69d3f71e64e815ea895bb5dbf6810b3` |
| validation | 1,976 | `5a6e2bef72257bb3aa33aba4ca4a93a13738e0a487be88e7846b986b33713455` |
| test | 3,080 | `4c519f47e6d1c640ccb71d322c3cb9b810642bd42ea4d8395293e0044952c468` |
| quarantine | 123 | excluded |

All train, validation, and test partitions retain all 77 intents. After quarantine, no exact normalized text remains shared between the fit/validation pool and the official test split.

## Interpretation

This procedure reduces a concrete leakage risk without claiming that lexical similarity detection can identify every semantic duplicate. The frozen official test remains the confirmatory benchmark, and no Phase 1 model tuning has occurred.

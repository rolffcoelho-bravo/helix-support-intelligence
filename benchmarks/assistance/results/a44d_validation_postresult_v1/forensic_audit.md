# A4.4d post-result forensic audit

Status: `CLOSED_FAILED_REGISTERED_VALIDATION_NO_RESCUE`

The one-shot A4.4d validation execution at scientific SHA `794562f6d9914bfc36e929c6c9df57e06969665a` completed with workflow-integrity success and preregistered scientific failure. The result is preserved exactly as observed. No rerun, refit, threshold search, model substitution, candidate comparison, confirmatory inspection, or post-result rescue is authorized.

## Integrity

The immutable GitHub Actions artifact is `9456245570`, with ZIP SHA256 `e6e31bd269d3da9b435ec6097d7df492dc4d151ee869b97152895861fad1956c`. External inspection verified the exact nine-file evidence set and every inner SHA256 checksum.

The external reconstruction found exactly 144 unique validation cases and 246 unique validation semantic pairs, all marked `split="validation"`. Gold relation counts reconstruct to 106 ENTAILED, 20 CONTRADICTED, and 120 UNKNOWN. Every stored raw argmax and correctness flag reconstructs directly from the raw logits, and scaling by the frozen `T = 3.67` preserves every argmax.

## Registered result

Eleven of seventeen preregistered A4.4a requirements pass. Six fail:

- atomic relation macro F1: `0.413460347498` against `>= 0.95`
- UNKNOWN recall: `0.008333333333` against `>= 0.95`
- macro case-category accuracy: `0.788888888889` against `>= 0.95`
- SUPPORTED recall: `0.666666666667` against `>= 0.95`
- multi-document SUPPORTED recall: `0.0` against `>= 0.90`
- partial multi-document UNSUPPORTED accuracy: `0.1` against `>= 0.95`

ENTAILED recall is `0.981132075472` and CONTRADICTED recall is `1.0`. SUPPORTED precision is `1.0`. Citation-invalid, stale-current-evidence, unresolved-conflict, literal-supported, paraphrase-supported, contradiction-unsupported, and unsupported-approval checks pass at their registered thresholds.

## Failure mechanism

The dominant registered failure is UNKNOWN discrimination. Of 120 UNKNOWN gold semantic pairs, the frozen verifier predicts 114 as CONTRADICTED, 5 as ENTAILED, and only 1 as UNKNOWN. This error propagates into the compositional multi-document verdicts, where irrelevant evidence is frequently treated as contradiction rather than unknown evidence.

This is a descriptive post-validation diagnosis only. It does not retroactively authorize a different verifier, calibration, threshold, abstention rule, prompt, or claim-verdict logic.

## Boundary

Calibration cases were not materialized or rescored. Candidate rows scored remain zero. Confirmatory records inspected and confirmatory queries scored remain zero. The 68-query confirmatory partition remains unopened.

A4.4d is therefore closed with a failed registered validation result. Any subsequent methodology or development action requires a separate approved checkpoint.

# A4.4d sealed validation-only execution

A4.4d is the one-shot validation checkpoint for the compositional grounding evaluator registered in A4.4a. It opens only the frozen 144-case validation split after A4.4b fixed the semantic verifier and A4.4c fixed the global calibration temperature.

## Frozen scientific surface

The semantic verifier remains `FacebookAI/roberta-large-mnli` at immutable revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`, with safetensors SHA256 `f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5`, CPU FP32 execution, batch size eight, and native raw-logit argmax labels contradiction, neutral, and entailment mapped to CONTRADICTED, UNKNOWN, and ENTAILED.

The A4.4c global temperature is frozen at `T = 3.67`. It is retained as probability-calibration provenance only. A positive common temperature cannot change argmax classes, and A4.4d explicitly verifies that class labels remain identical after scaling. No validation NLL objective, refit, threshold search, or probability-based rescue is permitted.

## Validation scope

A4.4d authorizes exactly 144 validation cases from 20 registered A4.4a validation intents. The registered construction deterministically implies 246 eligible semantic atom-document pairs with gold counts 106 ENTAILED, 20 CONTRADICTED, and 120 UNKNOWN. Calibration cases are not materialized by A4.4d.

The pre-execution CI path does not materialize validation cases or call the semantic verifier. Validation case construction first occurs inside the one-shot main-branch execution after the exact predecessor boundary is verified.

## Registered metrics and gates

A4.4d computes only the validation measures preregistered in A4.4a: atomic-relation macro F1; ENTAILED, CONTRADICTED, and UNKNOWN recall; macro case-category accuracy; SUPPORTED precision and recall; the seven semantic case-category requirements; the deterministic citation-invalid, stale-evidence, and unresolved-conflict accuracy requirements; and the false-SUPPORTED safety count.

Every A4.4a requirement must pass. A scientific failure is preserved as `FAILED_REGISTERED_VALIDATION_NO_RESCUE`; it does not convert the GitHub workflow into a model-selection or tuning loop. Execution-integrity success and scientific validity are deliberately separate.

## Evidence contract

The immutable artifact contains raw validation pair logits, gold relations, reconstructed raw argmax classes, final case verdicts, all registered metrics and gate decisions, runtime versions, frozen-input hashes, an independently reconstructed post-audit, and artifact checksums.

No calibration result is rescored. No G0/G1/G2 candidate is scored. The 68-query confirmatory partition remains unopened. Any later action after A4.4d requires a new explicit checkpoint and approval.

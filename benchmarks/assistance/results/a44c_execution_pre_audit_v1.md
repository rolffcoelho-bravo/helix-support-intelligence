# A4.4c calibration-only pre-execution audit

**Status: PASSED_PRE_EXECUTION_CALIBRATION_ONLY_NO_RESULTS**

A4.4c is limited to the frozen A4.4a calibration split. It authorizes semantic-verifier inference on exactly 288 calibration cases, yielding 491 eligible atom-document semantic pairs under the already frozen citation-identity rules. The deterministic gold relation totals are 211 ENTAILED, 40 CONTRADICTED, and 240 UNKNOWN.

The exact 288-row calibration surface is now frozen independently at SHA256 `a2a68dac77b644ed1f2b114dc0b59f7daba53a452140d97083be83fd95a4cf58` before any semantic inference.

The verifier remains `FacebookAI/roberta-large-mnli` at immutable revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`, using the registered safetensors artifact SHA256 `f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5`. Raw-logit argmax is the frozen semantic class decision.

Calibration may fit exactly one global positive temperature by minimizing three-class negative log likelihood over the deterministic 0.25 through 4.00 grid in 0.01 increments, 376 points, with the smallest temperature selected on a tie. Temperature scaling is diagnostic only and cannot change any raw-logit argmax class or final grounding verdict.

The hostile pre-execution audit caught an important boundary weakness before any model download or inference. The first implementation regenerated all 432 A4.4a cases and filtered to calibration afterward. Although validation would not have been scored, that still materialized validation records and therefore did not meet the strongest meaning of “unopened.” The runtime was repaired to use a dedicated calibration-only materializer. A4.4c now constructs only the 40 registered calibration intents and their 288 cases; the 144 validation cases are not materialized, scored, or written at all.

Two non-scientific CI issues were also caught and repaired before execution: canonical Ruff formatting in the independent verifier and the established mypy annotation for a dynamic benchmark import. Neither repair changed the model, labels, case content, gold relations, calibration grid, temperature objective, or any sealed-data boundary.

The complete pre-execution quality gate then passed in CI run `32500155015`, job `96827728026`: Ruff passed, strict mypy passed, **156 tests passed**, all predecessor preflights passed, the A4.4c preflight reconstructed the exact calibration hash and 491-pair arithmetic with `validation_cases_materialized=0`, and publication audit passed.

The execution must persist raw calibration logits and the full temperature grid, then independently reconstruct the selected temperature and arithmetic from those raw logits. No G0/G1/G2 candidate is evaluated, no model-family comparison is permitted, and the 68-query confirmatory partition remains unopened.

Before this execution, semantic-verifier calls are 0, calibration logits opened are 0, validation cases materialized are 0, validation results opened are 0, candidate calls are 0, and confirmatory queries scored are 0.

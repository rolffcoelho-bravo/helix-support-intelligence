# A4.4c calibration-only pre-execution audit

**Status: PASSED_PRE_EXECUTION_CALIBRATION_ONLY_NO_RESULTS**

A4.4c is limited to the frozen A4.4a calibration split. It authorizes semantic-verifier inference on 288 calibration cases only, yielding 491 eligible atom-document semantic pairs under the already frozen citation-identity rules. The deterministic gold relation totals are 211 ENTAILED, 40 CONTRADICTED, and 240 UNKNOWN.

The verifier remains `FacebookAI/roberta-large-mnli` at immutable revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`, using the registered safetensors artifact SHA256 `f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5`. Raw-logit argmax is the frozen semantic class decision.

Calibration may fit exactly one global positive temperature by minimizing three-class negative log likelihood over the deterministic 0.25 through 4.00 grid in 0.01 increments, 376 points, with the smallest temperature selected on a tie. Temperature scaling is diagnostic only and cannot change any raw-logit argmax class or final grounding verdict.

The execution must persist raw calibration logits and the full temperature grid, then independently reconstruct the selected temperature and arithmetic from those raw logits. The 144 A4.4a validation cases remain unscored, no G0/G1/G2 candidate is evaluated, no model-family comparison is permitted, and the 68-query confirmatory partition remains unopened.

Before this execution, semantic-verifier calls are 0, calibration logits opened are 0, validation results opened are 0, candidate calls are 0, and confirmatory queries scored are 0.

# A4.3a grounding-evaluator forensic audit

**Status: CLOSED_FAILED_EVALUATOR_VALIDITY**

The immutable A4.3a execution completed successfully, but the frozen MiniLM2 grounding evaluator failed the registered untouched validation requirements. This closure audit reran neither the benchmark nor NLI inference. It independently reconstructs the threshold choice and validation arithmetic from the immutable execution artifact.

## Provenance

Scientific SHA: `15ad4e92f0d73caf944ebdb7312d6366f933bd33`.
GitHub run: `32402729510`; job: `96534504574`; artifact: `9419102659`.
Artifact ZIP SHA256: `56909955e28f05e44b6c86d88a79739678e58afd741daed02fd3844fe7804c8e`.
Anchor-suite SHA256: `145c3cbda7deea0b4befe8d471ecaa6689af524ad528fd851ef20eabe35688ac`.

Execution guards remained closed: 0 candidate calls, 0 candidate scores, 0 OpenAI calls, 0 confirmatory-query scores, and no post-validation threshold adjustment.

## Independent reconstruction

The frozen calibration grid has 91 thresholds from 0.05 through 0.95. Reconstruction selects **0.05**, exactly matching the registered result. Calibration positive sensitivity is **0.8583**, negative specificity **0.9219**, and balanced accuracy **0.8901**.

On the untouched 124-anchor validation split, positive sensitivity is **0.9167** (55/60), negative specificity **0.9688** (62/64), and balanced accuracy **0.9427**. The frozen 0.95 requirements for overall positive sensitivity and balanced accuracy fail. The conflict false-consensus category also fails at **0/2** correct negatives.

Seven validation anchors are misclassified: five false negatives and two false positives. Exact anchor IDs and probabilities are preserved in `diagnostic_slices.json`.

## Disposition

The registered `FAILED_EVALUATOR_VALIDITY` status is independently confirmed. `cross-encoder/nli-MiniLM2-L6-H768` is rejected as the independent grounding evaluator for future assistance-candidate selection under this protocol. A4.3b is not authorized. No threshold rescue, A4.3a rescue rerun, replacement-model search, or confirmatory evaluation is permitted under this closed checkpoint. Any replacement evaluator requires a separately approved and versioned methodology gate.

# A4.3a pre-inference scientific audit

**Status: PASSED_PRE_INFERENCE_NO_RESULTS**

A4.3a has been audited before any local NLI inference, assistance-candidate call, OpenAI call, or confirmatory-query scoring.

The frozen evidence contract reclassifies ten already-opened development cases from `ANSWER_WITH_EVIDENCE` to `ESCALATE_CONFLICTING_EVIDENCE`, leaving zero answer decisions whose current direct evidence contains an unresolved conflict fixture. The 68-query confirmatory partition remains unscored.

The candidate-independent grounding suite contains 372 rows: 248 calibration anchors and 124 untouched validation anchors. Its final SHA256 is `145c3cbda7deea0b4befe8d471ecaa6689af524ad528fd851ef20eabe35688ac`. It uses only A4.0 development intents, no candidate output, no query text, and no confirmatory intent.

The A4.3a evaluator dependency lock is identical to the frozen A4.1 runtime lock at Git blob SHA `f2f4a13c6f311257873ae7132a08f67dcfc3de20` after aligning the PEP 723 dependency declaration.

The manual audit caught one construct-validity defect before inference. The first conflict-union negative repeated a proposition explicitly asserted by the controlled conflict FAQ, so a sentence-level NLI model could reasonably consider that proposition entailed even though the evidence set remained conflicted. Before any NLI result was observed, the hypothesis was replaced by an unambiguously false consensus claim: both documents agree that review is optional when unresolved uncertainty remains. No model or candidate result informed that repair.

The threshold remains selected only on the 248 calibration anchors from the frozen 0.05 to 0.95 grid at 0.01 increments. The 124 validation anchors cannot affect threshold selection. Failure of any registered validation requirement rejects the MiniLM2 evaluator for future candidate selection; A4.3a does not permit searching a replacement model family after seeing the result.

No HelixBank corpus content, Phase 2 or Phase 3 scientific input, A4.0 protocol, A4.1 binding, A4.2 raw result, or assistance prompt is modified by A4.3a. Temporary repair workflows are absent from the final protocol diff.

A passing A4.3a result would validate measurement behavior only on the deterministic fictional HelixBank anchor suite. It would not establish general real-world NLI validity.

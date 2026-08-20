# Phase 4 A4.3a evidence and grounding validity

A4.3a is a measurement-validity checkpoint. It does not score G0, G1, G2, or any replacement assistance candidate. It does not call OpenAI and it does not open the 68-query confirmatory partition.

The checkpoint exists because A4.2 completed its registered 240-query development execution but failed the mandatory post-result scientific audit. The mechanically reconstructed candidate metrics remain preserved as diagnostic evidence only. No A4.2 winner was accepted.

## Evidence contract v2

The HelixBank corpus remains unchanged at `helixbank-policy-v1.0.0`. A4.3a changes the assistance evaluation contract, not the corpus or its relevance judgments.

Eligible relevance-2-or-higher documents remain in each oracle evidence pack. Current conflict evidence is never hidden simply to preserve an answerable label. Instead, a query that would otherwise map to `ANSWER_WITH_EVIDENCE` is mapped to `ESCALATE_CONFLICTING_EVIDENCE` whenever its current eligible direct evidence contains an unresolved conflict fixture. Explicit conflict cases remain conflict escalations, ambiguous requests remain clarification cases, and grade-1-only missing-evidence cases remain low-confidence escalations.

This precedence rule is global. It is not conditioned on which A4.2 candidate succeeded or failed. On the already-opened development partition, the frozen preflight expects ten answer-labelled cases to be reclassified by this rule and zero resulting `ANSWER_WITH_EVIDENCE` cases to retain current unresolved conflict evidence.

## Candidate-independent grounding anchors

The grounding evaluator is validated without using assistance candidate outputs or query text. The suite is built only from frozen HelixBank documents belonging to the 60 A4.0 development intents.

The 60 development intents are split again for evaluator calibration and evaluator validation. The split is deterministic by intent and stratified so both sides contain ordinary, archived-FAQ, and current-conflict fixtures. Forty intents are calibration-only and twenty are validation-only.

The frozen suite contains 372 anchors: exact source entailments, deterministic paraphrase entailments, direct contradictions, unsupported approval claims, citation mismatches, multi-document conjunctions, stale-as-current negatives, and unresolved-conflict union negatives. The calibration surface contains 248 anchors and the untouched evaluator-validation surface contains 124.

## Threshold selection and validity requirements

The A4.1 independent evaluator identity remains fixed for this checkpoint: `cross-encoder/nli-MiniLM2-L6-H768` at revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`.

A4.3a does not preserve the A4.1 threshold by assumption because A4.2 demonstrated that the registered 0.80 cutoff lacked construct validity. Threshold calibration is therefore predeclared before A4.3a inference. The evaluator is scored on the calibration anchors across the fixed 0.05 through 0.95 grid in 0.01 increments. The chosen threshold maximizes balanced accuracy, then the weaker of sensitivity and specificity, then the higher threshold. Validation anchors cannot affect threshold selection.

The calibrated evaluator passes only if the untouched validation surface meets every frozen minimum in `configs/models/assistance_validity_a43a_v1.json`, including at least 0.95 overall positive sensitivity, 0.95 overall negative specificity, 0.95 balanced accuracy, and the registered category-specific minima.

If any validation requirement fails, the MiniLM2 evaluator is rejected for future candidate selection. A4.3a does not permit searching a replacement model family after seeing that failure. A replacement-family study would require a separately versioned and approved gate.

If all requirements pass, the calibrated threshold becomes the only eligible sentence-grounding threshold for a separately versioned A4.3b development experiment. A4.3a itself does not authorize A4.3b execution.

## Guards

A4.3a permits local NLI inference only on the frozen grounding-anchor suite. Candidate calls, candidate scoring, OpenAI calls, prompt changes, corpus mutation, A4.2 reinterpretation, and confirmatory-query scoring remain prohibited.

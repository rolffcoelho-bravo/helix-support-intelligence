# A4.4b pre-execution binding audit

**Status: PASSED_PRE_EXECUTION_BINDING_NO_RESULTS**

A4.4b binds one semantic-verifier identity and one future probability-calibration procedure. It performs no semantic inference, opens no A4.4a calibration or validation cases, scores no assistance candidates, and leaves the confirmatory partition unopened.

## Bound verifier

The frozen verifier is `FacebookAI/roberta-large-mnli` at immutable revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`. Validity execution is bound to `model.safetensors`, SHA256 `f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5`, with a documented remote size of 1.43 GB, a 512-token maximum, MIT licensing, remote code disabled, float32 CPU execution, and no fine-tuning, quantization, or replacement artifact for the validity experiment.

Native model labels are frozen as `0=CONTRADICTION`, `1=NEUTRAL`, and `2=ENTAILMENT`, mapped respectively to A4.4a relations `CONTRADICTED`, `UNKNOWN`, and `ENTAILED`. Class decisions use raw-logit argmax with no class-specific, margin, or abstention thresholds.

## Nomination interpretation

The bound model is one pre-registered verifier nomination, not an empirical claim that it is optimal among all eligible NLI models. No empirical replacement-model comparison was performed, no A4.2 candidate outcome or A4.3a validation error was used to rank models, and neither the A4.4a calibration nor validation cases were used for the nomination. If the verifier later fails the registered validity requirements, the same binding does not permit automatic substitution of another model.

The model card's reported MNLI score of 90.2 is external model documentation only. It is not HelixBank validity evidence.

## Calibration algebra

Future calibration is restricted to one global positive temperature used only for probability-calibration diagnostics. The registered deterministic grid is 0.25 through 4.00 inclusive in increments of 0.01, giving 376 values. The objective is three-class negative log likelihood with the smallest temperature selected on a tie.

Because every permitted temperature is strictly positive, dividing all three logits by the same temperature preserves their ordering. The calibration therefore cannot change the raw-logit argmax class or any final grounding verdict. Class-specific threshold search, validation-driven parameter selection, post-validation refitting, fine-tuning, and post-result model substitution are excluded.

## Independence and provenance

The selected RoBERTa family differs from the frozen DeBERTa-v3 G2 runtime-verifier family and from the rejected MiniLM2-RoBERTa A4.3a evaluator family. The immutable Hugging Face configuration, tokenizer configuration, safetensors pointer, license, and model card were checked against the registered model identity and artifact hash.

The 1.43 GB FP32 artifact may make future CPU calibration operationally slow. That is an execution constraint, not evidence for or against semantic validity, and it does not permit replacement with a smaller or quantized model after results.

## Pre-execution repair

The scientific review identified one wording ambiguity before any model download or inference: the original selection wording could be read as implying empirical optimality even though no empirical model comparison had occurred. The binding was tightened to define the model explicitly as a single pre-registered nomination with no optimality claim and no automatic replacement path.

## Verified guards

Candidate calls: 0. OpenAI calls: 0. Semantic-verifier downloads: 0. Semantic-verifier calls: 0. Replacement-family empirical searches: 0. A4.4a calibration records inspected: 0. A4.4a validation records inspected: 0. Confirmatory records inspected: 0. Confirmatory queries scored: 0.

The final pre-audit registration head `727f72ae0c9b7e3e5b88b8e1183d77e8ca0ab930` passed CI run `32433863469`, job `96630960694`, with 150 tests and a passing publication audit. The binding is ready to freeze after the audit-inclusive head passes the same repository checks.

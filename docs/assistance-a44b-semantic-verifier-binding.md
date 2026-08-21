# Phase 4 A4.4b: Semantic verifier binding

A4.4b binds one semantic-verifier identity and one future probability-calibration procedure for the compositional grounding construct frozen in A4.4a. This checkpoint performs no model download, no semantic inference, no candidate scoring, no A4.4a calibration-case inspection, no A4.4a validation-case inspection, and no confirmatory-query access.

## Bound verifier

The semantic verifier is `FacebookAI/roberta-large-mnli` at immutable Hugging Face revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`. The validity execution is bound to `model.safetensors`, SHA256 `f4dbab1bceb16f9800f7b9a9c96b187d5400511b66982e4e845de920f69b89b5`, with remote code disabled.

The selected architecture family is RoBERTa. This differs from the frozen G2 runtime verifier family, DeBERTa-v3, and from the rejected A4.3a MiniLM2-RoBERTa evaluator family. The model is documented under the MIT license and provides native three-way NLI labels.

The native labels are mapped exactly as follows:

| Native model label | A4.4a atomic relation |
| --- | --- |
| `CONTRADICTION` | `CONTRADICTED` |
| `NEUTRAL` | `UNKNOWN` |
| `ENTAILMENT` | `ENTAILED` |

No binary entailment threshold, class-specific threshold, margin threshold, or abstention threshold is part of A4.4b. The semantic class is the argmax of the three raw logits.

## Pair interface

Each future semantic-verifier input is one valid cited-presented atom-document pair from the A4.4a construct. The document body is the premise and the atomic proposition text is the hypothesis. Tokenization is frozen to a maximum length of 512 with truncation enabled and padding to the longest item in each batch. Validity execution uses CPU, float32, evaluation mode, batch size 8, safetensors weights, and no quantization or fine-tuning.

## Selection discipline

The verifier was nominated from public model metadata and structural requirements only. This is a single pre-registered verifier nomination, not a claim that RoBERTa-large-MNLI is empirically optimal among all eligible models. No A4.2 candidate result, A4.3a validation error, A4.4a calibration case, or A4.4a validation case was used to rank replacement models, and no empirical replacement-model bakeoff was performed.

The structural requirements were native three-way NLI output, a permissive public license, a safetensors artifact, an immutable revision, at least a 512-token input limit, disabled remote code, and an architecture family distinct from both the frozen G2 runtime verifier and the rejected A4.3a evaluator. If this bound verifier later fails the registered validity requirements, it is rejected under this binding. Another model cannot be substituted automatically; a replacement requires a new versioned binding.

The model card reports that RoBERTa-large-MNLI was fine-tuned on MultiNLI and reports an external MNLI score of 90.2. That number is model documentation only. It is not HelixBank validity evidence and cannot substitute for the untouched A4.4a validation experiment.

## Future calibration rule

A future separately versioned execution may use only the 288 registered A4.4a calibration cases. Model weights and class decisions remain frozen. The only fitted parameter is one global temperature used for probability-calibration diagnostics.

Temperature fitting is deterministic: evaluate values from 0.25 through 4.00 inclusive in steps of 0.01, minimize three-class negative log likelihood, and choose the smallest temperature on a tie. Temperature scaling cannot change the raw-logit argmax class and cannot change any final grounding verdict.

Class-specific threshold search, validation-driven parameter selection, model fine-tuning, prompt tuning, family switching after calibration, post-validation refitting, and post-result rescue are not permitted by this binding.

## Isolation boundary

A4.4b does not authorize semantic inference. It does not open the 288 calibration cases or the 144 validation cases. It does not compare G0, G1, or G2 and does not inspect or score the 68-query confirmatory partition.

The next execution checkpoint must be separately versioned and must remain calibration-only. The A4.4a validation split stays unopened until the semantic-verifier identity and calibration procedure are already frozen.

## External model evidence snapshot

The public evidence snapshot was taken on 2026-08-21 from the Hugging Face model card, immutable revision, configuration, tokenizer configuration, and safetensors artifact for `FacebookAI/roberta-large-mnli`. Those external sources establish model identity, label mapping, artifact type, license, tokenizer limit, and published model metadata. They do not establish performance on HelixBank.

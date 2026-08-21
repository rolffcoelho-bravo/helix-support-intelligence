# A4.4c calibration-only execution

A4.4c consumes only the already registered A4.4a calibration split. Its sole scientific purpose is to execute the frozen three-way RoBERTa verifier on eligible calibration atom-document pairs and bind one global temperature for probability-calibration diagnostics before any validation exposure.

## Frozen execution surface

The 288 calibration cases deterministically yield 491 semantic pairs after citation-invalid cases bypass semantic scoring. The gold relation totals are 211 ENTAILED, 40 CONTRADICTED, and 240 UNKNOWN. Premises use `document.body`, hypotheses use `atom.text`, and the semantic class is always the raw-logit argmax over native contradiction, neutral, and entailment logits.

The verifier identity, revision, safetensors SHA256, CPU FP32 execution, batch size eight, tokenizer limit, and label mapping are inherited unchanged from A4.4b. A4.4c cannot substitute another model, fine-tune the model, alter prompts, introduce thresholds, or use candidate outcomes.

## Temperature binding

The only fitted parameter is a single positive scalar temperature. The search grid is 0.25 through 4.00 inclusive in increments of 0.01, giving 376 points. The objective is mean three-class negative log likelihood across the 491 calibration semantic pairs. The smallest temperature is selected on an exact tie.

A positive common temperature preserves logit ordering. Therefore the fitted temperature is not allowed to change raw semantic classes, claim verdicts, or any downstream decision. The execution explicitly checks that every calibrated argmax equals the corresponding raw-logit argmax.

## Evidence and reconstruction

The immutable run artifact contains calibration pair identifiers, gold relations, three raw logits, raw argmax classes, the complete temperature grid, the selected temperature, runtime package versions, model artifact provenance, and zero-access counters for sealed surfaces. It does not contain validation rows or confirmatory rows.

A second script reconstructs every raw argmax, every grid NLL, the selected temperature, raw and calibrated NLL, and calibration accuracy directly from the raw-logit artifact. A4.4c cannot close unless that independent reconstruction passes.

## Sealed surfaces

The 144-case A4.4a validation split remains unscored throughout A4.4c. No validation metric may be computed, no validation-driven parameter may be selected, no G0/G1/G2 candidate may be scored, and the 68-query confirmatory partition remains unopened. Any validation execution is a separate checkpoint requiring separate approval.

# A4.4c calibration-only execution

A4.4c consumes only the already registered A4.4a calibration split. Its sole scientific purpose is to execute the frozen three-way RoBERTa verifier on eligible calibration atom-document pairs and bind one global temperature for probability-calibration diagnostics before any validation exposure.

## Closed result

A4.4c is formally closed. The one-shot execution ran as GitHub Actions run `32500393608`, attempt 1, on exact scientific head SHA `cf35b4e6ea495b776dd92c1420c7f7696e6a14db`. The frozen global temperature is **3.67**. Raw three-class NLL is `1.5764484640192402`, calibrated NLL is `0.8698291056737014`, and positive temperature scaling preserved every raw-logit argmax class.

The permanent post-result record lives under `benchmarks/assistance/results/a44c_calibration_postresult_v1/`. It freezes the scientific disposition, evidence provenance, artifact hashes, external reconstruction, and descriptive calibration diagnostics. The temperature may not be refit after validation.

## Frozen execution surface

The 288 calibration cases deterministically yield 491 semantic pairs after citation-invalid cases bypass semantic scoring. The exact calibration surface SHA256 is `a2a68dac77b644ed1f2b114dc0b59f7daba53a452140d97083be83fd95a4cf58`. The gold relation totals are 211 ENTAILED, 40 CONTRADICTED, and 240 UNKNOWN. Premises use `document.body`, hypotheses use `atom.text`, and the semantic class is always the raw-logit argmax over native contradiction, neutral, and entailment logits.

The verifier identity, revision, safetensors SHA256, CPU FP32 execution, batch size eight, tokenizer limit, and label mapping are inherited unchanged from A4.4b. A4.4c cannot substitute another model, fine-tune the model, alter prompts, introduce thresholds, or use candidate outcomes.

## Temperature binding

The only fitted parameter is a single positive scalar temperature. The search grid is 0.25 through 4.00 inclusive in increments of 0.01, giving 376 points. The objective is mean three-class negative log likelihood across the 491 calibration semantic pairs. The smallest temperature is selected on an exact tie.

A positive common temperature preserves logit ordering. Therefore the fitted temperature is not allowed to change raw semantic classes, claim verdicts, or any downstream decision. The execution and independent reconstruction both confirmed argmax preservation at `T=3.67`.

## Evidence and reconstruction

The immutable run artifact is GitHub artifact `9453435038`, with ZIP SHA256 `cc80620b3134169110ebbd3c11b567adb715ed955bd48cf391242a7c338246b4`. It contains calibration pair identifiers, gold relations, three raw logits, raw argmax classes, the complete temperature grid, runtime package versions, model artifact provenance, and zero-access counters for sealed surfaces.

Independent post-result inspection recomputed the ZIP digest, all inner evidence hashes, all 491 raw argmax classes and correctness flags, all 376 grid NLL values, and the selected temperature. The grid reconstruction differs from the stored grid by at most approximately `1.11e-16`, consistent with floating-point arithmetic.

A descriptive calibration warning is retained without changing the protocol. ENTAILED recall is `207/211 = 0.9810`, CONTRADICTED recall is `40/40 = 1.0000`, and UNKNOWN recall is only `6/240 = 0.0250`. This diagnostic cannot authorize model rescue, threshold search, model-family comparison, or refitting.

## Sealed surfaces

The 144-case A4.4a validation split remains unopened: validation cases materialized `0`, validation cases scored `0`, validation semantic pairs scored `0`, and validation metrics computed `0`. No G0/G1/G2 candidate was scored. The 68-query confirmatory partition also remains unopened, with confirmatory records inspected `0` and confirmatory queries scored `0`.

Any validation execution is a separate checkpoint requiring separate approval. The frozen temperature remains `3.67` regardless of future validation performance.

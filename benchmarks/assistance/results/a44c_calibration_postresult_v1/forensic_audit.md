# A4.4c post-result forensic audit

**Status: CLOSED_CALIBRATION_TEMPERATURE_FROZEN**

A4.4c is closed. The one-shot scientific execution was GitHub Actions run `32500393608`, attempt 1, on exact head SHA `cf35b4e6ea495b776dd92c1420c7f7696e6a14db`. The workflow completed successfully through model execution, registered-input re-verification, independent arithmetic reconstruction, checksum freeze, summary publication, and immutable evidence upload.

The evidence ZIP is artifact `9453435038`. Its independently recomputed SHA256 is `cc80620b3134169110ebbd3c11b567adb715ed955bd48cf391242a7c338246b4`, exactly matching GitHub's artifact digest. The ZIP contains exactly nine expected files. Every evidence-file SHA256 listed by the workflow manifest was independently recomputed and matched.

The frozen calibration surface contains 288 cases at SHA256 `a2a68dac77b644ed1f2b114dc0b59f7daba53a452140d97083be83fd95a4cf58`. Exactly 491 eligible semantic pairs were scored. The independently reconstructed global temperature is **3.67**. Raw NLL is `1.5764484640192402`, calibrated NLL is `0.8698291056737014`, and the NLL improvement is `0.7066193583455388`. Positive temperature scaling preserves every raw-logit argmax as registered.

The external reconstruction checked every raw row, not only the summary. All 491 pair identifiers are unique, every raw row is marked `calibration`, stored argmax classes reconstruct exactly from the three logits, stored correctness flags reconstruct exactly, and all 376 NLL grid values reproduce with a maximum absolute floating-point difference of approximately `1.11e-16`. The exact minimum is again `T=3.67`.

A descriptive calibration warning is now recorded. Gold ENTAILED recall is `207/211 = 0.9810`; gold CONTRADICTED recall is `40/40 = 1.0000`; gold UNKNOWN recall is only `6/240 = 0.0250`. Of the 240 gold UNKNOWN pairs, 202 are predicted CONTRADICTED and 32 ENTAILED. This explains the raw calibration accuracy of `0.515274949083503`. This diagnostic is not a tuning gate and cannot alter the frozen verifier, temperature, thresholds, or validation protocol.

The sealed boundaries survived. Validation cases materialized: **0**. Validation cases scored: **0**. Validation semantic pairs scored: **0**. Validation metrics computed: **0**. G0/G1/G2 candidates scored: **0**. Confirmatory query records inspected: **0**. Confirmatory queries scored: **0**. Model-family comparisons and post-result substitutions: **0**.

The resolved environment lock matches the runtime report exactly for the scientific dependencies: `huggingface-hub 0.36.2`, `numpy 2.5.2`, `safetensors 0.8.0`, `torch 2.13.0`, and `transformers 4.57.6`.

One minor operational issue is documented without altering A4.4c: the checksum manifest stores original workflow-relative paths while the uploaded ZIP is flat. Direct `sha256sum -c` on an extracted flat archive therefore needs path rehydration. Independent basename-matched SHA256 verification succeeded for every manifest-listed file. This is a portability issue only, not an integrity failure. The frozen A4.4c workflow must not be edited merely to repair this cosmetic artifact behavior because changing its path-scoped workflow file would create another execution event.

## Scientific disposition

A4.4c is formally closed. The single global temperature is frozen at **3.67** and may not be refit after validation. The semantic model may not be substituted, fine-tuned, rescued, or threshold-tuned on the basis of these calibration results. The 144-case A4.4a validation split remains unopened and requires a separate approval before execution. The 68-query confirmatory partition also remains unopened.

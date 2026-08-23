# Phase 4 A4.5b-M3 post-result closure

## Closure status

A4.5b-M3 is permanently closed with scientific status `FAILED_SCEC_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED`.

The registered SCEC calibration-only execution completed successfully at the workflow level, but the scientific readiness gate failed. All preregistered post-execution integrity stages also completed successfully, including frozen-input reverification, deterministic reconstruction of all 609 registered parameter candidates, checksum freezing, and immutable artifact upload.

This is therefore a registered scientific negative result, not an infrastructure failure and not an incomplete execution.

## Authoritative provenance

- scientific workflow run: `32596053185`;
- scientific job: `97087058066`;
- execution SHA: `9521c6dda9ea16eb3cccc91f1a7178064999c32b`;
- direct parent SHA: `806efae4fcbc1f24cc2db8cd8616b5c49d6a1f2a`;
- workflow conclusion: `success`;
- immutable artifact ID: `9483500754`;
- artifact ZIP SHA256: `605208633b9a550f1337f4fbf795027f5c25ee6ae6daeceb9fb19ad30b937824`;
- raw pair-score SHA256: `de996c5a5de1e3cf83ac23600b802a0bc1b27dc16be8935b20ee146de2ff2f09`;
- raw set-score SHA256: `0fb42a1b6ad2b6f5c74cefe3d71f152e362685eff71ad8dff026c378cae9b923`;
- results SHA256: `d58566c281938178951d4ad47c9d447309fa18e18b6fb07e3309e48fac115473`;
- post-audit SHA256: `ec6ca2608f226a894f1b579ec973a9ea397c319d672f45656afdfcbb76303157`;
- deterministic reconstruction: `PASSED_A45BM3_DETERMINISTIC_RECONSTRUCTION`.

The prior failed run `32594745676` remains classified as infrastructure-only and pre-inference. Its recovery changed only the isolated runtime by layering `protobuf==6.33.6`; no scientific configuration changed before the successful execution.

## Frozen scientific result

The calibration corpus contained 48 units, 768 pair rows, 384 evidence-set rows, and 384 claim rows.

The complete preregistered threshold grid contained 609 candidates. **Zero candidates satisfied all 42 readiness requirements.** The best-any candidate under the preregistered selection order used mismatch threshold `0.56` and coverage threshold `0.72`, but it is not a validated or deployable parameter setting because calibration readiness failed.

The selected candidate passed 14 of 42 readiness requirements and failed 28.

| Metric | Frozen result |
|---|---:|
| Compatibility macro F1 | 0.600250 |
| Compatible recall | 0.878472 |
| Incompatible recall | 0.435417 |
| Minimal compatible-span precision | 0.421756 |
| Minimal compatible-span recall | 0.767361 |
| Sufficiency macro F1 | 0.332289 |
| Sufficient recall | 0.703125 |
| Insufficient recall | 0.368056 |
| Polarity macro F1 | 0.478261 |
| SUPPORTS recall | 0.916667 |
| REFUTES recall | 0.000000 |
| Final relation macro F1 | 0.315255 |
| ENTAILED recall | 0.916667 |
| CONTRADICTED recall | 0.000000 |
| UNKNOWN recall | 0.626302 |
| Claim-category macro accuracy | 0.854167 |
| SUPPORTED precision | 0.666667 |
| SUPPORTED recall | 0.916667 |
| Conflict detection accuracy | 0.000000 |
| Support/refute conflict accuracy | 0.000000 |
| False-SUPPORTED safety rate | 0.152778 |

Several scope dimensions also fail structurally rather than marginally. Entity, predicate, target-slot, conditional-scope, and modality/quantification mismatch rejection are all `0.0`; organizational mismatch rejection is `0.770833`; temporal mismatch rejection is `0.854167`. Unresolved scope-gap insufficiency recall is only `0.052083`.

The negative result therefore cannot be characterized as a threshold-edge miss. The frozen learned primitive and registered hypothesis factorization fail to separate multiple SCEC dimensions reliably enough for the downstream sufficiency, polarity, relation, conflict, and safety gates.

## Integrity and sealed boundaries

The post-result audit verified probability-simplex validity, reconstructed all 609 candidates independently from immutable raw scores, reproduced the same selected thresholds and scientific failure, and verified all registered checksums.

The following remain zero and sealed:

- A4.5a fresh-validation rows materialized: 0;
- A4.5a fresh-validation rows scored: 0;
- closed A4.5b rows scored: 0;
- confirmatory records inspected: 0;
- confirmatory queries scored: 0.

Fresh validation is not authorized. The original 68-query confirmatory partition remains sealed. No next checkpoint is authorized by this result.

## Governance consequence

No post-result threshold rescue, prompt search, hypothesis rewrite, model substitution, class-specific thresholding, temperature scaling, validation-assisted tuning, or parallel rescue execution is admissible inside A4.5b-M3.

The diagnostic thresholds `0.56` and `0.72` are preserved only as the best-any calibration point selected by the preregistered ordering. They must not be promoted as approved SCEC parameters.

The machine-readable permanent closure record is `benchmarks/assistance/a45bm3_closure_v1.json`.

## Required next action class

Any further scientific work must begin as a **separately registered post-M3 methodology decision checkpoint**. That checkpoint may analyze this closed failure and external literature, identify which learned/non-learned decomposition assumptions failed, and propose a new predeclared methodology. It must not open A4.5a validation or the confirmatory partition, and it must not perform new model inference, threshold fitting, prompt shopping, or candidate comparison until a new protocol is explicitly approved.

# Phase 4 A4.5b-M6: TPAG Post-Result Closure

## Status

A4.5b-M6 is closed as a registered **positive calibration-readiness result**.

Scientific status:

`PASSED_TPAG_CALIBRATION_READINESS_PARAMETERS_FROZEN`

This result means that the bound TPAG implementation satisfied the complete preregistered M5 calibration-readiness gate and that the selected calibration parameter is frozen. It is **not** fresh-validation evidence, confirmatory evidence, independent generalization evidence, or a real-world performance claim.

Fresh A4.5a validation and the 68-query confirmatory partition remain sealed. No subsequent checkpoint is authorized by this closure.

## Authoritative execution provenance

The registered calibration-only execution was performed by GitHub Actions with:

- workflow run: `32650171688`;
- job: `97220391248`;
- event: `push`;
- run attempt: `1`;
- exact execution SHA: `6a2d77a2ff0a219197dd3acdd58aaaac82f6794d`;
- direct parent SHA: `1ac85b123a4804e5cc6f9b74662ff04fd9ef198b`;
- workflow conclusion: `success`.

The direct parent is the corrected M5 main after the pre-inference A18 fixture erratum. The workflow enforced both this exact parent and the exact nine-file M6 scientific diff before execution.

Corrected M5 alignment SHA-256:

`136322c6b4710fdf3d5ea3edb857a31d8a99133c9308899e176fb914ef6b6c09`

The M5 erratum was closed before M6 inference under:

`CLOSED_PREINFERENCE_FIXTURE_ERRATUM_NO_SEMANTIC_INFERENCE`

No M6 model call or threshold evaluation occurred on the defective M5 construction.

## Immutable evidence artifact

Authoritative artifact:

- artifact ID: `9495963075`;
- artifact name: `phase4-assistance-a45bm6-6a2d77a2ff0a219197dd3acdd58aaaac82f6794d-attempt-1`;
- ZIP SHA-256: `ab56b67f3f2542a7b81342c104d3c7976dc59902e252414aec258437e2a5dbb1`;
- ZIP size: 44,835 bytes.

The downloaded ZIP digest was independently re-hashed and matched the GitHub artifact digest exactly. Every file listed in `checksums.sha256` was also independently re-hashed and matched.

Frozen artifact file SHA-256 values:

- `environment.json`: `d0011096b5bd41f996a26fa4d1d4d93b7c0325f03643900013e3c84cdca1ebc5`
- `model_weight_verification.json`: `5963f782acb89b0297ebd491b9631e594d7ad42bfbd5d5a7606992fabd229860`
- `post_audit.json`: `f6a25a91e07b5f0fd1867e0f821f53db781ef9cd660af565790c5ff3445cdded`
- `post_audit.md`: `d33e27d6cb658f517f3883fa4db5c423058adbbb1e33d084432715eec3dfda6f`
- `raw_inference_manifest.json`: `eda935baae28ec05655fb4f0df1668bf57eeefa82384c6c2a777079603b05c7d`
- `registered_inputs.sha256`: `9b866aaf94f5244f8a219298fcafe61fed0b6b4ce6ca946cc9078223c7a869ea`
- `report.md`: `d4d59c4b30883934ab14eda4e40d6c244dc3e8799664bd6b78d2c077f1f9a9fb`
- `residual_raw_scores.jsonl`: `94a0fd23901fd250513e3415a131ab21ec1a1acbc605b53457fd4e18820ded00`
- `resolved_environment.lock`: `7b7636708fb540d8b3b1a83405f45ffc49f45de59bc614cb86a969b10a44b951`
- `results.json`: `b836410af1430858826be2a8e943c3034b870a47a7f36e57b690c842d6cff2c9`

## Frozen model and runtime

The learned residual was exactly the M6-bound model:

- model: `cross-encoder/nli-deberta-v3-base`;
- revision: `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`;
- weights: `model.safetensors`;
- weights SHA-256: `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`;
- tokenizer: `spm.model`;
- tokenizer SHA-256: `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd`;
- license: Apache-2.0.

Resolved execution environment included Python 3.12.3, Transformers 4.57.6, PyTorch 2.13.0, sentencepiece 0.2.1, safetensors 0.8.0, huggingface-hub 0.36.2, protobuf 6.33.6, one PyTorch thread, and one interop thread.

## Calibration execution

The execution used only the frozen M5 calibration construction:

- 64 calibration units;
- 512 proposition rows;
- 1,280 alignment rows;
- 768 evidence-group rows;
- 640 claim/safety rows;
- 56 readiness requirements;
- 128 learned residual requests;
- 7 preregistered threshold candidates.

The raw learned outputs were written and hashed before M5 gold was read for registered metric evaluation and parameter selection. The raw-score SHA-256 is:

`94a0fd23901fd250513e3415a131ab21ec1a1acbc605b53457fd4e18820ded00`

Post-execution arithmetic independently reconstructed all seven candidates and returned:

`PASSED_A45BM6_DETERMINISTIC_RECONSTRUCTION`

## Candidate geometry and frozen parameter

Six of the seven preregistered candidates satisfied all 56 readiness requirements:

| alignment confidence | feasible | requirements passed | slot-relation macro F1 | predicate-paraphrase recall | final-relation macro F1 |
|---:|:---:|---:|---:|---:|---:|
| 0.60 | yes | 56/56 | 0.9462257392 | 1.000000 | 1.000000 |
| 0.65 | yes | 56/56 | 0.9429798832 | 1.000000 | 1.000000 |
| 0.70 | yes | 56/56 | 0.9401619842 | 1.000000 | 1.000000 |
| 0.75 | yes | 56/56 | 0.9397162976 | 1.000000 | 1.000000 |
| 0.80 | yes | 56/56 | 0.9379385241 | 1.000000 | 1.000000 |
| 0.85 | yes | 56/56 | 0.9358529853 | 0.984375 | 0.9989669970 |
| 0.90 | no | 55/56 | 0.9323162361 | 0.890625 | 0.9926753449 |

The preregistered selection order therefore freezes:

`alignment_confidence_min = 0.60`

The selected candidate passes **56/56** requirements. Threshold 0.90 is the only infeasible registered point, because predicate-paraphrase recall falls below its readiness floor.

The breadth of the feasible interval is useful calibration evidence: the pass does not depend on a single isolated threshold. It still must not be interpreted as external generalization evidence.

## Selected calibration metrics

At the frozen 0.60 threshold:

- target-proposition F1: 1.0;
- proposition precision: 1.0;
- proposition recall: 1.0;
- scope-compatibility macro F1: 1.0;
- slot-relation macro F1: 0.9462257392420863;
- sufficiency macro F1: 1.0;
- polarity macro F1: 1.0;
- final-relation macro F1: 1.0;
- claim-category macro accuracy: 1.0;
- ENTAILED recall: 1.0;
- CONTRADICTED recall: 1.0;
- UNKNOWN recall: 1.0;
- conflicting-evidence recall: 1.0;
- supported precision: 1.0;
- supported recall: 1.0;
- minimal-sufficient-group exact match: 1.0;
- predicate mismatch rejection: 1.0;
- predicate paraphrase match recall: 1.0;
- false-supported safety: 0.0;
- different-condition false-conflict rate: 0.0;
- no-target false-positive rate: 0.0.

All registered safety, scope-mismatch, insufficiency, refutation, conflict, and claim-composition floors passed.

## What this result establishes

M6 establishes that the frozen TPAG implementation is **calibration-ready on the registered M5 calibration construction** under the registered 56-requirement gate. It also establishes that the specific M3 failure geometry is not reproduced on this calibration substrate: explicit scope mismatch handling, insufficiency, refutation, conflict, and safety composition all satisfy their registered floors.

The result does **not** establish that TPAG generalizes to unseen fresh data, that it performs at these levels on the sealed A4.5a validation partition, that it succeeds on the confirmatory partition, or that it has demonstrated real-world operational performance. M5 was explicitly registered as calibration evidence, not independent generalization evidence.

## Sealed boundaries after PASS

The completed M6 execution records:

- A4.5a fresh-validation rows scored: 0;
- confirmatory queries scored: 0;
- confirmatory records inspected: 0;
- closed M2/M3 rows rescored: 0;
- future-validation rows constructed: 0;
- model-family comparisons: 0;
- prompt searches: 0;
- post-result rescue authorized: false;
- next checkpoint authorized: false.

A4.5c remains permanently ineligible and is not repurposed.

## Closure decision

The M6 parameter is frozen at `alignment_confidence_min = 0.60` and A4.5b-M6 is closed as a successful calibration-readiness checkpoint.

No validation execution follows automatically from this PASS. Any use of fresh A4.5a validation, confirmatory evidence, or any separately named post-M6 checkpoint requires explicit separate authorization under the existing governance.

# A4.5b calibration failure diagnostic

A4.5b is closed as a negative calibration-readiness result. The authoritative scientific status is `FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED`.

## What passed

The frozen calibration stack produced perfect end-relation behavior on the 360 calibration pairs at the selected threshold pair: final relation macro F1 = 1.0, ENTAILED recall = 1.0, CONTRADICTED recall = 1.0, UNKNOWN recall = 1.0, sufficiency macro F1 = 1.0, polarity macro F1 = 1.0, and claim-composition accuracy = 1.0. Minimal evidence span precision and recall were also 1.0.

These values do not override the component-level validity requirements registered in A4.5a and A4.5b.

## What failed

No one of the 12,050 preregistered relevance/sufficiency threshold pairs satisfied every readiness requirement. The selected best-any pair was relevance = 8.1 and sufficiency = 0.99, with zero feasible candidates.

Three preregistered relevance requirements failed:

- relevance macro F1 = 0.7703703703703704, required >= 0.90;
- relevant recall = 0.8535714285714285, required >= 0.90;
- irrelevant recall = 0.7375, required >= 0.90.

The selected-threshold relevance confusion matrix is:

| Gold | Predicted relevant | Predicted irrelevant |
| --- | ---: | ---: |
| RELEVANT | 239 | 41 |
| IRRELEVANT | 21 | 59 |

The failure geometry is concentrated in semantically important boundary cases. All 40 `relevant_but_insufficient` pairs were classified as irrelevant by the relevance stage. Of the 40 `cross_document_irrelevance` pairs, 21 were classified as relevant. All 40 `same_domain_irrelevance` pairs were correctly classified as irrelevant. Of the 40 `temporal_insufficiency` pairs, 39 were classified as relevant and one as irrelevant.

This means the current relevance measurement is not a valid proxy for the conceptual distinction AERF needs. In particular, it conflates two different questions: whether evidence is topically or semantically aligned with an atom, and whether that evidence is sufficiently informative to license support or refutation. The final relation layer can still return UNKNOWN correctly because the sufficiency gate compensates for this error, but the internal factorization fails its own registered measurement requirements.

## Scientific interpretation

The result does not justify changing thresholds, substituting a model, weakening readiness floors, or opening fresh validation. It demonstrates that the current scalar relevance/alignment primitive is the bottleneck. Any repair must therefore be methodological and registered before new fitting or validation.

The next scientific work must examine a relevance/alignment construction that can separately preserve:

1. topical or semantic alignment between atom and candidate evidence;
2. evidence scope and entity/attribute matching;
3. insufficiency as a separate state rather than a low-relevance surrogate;
4. explicit non-evidence rejection for cross-document distractors;
5. deterministic composition into the existing AERF relation semantics.

No claim is made here that any particular replacement method will succeed. Model binding, threshold fitting, fresh validation, and confirmatory scoring remain unauthorized.

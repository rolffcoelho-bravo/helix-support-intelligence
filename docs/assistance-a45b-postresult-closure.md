# Phase 4 A4.5b post-result closure

## Closure status

A4.5b is permanently closed with scientific status `FAILED_CALIBRATION_READINESS_NO_VALIDATION_AUTHORIZED`.

The original calibration inference run scored exactly 360 calibration pairs and produced no validation or confirmatory scores. A deterministic recovery subsequently consumed only those immutable raw scores, reconstructed the full preregistered 12,050-point threshold grid twice, and independently reproduced the same negative readiness decision without a second model inference.

## Authoritative provenance

- original inference workflow run: `32581433921`;
- original partial artifact: `9477913279`;
- original artifact ZIP SHA256: `0bb5bd16c5a582f69e6421d0017bfa3e33e60cb5d4b31a005f6929d6cfc2c633`;
- raw calibration score SHA256: `fe6b22278945e4df094fe8d0314706281ae6fd7915a2f613606b29f63c5d32b6`;
- deterministic recovery workflow run: `32581996227`;
- recovery job: `97052657366`;
- recovery main SHA: `735eb357bf8693b3aa0d7a881a748a5123173006`;
- recovery artifact: `9478056833`;
- recovery artifact ZIP SHA256: `661bffe06934098d1c6764c0705f3d70a9216f04a09b2e00d6ba4a81db184598`;
- independent recovery audit: `PASSED_A45B_DETERMINISTIC_RECOVERY_RECONSTRUCTION`.

## Frozen result

The selected best-any threshold pair is relevance `8.1` and sufficiency `0.99`. No candidate in the full preregistered grid satisfied all readiness gates.

The current stack passes final relation, sufficiency, polarity, evidence-span, contradiction-safety, context-contamination, and claim-composition metrics on the calibration fixture, but fails three relevance requirements: relevance macro F1, relevant recall, and irrelevant recall.

The negative result is not eligible for post-result threshold rescue or model substitution inside A4.5b.

## Sealed boundaries

The following remain zero and sealed:

- A4.5a fresh-validation rows materialized: 0;
- A4.5a fresh-validation rows scored: 0;
- original 68-query confirmatory records inspected: 0;
- original confirmatory queries scored: 0;
- A4.4a validation rows rescored: 0;
- A4.4d validation rows rescored: 0.

A4.5c remains registered as the fresh validation-only execution checkpoint, but it is currently **ineligible and unauthorized** because its frozen eligibility condition requires A4.5b calibration readiness to pass.

## Required next action class

The only admissible next scientific action is a separately registered methodological decision addressing the relevance/alignment measurement failure. Such a decision may analyze the closed calibration failure and external literature, but it must not perform new model inference, threshold fitting, validation scoring, confirmatory inspection, or candidate shopping unless those activities are separately preregistered and approved later.

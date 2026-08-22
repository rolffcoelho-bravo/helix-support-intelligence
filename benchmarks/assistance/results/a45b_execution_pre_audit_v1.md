# A4.5b pre-execution audit

Status: **REGISTERED_PRE_EXECUTION_NO_RESULTS**

A4.5b binds one AERF implementation before calibration results are opened. The alignment/relevance component is `cross-encoder/ms-marco-MiniLM-L6-v2` at revision `c5f2b386de279a97c53a702dd5189d1c407160dc`. The sufficiency/polarity component is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` at revision `0e2603d5d3d3ef9b2910814b34eebe1a2101da65`. Exact safetensor SHA256 values are frozen in the binding and verified before inference.

Only the 40 registered calibration units are authorized: 360 semantic pairs and 360 claim-composition cases. Their hashes must reproduce the A4.5a calibration hashes exactly. The joint threshold grid contains 12,050 preregistered relevance/sufficiency pairs.

At this checkpoint no model inference has occurred, no threshold has been fit, no performance result is known, no fresh-validation row has been materialized or scored, and no confirmatory query record has been inspected. The 20-unit fresh validation partition and the 68-query confirmatory partition remain sealed.

Scientific failure during calibration will be preserved. A4.5b does not authorize threshold rescue, model substitution, candidate comparison, fresh validation, or A4.5c.

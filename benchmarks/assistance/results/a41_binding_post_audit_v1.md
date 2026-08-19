# Phase 4 A4.1 binding post-audit

## Verdict

**PASSED, NO PERFORMANCE RESULTS.**

A4.1 binds the assistance runtime before any G0, G1, or G2 assistance-performance scoring. The audited implementation preserves the frozen A4.0 methodology and leaves the 240-query development and 68-query confirmatory performance surfaces unopened.

## Repository validation

The successful repository-wide gate was GitHub Actions run `32311693388`, job `96255805179`, on tested head `a5da94ccd76c7c79f710ddd0ae66c51cea70525c`.

- Ruff check passed.
- Ruff format check passed across 99 files.
- strict mypy passed across 45 source files.
- pytest passed: 125 passed, 0 failed, 1 existing non-blocking warning.
- Phase 1 offline data contracts passed.
- Phase 3 retrieval preflight passed with 147 eligible documents, 7 current conflict fixtures, 5 current untrusted-content fixtures, and zero retrieval scores computed.
- A4.1 assistance preflight reconstructed 60 development intents / 240 queries and 17 confirmatory intents / 68 queries, with 30 repeatability cases and 60 latency cases.
- the repository and benchmark-runtime preflights both reported zero generator calls, zero NLI calls, and zero assistance performance scores.
- publication audit passed.

## Frozen A4.1 binding

G1 and G2 share the exact generator snapshot `gpt-5.4-mini-2026-03-17`, with reasoning effort `none`, temperature `0.0`, 512 maximum output tokens, tools disabled, response storage disabled, and no automatic quality-call retries.

Exact prompt and schema hashes:

- system prompt: `ab0a472476fbf023a154f1f1c6fa29c3c06da8590ac05b248aba209cadca878b`;
- request template: `433fdd0b2374b55cdea1bbda6e22fa93628648bb07a7468ab0a520b4fc00fd15`;
- candidate output schema: `3c0ec9a6864b0ac316197be48e399264ce841ac0ac48399bd975d858f36a963f`.

G2's runtime verifier is `cross-encoder/nli-deberta-v3-small` at revision `fa2804872c3b4bd748f38c0185cc85775361e735`, architecture family `deberta-v3`, entailment label 1, threshold 0.80.

The independent evaluation verifier is `cross-encoder/nli-MiniLM2-L6-H768` at revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, architecture family `minilm2-roberta`, entailment label 1, threshold 0.80. The runtime and evaluation verifier families are deliberately different.

The diagnostic subset file SHA-256 is `60f8a4ec2be040410d355a06dfc7d36608d3fa990962179c9d99da29193f2c1e`. The benchmark runtime SHA-256 is `fae0a81512f99219f243a29284db2cc1d28118a9a87247d16785616f131817d8`, and its dependency lock SHA-256 is `f094583ce93b48eeac5ed70fba71390c163779a73e782e797742b5501b8ea796`.

Frozen P95 latency ceilings are 100 ms for G0, 6,000 ms for G1, and 8,000 ms for G2. Frozen maximum estimated provider cost per request is USD 0 for G0 and USD 0.005 for G1/G2.

## Manual code and scope audit

Before these audit artifacts were added, the branch changed exactly 12 expected A4.1 files. No Phase 2 or Phase 3 scientific code was modified. R3.2 evidence, `retrieval-selected-v1`, and the HelixBank records were untouched. `pyproject.toml` is unchanged, so heavy ONNX/Transformers dependencies remain benchmark-scoped rather than becoming normal package dependencies. The temporary dependency-lock workflow is absent from the merge candidate.

Candidate-visible query/evidence construction excludes the registered benchmark labels and gold fields. No generated candidate output, development metric, confirmatory metric, or assistance result artifact existed during this audit.

## Defects caught before closure

Three non-scientific issues were detected and repaired:

1. The first full gate required canonical Ruff formatting in the new preflight and guard tests. Only formatter-prescribed layout changed.
2. The temporary dependency-lock workflow initially could not stage the explicit `configs/models` binding path because the repository ignore rule also matches directories named `models`. The workflow was corrected to force-add only that explicit file after the lock and no-call preflight had already succeeded.
3. One concurrent lock-generation workflow lost a push race because another run had already committed the same deterministic generated lock. The first committed lock was retained; the temporary workflow was removed before merge.

None of these changed the generator, prompts, verifier identities or thresholds, diagnostic subsets, pricing, budgets, metrics, hypotheses, or A4.0 methodology.

## Boundary and limitation

A4.1 validates request construction and provider/model provenance statically. It intentionally does not call the OpenAI API, download or execute the NLI models, or compute any assistance quality metric. Therefore this checkpoint does **not** establish provider runtime success or model quality and cannot support a claim that G1 or G2 improves over G0.

The first empirical provider/NLI compatibility test belongs to the controlled development execution. The 68-query confirmatory partition remains unopened for assistance scoring.

**Rerun A4.0 required: no.**  
**A4.1 rework required: no.**  
**Assistance performance result: none.**

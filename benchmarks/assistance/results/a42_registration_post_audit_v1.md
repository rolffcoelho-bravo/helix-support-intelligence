# Phase 4 A4.2 pre-execution registration audit

## Verdict

**PASSED, NO PERFORMANCE RESULTS.**

A4.2 is registered as a development-only execution. The audited implementation preserves the frozen A4.0 methodology and A4.1 runtime binding, reconstructs 60 development intents / 240 development queries, and leaves all 17 confirmatory intents / 68 confirmatory queries unopened.

## Repository validation

The successful pre-audit repository gate was GitHub Actions run `32314875083`, job `96264970064`, on tested head `9ea63e39925d15fc865f05dcdbe74afc051ee06f`.

- Ruff check passed.
- Ruff format check passed across 108 files.
- strict mypy passed across 47 source files.
- pytest passed: 131 passed, 0 failed, 1 existing non-blocking warning.
- Phase 1 offline data contracts passed.
- Phase 3 retrieval preflight passed.
- A4.1 assistance preflight passed with zero generator calls, zero NLI calls, and zero assistance scores.
- A4.2 preflight passed with 60 development intents / 240 queries, zero confirmatory intents or queries opened, and development adversarial counts of 60 direct-injection, 60 citation-spoof, 16 indirect-injection, and 7 archived-distractor cases.
- publication audit passed.

## Registered execution surface

A4.2 is configured for one quality pass over G0, G1, and G2, yielding 720 expected development quality records. Development-only adversarial evaluation yields 429 expected records. The frozen repeatability diagnostic yields 270 records from 30 cases, three repetitions, and three candidates. The frozen latency diagnostic yields 540 timed records from 60 cases, three timed passes, and three candidates, plus ten warmup requests per candidate.

Inference remains the registered paired cluster bootstrap by intent with 5,000 replicates and seed `20260819`. Provider automatic quality retries remain zero. Both pinned NLI components use the frozen A4.1 batch size of eight.

The one-shot workflow is triggered only when its registered workflow file reaches `main`. It requires the provider credential before model runtime, copies the exact A4.1 benchmark dependency lock to the execution and verification entry points, hashes all scientific inputs before the first provider request, rechecks them after execution, independently reconstructs raw evidence, and freezes checksums before artifact upload.

## Pre-score defects caught and repaired

Four issues were found before any assistance performance result existed.

1. The initial NLI implementation evaluated one sentence pair per ONNX call despite the A4.1 batch-size-eight binding. A batched adapter now executes groups of up to eight sentence pairs, with a final partial batch allowed.
2. The initial deterministic G0 rendering placed citations after terminal punctuation, which would have separated a factual claim from its citations under the frozen sentence segmenter. The same citations are now placed before the terminal period.
3. The first PR quality gate found only static hygiene issues in new A4.2 code: unused imports/directives, import ordering, formatting differences, and three over-length literals. Canonical Ruff fixes and formatting were applied without changing evaluation behavior.
4. The verifier initially mislabeled a provider-cost subtotal as covering all A4.2 calls even though the compatibility probe and latency warmups were excluded. The final artifact now names the subtotal according to its exact scope and explicitly records those exclusions. The registered adoption cost rule was not changed.

None of these repairs was informed by model-performance results. No generator output, NLI benchmark output, development metric, hypothesis result, latency result, adversarial result, or confirmatory result had been opened.

## Scope audit

Before these audit artifacts were added, the branch changed exactly 11 expected A4.2 files. No Phase 2 or Phase 3 scientific file, R3.2 evidence, `retrieval-selected-v1`, A4.0 protocol, A4.1 binding, or HelixBank record was modified. The temporary formatter workflow is absent from the merge candidate.

The full A4.0 adversarial surface spans 77 intents. Because A4.2 is explicitly development-only, this execution applies the same registered attack transformations only to the 60 development intents. The remaining confirmatory-intent adversarial variants are deferred rather than being used to open the confirmatory partition indirectly.

## Boundary

This checkpoint is registration evidence, not model-performance evidence. Actual OpenAI request compatibility and pinned NLI runtime compatibility are first tested inside the controlled one-shot execution. If the provider credential, generator request, or pinned NLI runtimes are unavailable, the workflow must fail before the first development performance score is opened.

**A4.0 rework required: no.**  
**A4.1 rework required: no.**  
**A4.2 performance result: none yet.**

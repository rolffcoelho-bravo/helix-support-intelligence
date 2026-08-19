# Phase 4 A4.1 assistance binding

A4.1 binds the concrete evidence-grounded assistance runtime before any assistance candidate is scored. It implements the remaining choices permitted by the frozen A4.0 protocol while keeping the development and confirmatory performance surfaces unopened.

## Frozen generator

G1 and G2 use the same OpenAI Responses API generator snapshot:

- model: `gpt-5.4-mini-2026-03-17`;
- reasoning effort: `none`;
- temperature: `0.0`;
- maximum output: 512 tokens;
- tools: disabled;
- response storage: disabled;
- automatic quality-call retries: zero;
- structured output: strict JSON Schema response format.

The exact system prompt and request template are repository files under `prompts/assistance/`. Their UTF-8 bytes, byte lengths, and SHA-256 digests are recorded in `configs/models/assistance_binding_a41_v1.json`. Evidence serialization is deterministic, documents are ordered by document ID, and evaluator-only labels are removed before candidate input is constructed.

## Runtime and evaluation verifiers

G2 uses a local CPU ONNX sentence-support gate based on `cross-encoder/nli-deberta-v3-small` at exact revision `fa2804872c3b4bd748f38c0185cc85775361e735`. The registered entailment label is 1, the threshold is 0.80, maximum sequence length is 512, and batch size is 8.

Independent grounding evaluation uses `cross-encoder/nli-MiniLM2-L6-H768` at exact revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, also on CPU ONNX with entailment label 1, threshold 0.80, maximum sequence length 512, and batch size 8. Its `minilm2-roberta` architecture family differs from the G2 runtime verifier's `deberta-v3` family, preserving the A4.0 independence requirement.

The verifier cannot repair or regenerate a G2 answer. Unsupported or uncited factual content fails closed to the registered low-confidence state; verifier runtime failure maps to the registered system-failure state.

## Pricing and budgets

The dated pricing snapshot is 2026-08-19 and records OpenAI standard API rates for the bound generator as USD 0.75 per million input tokens, USD 0.075 per million cached input tokens, and USD 4.50 per million output tokens. Candidate selection uses the uncached input rate even when provider caching occurs, so cache state cannot improve the adoption calculation. Provider-reported usage may be retained separately as diagnostic evidence.

Local NLI provider cost is recorded as zero because no external model API is billed. That value does not represent CPU hardware or energy cost, which remains outside this provider-cost measure.

Frozen candidate ceilings are:

| Candidate | P95 latency ceiling | Maximum estimated provider cost/request |
|---|---:|---:|
| G0 | 100 ms | USD 0.000 |
| G1 | 6,000 ms | USD 0.005 |
| G2 | 8,000 ms | USD 0.005 |

These are pre-result ceilings and cannot be relaxed after performance results are observed.

## Deterministic diagnostic subsets

A4.1 freezes two development-only diagnostic subsets before scoring:

- repeatability: 30 cases, three repetitions;
- latency: 60 cases, ten warmup requests and three timed passes.

Selection is deterministic within case type using SHA-256 ordering of query IDs. All five development conflicting-evidence cases are included in both subsets. The repeatability subset is contained in the latency subset. No confirmatory query belongs to either diagnostic set.

The exact query IDs and case-type quotas are stored in `configs/models/assistance_a41_subsets_v1.json`.

## Reproducible runtime

Heavy assistance-evaluation dependencies remain scoped to `benchmarks/assistance/runtime_a41.py` through PEP 723 metadata and the committed `runtime_a41.py.lock`. They are not added to the main Helix runtime dependency surface.

The A4.1 preflight verifies:

- A4.0 corpus and partition continuity;
- exact prompt, schema, runtime, subset, and dependency-lock hashes;
- exact generator and NLI identities;
- verifier-family independence;
- deterministic development-only subset reconstruction;
- frozen pricing, latency, and cost ceilings;
- zero generator calls;
- zero NLI calls;
- zero assistance performance scores.

## Evidence boundary

A4.1 is not a model benchmark and creates no evidence that G1 or G2 improves over G0. No OpenAI response is requested by the preflight, no NLI model is executed, and neither the 240-query development set nor the 68-query confirmatory set is scored.

Performance evaluation belongs to a later execution that consumes this immutable binding. Any result-driven change to model identity, prompt bytes, decoding, verifier identities or thresholds, diagnostic subsets, latency ceilings, or cost ceilings requires a new version rather than modifying A4.1 v1.

# Helix Support Intelligence

> Production-oriented search, routing, recommendation, and evidence-grounded assistance for customer-support operations.

[![CI](https://github.com/rolffcoelho-bravo/helix-support-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/rolffcoelho-bravo/helix-support-intelligence/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-Phase%202%20verified-16a34a)](#project-status)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-6b7280)](LICENSE)

Helix Support Intelligence is a public applied-AI engineering project focused on a demanding operational question:

> How can a support platform automate useful decisions while keeping evidence, uncertainty, safety, latency, and cost observable?

The system combines machine learning, information retrieval, ranking, and generative AI inside one controlled support workflow. Language models are treated as replaceable components rather than as the system itself.

## What Helix does

Helix is designed around a fictional digital-banking service desk. It processes customer-support requests and helps determine the appropriate operational outcome:

- identify intent and route tickets;
- detect uncertain or out-of-scope requests;
- retrieve and rank relevant knowledge articles;
- recommend useful policy or resolution paths;
- draft responses grounded in approved evidence;
- attach traceable citations;
- request clarification or escalate when automation is unsafe.

The repository uses public or fictional data. It does not connect to real bank accounts, initiate transactions, or contain real customer information.

## Verified routing evidence

Only results produced by frozen, reproducible evaluation are reported.

| System | Routing macro-F1 | nDCG@10 | Unsupported claims | P95 latency | Cost/request |
|---|---:|---:|---:|---:|---:|
| Routing reference | pending | pending | pending | pending | pending |
| Phase 2 routing candidate | **0.9016** | pending | pending | pending | pending |

The routing result above is the official BANKING77 confirmatory-test macro-F1 for the frozen Phase 2 A2 router. The test split was not used for model, calibration, or threshold selection.

Additional confirmatory evidence:

- balanced accuracy: **0.9016**;
- top-3 recall: **0.9744**;
- 15-bin ECE: **0.0169**;
- Brier score: **0.1456**;
- full-automation routing error risk: **9.84%**;
- routing error risk at exactly 75% confidence-ranked coverage: **1.95%**;
- selective-minus-full risk: **-7.89 percentage points**, with paired-bootstrap 95% CI **[-8.78 pp, -6.96 pp]**.

The calibration result is deliberately interpreted separately from the selective-routing result. On development evidence, calibration did not reduce the registered mixed routing-cost endpoint. On the untouched BANKING77 test, the independently estimable in-domain calibrated-minus-raw cost difference was `+0.0041`, with 95% CI `[-0.0260, 0.0300]`, so the in-domain cost comparison is **inconclusive**.

Permanent evidence is under `benchmarks/routing/results/confirmatory_test_v1.{json,md}` and `benchmarks/routing/results/confirmatory_post_audit_v1.{json,md}`.

## Product workflow

```mermaid
flowchart LR
    A[Support request] --> B[Routing]
    B --> C[Knowledge search]
    C --> D[Evidence-grounded assistance]
    D --> E{Validated outcome}
    E -->|Supported| F[Response or recommendation]
    E -->|Uncertain| G[Human escalation]
```

The public [product contract](docs/product-contract.md) defines the bounded v1 scope and terminal decisions. The [architecture document](docs/architecture.md) describes the production boundaries.

## Core capabilities

### Intelligent ticket routing

Helix models fine-grained customer intent, operational destination, uncertainty, and out-of-scope behaviour. Its purpose is not to maximize automation blindly, but to distinguish decisions that can be automated from those requiring human judgment.

The verified Phase 2 routing configuration uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with a multiclass logistic-regression classifier, temperature scaling at `0.457974`, and automatic routing only when maximum calibrated class probability is at least `0.892704`.

### Hybrid search and ranking

The retrieval layer combines lexical and semantic search with reranking. Search quality is evaluated independently from generated answers so retrieval failures remain visible instead of being hidden behind fluent text.

Hybrid retrieval and reranking are the current development focus.

### Evidence-grounded assistance

Generated responses are constrained by retrieved knowledge and returned with citations. Unsupported, ambiguous, or conflicting cases are designed to fail safely through clarification or escalation.

### Next-best-resolution recommendation

Helix ranks relevant articles, approved troubleshooting paths, clarification questions, and escalation destinations for support agents. Recommendations remain read-only.

### Evaluation and observability

Model quality, retrieval relevance, calibration, citation behaviour, latency, cost, and operational decisions are treated as measurable system properties. The repository emphasizes reproducible comparisons against clear baselines.

## Routing model comparison

The routing study used a bounded comparison set rather than open-ended model searching.

| Model | Development macro-F1 | Balanced accuracy | Top-3 recall | Outcome |
|---|---:|---:|---:|---|
| A1 TF-IDF + logistic regression | 0.8422 | 0.8407 | 0.9534 | simpler reference |
| **A2 MiniLM embeddings + logistic regression** | **0.8986** | **0.8963** | **0.9732** | **selected** |
| A3 fixed three-epoch fine-tuning | 0.6898 | 0.7105 | 0.9226 | negative result |

Temperature scaling improved A2 validation ECE from about `0.2910` to `0.0162`, Brier from `0.2501` to `0.1398`, and NLL from `0.6683` to `0.3350` without changing classification decisions.

The separate frozen development OOS benchmark contains 160 support-like queries across 20 categories. A2 achieved OOS AUROC `0.8956`, but the in-domain false-positive rate at at least 95% OOS recall remained `0.4342`. The OOS set is therefore reported as development evidence rather than independent confirmation.

The routing cost matrix uses explicit synthetic decision-analysis units and is not claimed to represent real-bank economics.

## Confirmatory routing result

The official 3,080-row BANKING77 test split remained separate from model, calibration, cost, OOS, and threshold selection. The confirmatory evaluation then measured the already-fixed routing configuration.

| Confirmatory quantity | Result |
|---|---:|
| Accuracy | 0.9016 |
| Macro-F1 | 0.9016 |
| Balanced accuracy | 0.9016 |
| Top-3 recall | 0.9744 |
| ECE | 0.0169 |
| Brier | 0.1456 |
| NLL | 0.3467 |
| Full risk | 9.84% |
| Risk at 75% coverage | 1.95% |
| Selective-minus-full risk | -7.89 pp |

At the fixed application threshold `0.892704`, realized test coverage is **74.12%** and selective risk is **1.88%**. The threshold was not moved after observing the test result.

Three unsafe high-risk wrong automatic routes remain at that threshold. This residual failure mode is part of the published limitations.

An independent post-result verification reconstructed the event costs, routing decisions, point estimates, and 5,000-replicate bootstrap intervals from the stored evidence.

## Engineering character

Helix is developed as a production-shaped repository rather than a notebook demonstration. The engineering surface includes:

- typed Python packages and API contracts;
- reproducible dependency and data management;
- automated testing and quality checks;
- versioned models, corpora, and configurations;
- container-ready local execution;
- experiment tracking and model lifecycle management;
- traces, metrics, dashboards, and safe failure behaviour;
- public documentation for data provenance, limitations, and responsible use.

Exploratory analysis can support development, but authoritative behaviour lives in versioned code and contracts.

## Quick start

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rolffcoelho-bravo/helix-support-intelligence.git
cd helix-support-intelligence
make setup
make quality
uv run helix
```

Stable development commands:

```bash
make lint
make typecheck
make test
make data-check
make publication-audit
make quality
```

## Technology direction

The project is centered on Python, FastAPI, scikit-learn, PyTorch, Transformers, hybrid information retrieval, MLflow, OpenTelemetry, PostgreSQL, Redis, Docker Compose, and GitHub Actions.

A more complex component is adopted only when it demonstrates measurable value over a simpler baseline.

## Evaluation philosophy

Helix separates five forms of evidence:

| Layer | Representative evidence |
|---|---|
| Routing | classification quality, calibration, selective coverage |
| Retrieval | ranking relevance, recall, latency |
| Assistance | factual support, citation quality, refusal behaviour |
| Safety | privacy, adversarial robustness, bounded actions |
| System | reliability, latency, cost, observability |

Results are published only after reproducible evaluation. Empty result fields remain marked rather than being filled with projections.

## Responsible-use boundaries

Helix is a research and portfolio system built with fictional or redistributable data. It is not a banking product, financial adviser, authentication service, or autonomous transaction agent.

The public implementation does not include:

- credentials, secrets, or private prompts;
- personal or real customer data;
- unrestricted tool execution;
- autonomous financial actions;
- claims of real-world impact unsupported by deployment evidence.

Security concerns should be reported through [SECURITY.md](SECURITY.md). Do not open a public issue for a suspected vulnerability.

## Project status

The `main` branch contains the reproducible engineering foundation, data/evaluation contracts, and the verified routing subsystem. Retrieval and reranking are under active development in the public repository, followed by evidence-grounded assistance, safety/observability integration, and system-level validation.

The completed routing work supports three distinct conclusions:

- A2 classification generalizes strongly on untouched BANKING77 test data;
- temperature scaling provides strong probability calibration but does not establish lower routing cost;
- selective abstention is independently supported as a risk-control mechanism at the registered coverage level.

The project targets a documented `v1.0.0` release. Capabilities beyond the v1 product contract will be treated as separate later work rather than unfinished obligations of the initial repository.

## Author

**Rodolfo Pereira**  
ShockBridge Pulse Research Lab

## Citation

Pereira, Rodolfo. (2026). *Helix Support Intelligence: Production-Oriented Search, Routing, Recommendation, and Evidence-Grounded Support AI*. ShockBridge Pulse Research Lab. Python research software.

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

The public Helix Support Intelligence software and documentation are licensed under the [Apache License 2.0](LICENSE).

External datasets, model weights, and third-party components retain their original licences and are documented with the relevant public artifacts. The Apache-2.0 licence applies only to material intentionally published in this repository.

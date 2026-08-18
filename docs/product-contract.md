# Public Product Contract

## Purpose

Helix is a research and portfolio platform for bounded customer-support decisions in a fictional digital-banking setting. It combines ticket routing, information retrieval, recommendation, evidence-grounded assistance, selective abstention, and human escalation.

The public product claim is intentionally narrow: Helix aims to make uncertainty, evidence, safety, latency, and cost observable and testable. It does not claim deployment by a real bank or measured customer impact.

## Included in v1

- English-language fictional banking-support tickets.
- Intent and queue routing with explicit uncertainty.
- Out-of-scope detection and abstention.
- Hybrid retrieval and reranking over a redistributable policy corpus.
- Citation-grounded response assistance.
- Read-only next-best-policy and resolution recommendation.
- Human escalation with stable reason codes.
- Reproducible offline evaluation, observability, and local deployment.

## Excluded from v1

- Real customer or account data.
- Authentication or account access.
- Payments, refunds, card blocking, or any irreversible financial action.
- Unrestricted agents or arbitrary tool execution.
- Pricing, liquidity, credit, or market-risk models.
- Speech, image, multilingual, mobile, or Kubernetes requirements.
- Claims of production effectiveness based only on synthetic users.

Excluded capabilities require a separate post-v1 proposal and cannot delay v1 completion.

## Terminal decisions

Every processed request must end in exactly one declared state:

| State | Public meaning |
|---|---|
| `ANSWER_WITH_EVIDENCE` | Return a supported response with resolvable evidence. |
| `AUTO_ROUTE` | Route a sufficiently supported ticket to a declared queue. |
| `RECOMMEND_TO_AGENT` | Offer a read-only recommendation to a human agent. |
| `ASK_FOR_CLARIFICATION` | Request information needed for a bounded decision. |
| `ESCALATE_LOW_CONFIDENCE` | Transfer because support is insufficient. |
| `ESCALATE_OUT_OF_SCOPE` | Transfer because the request exceeds the product boundary. |
| `ESCALATE_CONFLICTING_EVIDENCE` | Transfer because approved evidence conflicts. |
| `ESCALATE_SAFETY_RISK` | Transfer because a safety condition blocks automation. |
| `ESCALATE_SYSTEM_FAILURE` | Transfer because a required component failed. |

There is no silent fallback to unconstrained generation.

## Completion

The repository concludes at a measured, documented `v1.0.0` release. Negative or inconclusive results remain valid outputs when they are reproducible and the resulting fallback or escalation behaviour is explicit.

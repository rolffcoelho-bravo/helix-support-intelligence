# A4.5b-M1 literature rationale

This note records the literature basis for the post-A4.5b methodological decision. It is a design rationale, not a novelty claim and not evidence that the selected replacement implementation will pass future calibration or validation.

## Problem exposed by A4.5b

A4.5b bound `cross-encoder/ms-marco-MiniLM-L6-v2` as the authoritative relevance/alignment primitive. On the frozen calibration set, the final AERF relation decisions were perfect at the selected threshold pair, but the relevance construct failed its preregistered component requirements. All 40 `relevant_but_insufficient` cases were classified as irrelevant, while 21 of 40 `cross_document_irrelevance` cases were classified as relevant. The result demonstrates that generic passage-ranking relevance is not a valid stand-in for the evidential compatibility construct required by AERF.

## Literature anchors

### Evidence sufficiency is a separate task

Atanasova, Simonsen, Lioma, and Augenstein (2022), *Fact Checking with Insufficient Evidence*, TACL 10:746–763, DOI `10.1162/tacl_a_00486`, explicitly studies whether the available evidence is sufficient for fact checking and shows that missing-evidence detection is its own difficult prediction problem. This directly supports keeping evidence compatibility/relevance separate from sufficiency rather than encoding insufficiency as low relevance.

### Retrieval relevance and verification utility are not equivalent

Zhang, Zhang, Guo, de Rijke, Fan, and Cheng (2023), *From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification*, Findings of EMNLP 2023, DOI `10.18653/v1/2023.findings-emnlp.422`, argues that off-the-shelf retrieval models optimize generic relevance while fact verification needs evidence useful to a verifier. This is closely aligned with the A4.5b failure, where an MS MARCO reranker separated passage relevance differently from the registered evidential relevance construct.

### Entity and relation structure matters for evidential search

Wuehrl, Menchaca Resendiz, Grimminger, and Klinger (2024), *What Makes Medical Claims (Un)Verifiable? Analyzing Entity and Relation Properties for Fact Verification*, EACL 2024, DOI `10.18653/v1/2024.eacl-long.124`, treats entities and relations as core variables in claim anatomy and reports that evidence search is refined through entity normalization and added constraints. This supports an explicit compatibility representation over entity, predicate/relation, and scope constraints rather than relying on one undifferentiated scalar similarity score.

### Fine-grained evidence retrieval and aggregation matter

Chen, Kim, Sriram, Durrett, and Choi (2024), *Complex Claim Verification with Evidence Retrieved in the Wild*, NAACL 2024, DOI `10.18653/v1/2024.naacl-long.196`, uses claim decomposition, fine-grained evidence retrieval, claim-focused summarization, and aggregation. The work supports preserving atomic claims while reasoning over fine-grained and potentially aggregated evidence rather than forcing a single highest-scoring sentence to represent the entire evidence state.

Sriram, Xu, Choi, and Durrett (2024), *Contrastive Learning to Improve Retrieval for Real-World Fact Checking*, FEVER 2024, DOI `10.18653/v1/2024.fever-1.28`, shows that generic retrieval can miss evidence that is useful for fact checking and that task-specific signals such as subquestions and evidence answers can improve retrieval and downstream verification. This supports task-specific evidence compatibility but does not by itself justify any particular future model binding.

### Attribution remains difficult even with strong models

Li, Yue, Liao, and Sun (2024), *AttributionBench: How Hard is Automatic Attribution Evaluation?*, Findings of ACL 2024, DOI `10.18653/v1/2024.findings-acl.886`, reports that automatic claim-to-evidence attribution remains challenging and that many errors involve nuanced information. This supports retaining explicit component-level validity gates rather than trusting a strong final relation score alone.

### Decomposition must remain controlled

Hu, Long, and Wang (2025), *Decomposition Dilemmas: Does Claim Decomposition Boost or Burden Fact-Checking Performance?*, NAACL 2025, DOI `10.18653/v1/2025.naacl-long.320`, finds that decomposition can introduce noise and destabilize downstream verification. For Helix, the implication is to preserve the already registered minimal AERF atoms and avoid a new free-form decomposition layer as part of the relevance repair.

### Sufficiency and complementarity should be represented at evidence-set level

Alt, Hirsch, Basch, Dagan, and Glickman (2026), *User-Centric Evidence Ranking for Attribution and Fact Verification*, EACL 2026, DOI `10.18653/v1/2026.eacl-long.340`, emphasizes presenting sufficient information and finds incremental ranking better captures complementary evidence. This supports treating sufficiency as a property of a compatible evidence set rather than only of a single top-ranked span.

## Methodological implication

The literature does not identify one universally correct relevance model. It does, however, support four principles that directly address the A4.5b failure:

1. generic retrieval relevance is not the same construct as evidence useful for verification;
2. evidence compatibility and evidence sufficiency should be measured separately;
3. entity, relation/predicate, and contextual constraints should be represented explicitly;
4. sufficient evidence can be complementary and therefore must be allowed to accumulate across compatible spans.

The selected Helix design, **Scope-Conditioned Evidence Compatibility (SCEC)**, operationalizes those principles as an internal engineering methodology. No claim is made that the name or the overall concept is novel in the literature.

## References

- Atanasova, Pepa, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2022. “Fact Checking with Insufficient Evidence.” *Transactions of the Association for Computational Linguistics* 10: 746–763. https://aclanthology.org/2022.tacl-1.43/
- Zhang, Hengran, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi Cheng. 2023. “From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification.” *Findings of EMNLP 2023*: 6373–6384. https://aclanthology.org/2023.findings-emnlp.422/
- Wuehrl, Amelie, Yarik Menchaca Resendiz, Lara Grimminger, and Roman Klinger. 2024. “What Makes Medical Claims (Un)Verifiable? Analyzing Entity and Relation Properties for Fact Verification.” *EACL 2024*: 2046–2058. https://aclanthology.org/2024.eacl-long.124/
- Chen, Jifan, Grace Kim, Aniruddh Sriram, Greg Durrett, and Eunsol Choi. 2024. “Complex Claim Verification with Evidence Retrieved in the Wild.” *NAACL 2024*: 3569–3587. https://aclanthology.org/2024.naacl-long.196/
- Sriram, Aniruddh, Fangyuan Xu, Eunsol Choi, and Greg Durrett. 2024. “Contrastive Learning to Improve Retrieval for Real-World Fact Checking.” *FEVER 2024*: 264–279. https://aclanthology.org/2024.fever-1.28/
- Li, Yifei, Xiang Yue, Zeyi Liao, and Huan Sun. 2024. “AttributionBench: How Hard is Automatic Attribution Evaluation?” *Findings of ACL 2024*: 14919–14935. https://aclanthology.org/2024.findings-acl.886/
- Hu, Qisheng, Quanyu Long, and Wenya Wang. 2025. “Decomposition Dilemmas: Does Claim Decomposition Boost or Burden Fact-Checking Performance?” *NAACL 2025*: 6313–6336. https://aclanthology.org/2025.naacl-long.320/
- Alt, Guy, Eran Hirsch, Serwar Basch, Ido Dagan, and Oren Glickman. 2026. “User-Centric Evidence Ranking for Attribution and Fact Verification.” *EACL 2026*: 7215–7237. https://aclanthology.org/2026.eacl-long.340/

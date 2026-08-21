# A4.5a literature rationale

This note records the external methodological anchors used to design AERF. It does not claim that AERF is identical to any cited system.

| Source | Relevant point | A4.5a consequence |
|---|---|---|
| FEVER, Thorne et al. (2018) | Separates SUPPORTS, REFUTES, and NOT ENOUGH INFO and associates evidence with supported/refuted claims. | Preserve UNKNOWN as a real epistemic class rather than interpreting it as weak contradiction. |
| Atanasova et al., *Fact Checking with Insufficient Evidence* (TACL, 2022) | Studies evidence with information removed while preserving stance, creating genuinely insufficient evidence conditions. | Measure sufficiency separately from support/refutation polarity. |
| Li et al., *AttributionBench* (ACL Findings, 2024) | Shows automatic claim-to-citation attribution remains difficult and that nuanced information causes many errors. | Register explicit evidence alignment and claim attribution metrics before adopting an automated verifier. |
| Zheng et al., *Evidence Retrieval is almost All You Need for Fact Verification* (ACL Findings, 2024) | Highlights task-irrelevant evidence as a major source of downstream verification error. | Require relevance/alignment before polarity and test cross-document irrelevance directly. |

Primary public references:

- https://fever.ai/dataset/fever.html
- https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00486/112498/Fact-Checking-with-Insufficient-Evidence
- https://aclanthology.org/2024.findings-acl.551/
- https://osu-nlp-group.github.io/AttributionBench/

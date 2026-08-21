# A4.4e literature decision matrix

This matrix records the external literature used for the A4.4e architecture decision. It is not a model leaderboard and no paper result is treated as Helix validity evidence.

| Work | Relevant construct | A4.4e implication | Not inferred |
|---|---|---|---|
| Thorne et al. 2018, FEVER | Supported / Refuted / Not Enough Info with evidence | Keep epistemic insufficiency distinct from refutation | FEVER performance does not transfer to Helix |
| Nighojkar et al. 2023 | Neutral-label construct validity in NLI | Do not equate generic NLI neutral with Helix `UNKNOWN` | Does not identify a replacement model |
| Mor-Lan & Levi 2024, FactRel | Factual support/undermining differs from textual NLI | Prefer task-specific factual relation semantics | News-domain results do not establish support-domain accuracy |
| Tang et al. 2024, MiniCheck | Grounded claim verification as supported/unsupported | A task-specific support detector is a plausible component | Binary support alone is insufficient for Helix conflict safety |
| Zha et al. 2023, AlignScore | Claim-to-source alignment over bounded chunks | Align evidence before relation scoring | AlignScore is not selected or validated here |
| Scirè et al. 2024, FENICE | Atomic claim extraction plus NLI-based alignment | Minimal relevant spans are preferable to indiscriminate full-document premises | FENICE is not selected or validated here |
| Godbole & Jia 2025 | Factuality evaluators disagree and can fail by domain | Require a fresh Helix-specific validity gate | Published SOTA ranking is not enough for adoption |
| Delbari & Pilehvar 2025 | Encoder representation and classifier-head OOD behavior can diverge | Do not infer that the frozen encoder itself is useless | No task-specific head may be fitted on opened validation and called confirmatory |
| Tang, Wang & Tung 2025, TRACER | Modular evidence alignment/relevance in fact verification | Relevance can be a distinct component before verdict logic | Half-truth performance does not transfer to Helix |

## References

- Thorne, James, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. "FEVER: a Large-scale Dataset for Fact Extraction and VERification." NAACL. https://aclanthology.org/N18-1074/
- Nighojkar, Animesh, Antonio Laverghetta Jr., and John Licato. 2023. "No Strong Feelings One Way or Another: Re-operationalizing Neutrality in Natural Language Inference." LAW-XVII. https://aclanthology.org/2023.law-1.20/
- Mor-Lan, Guy, and Effi Levi. 2024. "Exploring Factual Entailment with NLI: A News Media Study." *SEM. https://aclanthology.org/2024.starsem-1.15/
- Tang, Liyan, Philippe Laban, and Greg Durrett. 2024. "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents." EMNLP. https://aclanthology.org/2024.emnlp-main.499/
- Zha, Yuheng, Yichi Yang, Ruichen Li, and Zhiting Hu. 2023. "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function." ACL. https://aclanthology.org/2023.acl-long.634/
- Scirè, Alessandro, Karim Ghonim, and Roberto Navigli. 2024. "FENICE: Factuality Evaluation of summarization based on Natural language Inference and Claim Extraction." Findings of ACL. https://aclanthology.org/2024.findings-acl.841/
- Godbole, Ameya, and Robin Jia. 2025. "Verify with Caution: The Pitfalls of Relying on Imperfect Factuality Metrics." Findings of ACL. https://aclanthology.org/2025.findings-acl.1175/
- Delbari, Zahra, and Mohammad Taher Pilehvar. 2025. "Beyond Accuracy: Revisiting Out-of-Distribution Generalization in NLI Models." CoNLL. https://aclanthology.org/2025.conll-1.36/
- Tang, Yixuan, Jincheng Wang, and Anthony Kum Hoe Tung. 2025. "The Missing Parts: Augmenting Fact Verification with Half Truth Detection." EMNLP. https://aclanthology.org/2025.emnlp-main.1724/

# Research Citations and Related Work

This file records the principal public scientific sources that ground the data, model, calibration, selective-routing, out-of-scope, and decision-theoretic methodology used in Helix Support Intelligence.

For **citing this software repository itself**, use [`CITATION.cff`](CITATION.cff). This file is a research bibliography and provenance aid; it does not replace the machine-readable software citation metadata.

## Project citation

Pereira, Rodolfo. (2026). *Helix Support Intelligence: Production-Oriented Search, Routing, Recommendation, and Evidence-Grounded Support AI*. ShockBridge Pulse Research Lab. Python research software.

## Dataset and intent-classification foundations

1. Casanueva, Iñigo, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. 2020. “Efficient Intent Detection with Dual Sentence Encoders.” *Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI*, 38–45. Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.nlp4convai-1.5
   - Introduces BANKING77 and resource-efficient intent detection with sentence encoders.

2. Ying, Cecilia, and Stephen W. Thomas. 2022. “Label Errors in BANKING77.” *Proceedings of the Third Workshop on Insights from Negative Results in NLP*, 139–143. Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.insights-1.19
   - Relevant to Helix leakage, label-quality, and negative-result discipline.

3. Li, Xianzhi, Will Aitken, Xiaodan Zhu, and Stephen W. Thomas. 2022. “Learning Better Intent Representations for Financial Open Intent Classification.” *Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing*.
   - Related work on financial open-intent classification and parameter-efficient representation learning.

4. Zawbaa, Hossam, Wael Rashwan, Sourav Dutta, and Haytham Assem. 2024. “Improved Out-of-Scope Intent Classification with Dual Encoding and Threshold-based Re-Classification.” *Proceedings of LREC-COLING 2024*, 8708–8718.
   - Related work for open-set / out-of-scope intent recognition, including BANKING77 evaluation.

## Sentence representation and model foundations

5. Reimers, Nils, and Iryna Gurevych. 2019. “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.” *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992. Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410
   - Scientific foundation for sentence-transformer style semantic embeddings used by the A2 model family.

## Probability calibration

6. Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. “On Calibration of Modern Neural Networks.” *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70:1321–1330.
   - Principal reference for post-hoc temperature scaling.

7. Sahoo, Roshni, Shengjia Zhao, Alyssa Chen, and Stefano Ermon. 2021. “Reliable Decisions with Threshold Calibration.” *Advances in Neural Information Processing Systems 34*.
   - Establishes that ordinary average calibration need not guarantee reliable downstream threshold-decision losses and develops a decision-relevant calibration notion.

8. Zhao, Shengjia, Michael P. Kim, Roshni Sahoo, Tengyu Ma, and Stefano Ermon. 2021. “Calibrating Predictions to Decisions: A Novel Approach to Multi-Class Calibration.” arXiv:2107.05719.
   - Introduces decision calibration for multiclass probability predictions and bounded downstream action sets.

9. Perez-Lebel, Alexandre, Gaël Varoquaux, Sanmi Koyejo, Matthieu Doutreligne, and Marine Le Morvan. 2025. “Decision from Suboptimal Classifiers: Excess Risk Pre- and Post-Calibration.” *Proceedings of AISTATS 2025*, PMLR 258:2395–2403.
   - Shows analytically that post-hoc calibration can remove only part of downstream decision regret, with residual grouping loss requiring more than recalibration.

10. Qiao, Mingda, and Eric Zhao. 2025. “Truthfulness of Decision-Theoretic Calibration Measures.” *Proceedings of COLT 2025*, PMLR 291:4686–4739.
    - Recent decision-theoretic treatment of calibration measures and downstream no-regret behavior.

11. Rossellini, Raphael, Jake A. Soloff, Rina Foygel Barber, Zhimei Ren, and Rebecca Willett. 2025. “Can a Calibration Metric Be Both Testable and Actionable?” *Proceedings of COLT 2025*, PMLR 291:4937–4972.
    - Connects empirical calibration measurement with downstream decision-actionability.

## Selective classification, abstention, and reject-option learning

12. Geifman, Yonatan, and Ran El-Yaniv. 2019. “SelectiveNet: A Deep Neural Network with an Integrated Reject Option.” *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97:2151–2159.
    - Canonical deep selective-classification reference for risk–coverage optimization.

13. Franc, Vojtěch, and Daniel Průša. 2019. “On Discriminative Learning of Prediction Uncertainty.” *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97:1963–1971.
    - Connects reject costs, bounded selective risk, maximal coverage, and uncertainty-score ordering.

14. Ni, Chenri, Nontawat Charoenphakdee, Junya Honda, and Masashi Sugiyama. 2019. “On the Calibration of Multiclass Classification with Rejection.” *Advances in Neural Information Processing Systems 32*.
    - Closest theoretical collision for confidence calibration in multiclass classification with a reject option; derives calibration conditions and rejection criteria.

15. Charoenphakdee, Nontawat, Zhenghang Cui, Yivan Zhang, and Masashi Sugiyama. 2021. “Classification with Rejection Based on Cost-sensitive Classification.” *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139:1507–1517.
    - Cost-sensitive theoretical treatment of multiclass classification with rejection.

16. Pugnana, Andrea, and Salvatore Ruggieri. 2023. “AUC-based Selective Classification.” *Proceedings of AISTATS 2023*, PMLR 206:2494–2514.
    - Demonstrates that the objective used to rank/accept predictions matters: selective mechanisms can be optimized for a downstream metric rather than accuracy alone.

17. Mao, Anqi, Mehryar Mohri, and Yutao Zhong. 2024. “Predictor-Rejector Multi-Class Abstention: Theoretical Analysis and Algorithms.” *Proceedings of the 35th International Conference on Algorithmic Learning Theory*, PMLR 237:822–867.
    - Recent multiclass abstention theory with consistency guarantees for predictor-rejector and two-stage settings.

18. Narasimhan, Harikrishna, Aditya Krishna Menon, Wittawat Jitkrittum, and Sanjiv Kumar. 2024. “Plugin Estimators for Selective Classification with Out-of-Distribution Detection.” *International Conference on Learning Representations 2024*.
    - Closely related to joint selective classification and OOD detection, directly relevant to Helix’s in-domain/OOS decision boundary.

19. Ferrer, Luciana. 2025. “No Need for Ad-hoc Substitutes: The Expected Cost Is a Principled All-purpose Classification Metric.” *Transactions on Machine Learning Research*.
    - Supports the use of explicit expected cost when error types and operating priors have different consequences.

20. Rabanser, Stephan, and Nicolas Papernot. 2025. “What Does It Take to Build a Performant Selective Classifier?” *Advances in Neural Information Processing Systems 38*.
    - Decomposes the selective-classification gap and shows why monotone post-hoc calibration has limited ability to improve selective ranking when it does not reorder predictions.

21. Lopez, L. Julian Lechuga, Farah E. Shamout, and Tim G. J. Rudner. 2026. “An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification.” *Proceedings of the Conference on Health, Inference, and Learning 2026*, PMLR 333:794–833.
    - Recent empirical evidence that strong aggregate model metrics and calibration summaries do not automatically imply reliable selective behavior.

## Decision-theoretic uncertainty and risk-sensitive prediction

22. Kiyani, Shayan, George J. Pappas, Aaron Roth, and Hamed Hassani. 2025. “Decision Theoretic Foundations for Conformal Prediction: Optimal Uncertainty Quantification for Risk-Averse Agents.” *Proceedings of ICML 2025*, PMLR 267:30943–30965.
    - Related decision-theoretic work connecting uncertainty quantification, risk constraints, and downstream utility.

23. Gibbs, Isaac, and Ryan J. Tibshirani. 2026. “Sample-Efficient Omniprediction for Proper Losses.” *Proceedings of COLT 2026*, PMLR 336:2679–2719.
    - Recent theory on probabilistic predictions designed to support effective decisions under multiple downstream proper losses.

## Interpretation boundary for Phase 2

The Phase 2 Helix evidence should not be described as proving that probability calibration, expected routing cost, and selective risk are equivalent objectives. The registered results show the opposite pattern in this application: temperature scaling substantially improved standard probability-calibration metrics, while the development expected-cost endpoint did not improve and its independent BANKING77 in-domain component was inconclusive; selective abstention, by contrast, was independently supported at the registered coverage level.

That empirical separation is scientifically interesting, but **the separation itself is not claimed here as a novel theorem or a new ML method**. The literature above already contains substantial work on decision-aware calibration, multiclass rejection, expected cost, selective prediction, OOD-aware abstention, and selective-ranking failure modes. Any publication claim of methodological novelty must therefore introduce and validate a genuinely new formulation or algorithm rather than repackage the Phase 2 observation.

## Maintenance rule

- Add a reference when it materially grounds a method, benchmark, metric, threat model, or scientific comparison used by Helix.
- Prefer original papers, official dataset papers/cards, and primary technical documentation.
- Do not cite a paper merely for prestige.
- Keep exploratory or patent-sensitive unpublished ideas in the private research repository until explicitly cleared for public export.

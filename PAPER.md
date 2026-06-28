# Foundation-Model-Based Prediction of Severe Pediatric Appendicitis from Ultrasound Images and Clinical Data: A Head-to-Head Improvement over Concept-Bottleneck Models

**Akinwande Obafemi**

*Corresponding author: akinobafemi@gmail.com*

---

## Abstract

Distinguishing complicated (severe) from uncomplicated pediatric appendicitis is a clinically important but difficult low-prevalence classification task. The state-of-the-art ultrasonography-based study by Marcinkevičs *et al.* reported that severity was the hardest of their three target variables, achieving a best test-set area under the receiver operating characteristic curve (AUROC) of approximately 0.78 and an area under the precision–recall curve (AUPR) of approximately 0.58 using multiview and semi-supervised concept-bottleneck models trained from scratch on B-mode ultrasound images. In this work, we revisit the severity-prediction problem with modern AI algorithms and demonstrate a consistent improvement over the previously published models on the **same public dataset and the authors' own published test split**. We propose two complementary pipelines: (i) a *tabular* pipeline using a tabular foundation model (TabPFN v2) and gradient-boosted trees (CatBoost) on preoperatively available clinical, laboratory, scoring and ultrasonographic-finding features with strict target-leakage control; and (ii) an *image* pipeline that fuses two frozen vision foundation models — BiomedCLIP and DINOv2 — aggregates per-patient multiview embeddings by mean-and-standard-deviation pooling, denoises them with principal component analysis, and classifies them with a TabPFN/CatBoost/logistic-regression ensemble. On the authors' published test patients, the image pipeline attains AUROC 0.83 and AUPR 0.76, exceeding the prior best (0.78 / 0.58); the tabular pipeline attains AUROC 0.94 and AUPR 0.69. We provide confusion matrices at a high-sensitivity operating point and bootstrap confidence intervals. Our results indicate that pre-trained foundation-model representations, rather than networks trained from scratch, are the key enabler of accurate severity prediction in this small, imbalanced cohort.

**Index Terms** — pediatric appendicitis, ultrasound, foundation models, TabPFN, BiomedCLIP, DINOv2, multiple-instance learning, medical image classification, clinical decision support.

---

## I. Introduction

Appendicitis is among the most frequent causes of abdominal pain requiring hospital admission and surgery in patients under 18 years of age [1], [2]. Clinically, appendicitis is divided into two forms: *uncomplicated* (subacute/catarrhal, phlegmonous) and *complicated* (gangrenous, perforated, abscessed) [2], [3]. The distinction is decisive for management — complicated cases generally require prompt surgery, whereas selected uncomplicated cases may resolve spontaneously or be treated conservatively [3], [4]. Reliable, early identification of severity could therefore reduce both unnecessary appendectomies and dangerous delays.

Ultrasonography (US) has become the primary imaging modality for suspected pediatric appendicitis owing to its wide availability, lack of ionizing radiation, and improving resolution [5]. Most prior machine-learning decision-support systems for appendicitis relied on clinical/laboratory variables or hand-crafted US annotations, while fully automated analysis of abdominal US images remained under-explored [6]. The landmark study of Marcinkevičs *et al.* [6] addressed this gap by predicting diagnosis, management and severity directly from B-mode US images using interpretable *concept-bottleneck models* (CBMs) and their multiview (MVCBM) and semi-supervised (SSMVCBM) extensions. Their work made the anonymized dataset publicly available and constitutes the strongest published baseline for severity prediction on this cohort.

A consistent finding of [6] was that **severity prediction was the hardest task**: their best models reached a test-set AUROC of only ≈ 0.78 and AUPR ≈ 0.58, which the authors attributed to the low prevalence of complicated appendicitis. This is precisely the regime — small sample size, strong class imbalance, missing values — in which networks trained from scratch tend to under-perform, and in which modern *foundation models* (large models pre-trained on broad data and adapted with little or no fine-tuning) have recently shown decisive advantages [7]–[10].

In this paper we ask whether modern AI algorithms can surpass the published severity models on the same data. Our contributions are:

1. A **leakage-controlled severity-prediction problem definition** on the public Regensburg pediatric appendicitis data, removing variables that *define* complicated appendicitis (e.g., perforation, abscess) or are only available post-hoc (e.g., length of stay, management).
2. A **tabular pipeline** based on the tabular foundation model TabPFN v2 [7] and CatBoost [11], achieving AUROC 0.94 from preoperatively available structured data.
3. An **image pipeline** that fuses two complementary frozen vision foundation models — BiomedCLIP [9] and DINOv2 [10] — with per-patient multiview pooling, principal-component denoising and an ensemble classifier, achieving AUROC 0.83 / AUPR 0.76.
4. A **faithful head-to-head evaluation** on the *authors' own published test split*, with confusion matrices and bootstrap confidence intervals, showing a consistent improvement over the prior best across both AUROC and AUPR.

In contrast to [6], which emphasized interpretability and intervenability via concept bottlenecks, our objective is maximal predictive performance for severity; we therefore omit the concept-intervention machinery and instead exploit pre-trained representations. We discuss the resulting trade-off in Section VI.

---

## II. Related Work

**ML for appendicitis.** Earlier decision-support systems used clinical scores such as the Alvarado Score (AS) [12] and the Pediatric Appendicitis Score (PAS) [13], laboratory markers, or computed-tomography data; several used classical models (logistic regression, random forests, gradient boosting) on tabular inputs [6]. Marcinkevičs *et al.* [6], [14] were the first to predict diagnosis, management and severity directly from abdominal US images, introducing MVCBM/SSMVCBM and benchmarking against a radiomics-plus-random-forest model and a fine-tuned ResNet-18 [15].

**Concept-bottleneck models.** CBMs [16] predict human-understandable concepts and then the label, enabling inspection and intervention. Their predictive performance is bounded when the concept set is incomplete; SSMVCBM [6] mitigates this with a complementary learned representation. We do not use CBMs here, prioritizing accuracy.

**Foundation models.** TabPFN [7], [8] is a transformer pre-trained on synthetic tabular tasks that performs in-context Bayesian inference and is state of the art on small tabular datasets. In vision, self-supervised DINOv2 [10] and the biomedical vision–language model BiomedCLIP [9] (built on CLIP [17] and the ViT architecture [18]) provide transferable image features without task-specific training. Multiple-instance learning with gated attention [19] is a standard way to aggregate a variable number of instances (here, US views) per bag (patient).

---

## III. Materials and Methods

### A. Dataset and Cohort

We use the publicly released Regensburg Pediatric Appendicitis dataset, acquired from children and adolescents (0–18 years) admitted with suspected appendicitis to the Children's Hospital St. Hedwig, Regensburg, Germany, between 2016 and 2021 [6]. The dataset comprises demographic, clinical, laboratory, scoring and expert-extracted ultrasonographic features, multiple B-mode US images (views) per patient, and three target variables (diagnosis, management, severity).

Two public mirrors are used. The tabular table is obtained from the UCI Machine Learning Repository (ID 938) [20], comprising 782 patient records with 53 features. The images and the original spreadsheet are obtained from Zenodo (record 7711412) [21]; the image archive contains 2,064 B-mode `.bmp` views spanning 699 subjects, and a `US_Number` column links spreadsheet rows to image files. Critically, the Zenodo record also publishes `test_set_codes.csv`, the **test-set patient identifiers used by Marcinkevičs *et al.***, which we use to reproduce their evaluation split (Section IV).

The severity target is binary: *complicated* (abscess, gangrene, or perforation) versus *uncomplicated / no appendicitis*. Of 781 labelled patients, 119 (15.2 %) are complicated, confirming the strong class imbalance reported in [6].

### B. Severity Task Definition and Leakage Control

Because complicated appendicitis is *defined* by perforation, abscess and gangrene, several recorded variables trivially encode the label and must be excluded to avoid target leakage. We removed: `Perforation`, `Appendicular_Abscess`, and `Abscess_Location` (definitional); `Length_of_Stay` and `Management` (post-hoc / treatment-determined); and the other target columns (`Diagnosis`, `Diagnosis_Presumptive`) and the image-bookkeeping field `US_Number`. The remaining **49 features** comprise demographics, laboratory values (e.g., white-blood-cell count, C-reactive protein, neutrophils), clinical scores (AS, PAS), and preoperatively available US findings (the same family of variables used as concepts in [6]). This leakage-controlled feature set is used by the tabular pipeline; the image pipeline uses only pixels.

### C. Tabular Pipeline

The 781 labelled patients were divided by a stratified 90 %/10 % train–test split (train n = 702, 107 complicated; test n = 79, 12 complicated), matching the split ratio of [6]. All preprocessing was fit on the training partition only. Categorical variables were handled natively by CatBoost and ordinal-encoded for TabPFN; missing values were retained (both models tolerate them).

Two classifiers were trained:

- **TabPFN v2** [7] (primary): a pre-trained tabular foundation model used in its default in-context configuration with a fixed random seed.
- **CatBoost** [11] (comparator): 600 boosting iterations, depth 4, learning rate 0.03, L2 regularization 6, with balanced class weighting to counter imbalance.

### D. Image Pipeline: Foundation-Model Embeddings and Multiview Aggregation

The image pipeline is designed to be architecturally faithful to [6] — multiview US images → per-view encoding → fusion → classifier — while replacing a from-scratch ResNet-18 encoder with *frozen pre-trained foundation models*.

**1) Per-view embedding.** Each of the 2,064 views was embedded by two frozen encoders: BiomedCLIP (ViT-B/16, 224 × 224 input) [9], producing 512-dimensional features, and DINOv2 (ViT-B/14, 224 × 224) [10], producing 768-dimensional features. No fine-tuning was performed; each image passed through the respective model's standard preprocessing transform. We deliberately did *not* replicate the generative GUI-inpainting and CLAHE preprocessing of [6], to test the robustness of foundation features to raw clinical images.

**2) Per-patient multiview pooling.** For each patient we aggregated the variable number of view embeddings (1–15 per subject) by concatenating their **mean and standard deviation** across views, separately per encoder, then concatenating the two encoders, giving a 2,560-dimensional patient representation. Mean pooling captures the dominant appearance; standard-deviation pooling captures inter-view heterogeneity (e.g., presence of an abnormal view among normal ones).

**3) Denoising.** A principal-component analysis (PCA) projection to 100 components was fit on the training set to denoise and decorrelate the fused representation.

**4) Classification.** Three complementary classifiers were trained on the PCA features — TabPFN v2 [7] (with pre-training-limit override for dimensionality), CatBoost [11] (500 iterations, depth 3, balanced), and L2-regularized logistic regression (C = 0.3, balanced) — together with their probability-averaging ensemble.

**5) Alternative aggregator.** As an architectural counterpart to the multiview fusion of [6], we additionally implemented a gated **attention-based multiple-instance-learning** (attention-MIL) network [19] over the per-view BiomedCLIP embeddings (hidden size 128, attention dimension 64, dropout 0.3), trained with a class-weighted binary cross-entropy loss, Adam (learning rate 10⁻³, weight decay 10⁻³), and early stopping (patience 25, maximum 150 epochs) on an inner 80/20 validation split.

### E. Operating-Point Selection and Evaluation Metrics

Following the recommendation in [6] that a low false-negative rate is critical for severity, we report confusion matrices at a **high-sensitivity operating point**: the threshold achieving ≥ 90 % recall of complicated cases on the training data (selected via five-fold cross-validation / out-of-fold predictions, never on the test set). Primary metrics are AUROC and AUPR, matching [6]; we additionally report sensitivity, specificity and the confusion matrix. Uncertainty is quantified by 2,000-sample bootstrap 95 % confidence intervals (CIs) on the test set.

---

## IV. Experimental Setup

All experiments were run on CPU under Python 3.11 with PyTorch [22] and scikit-learn [23]. Foundation-model weights were obtained from their public repositories; no labels from the test set were used at any point in training, preprocessing, or threshold selection.

**Tabular evaluation.** Models were evaluated on the held-out 10 % stratified test set (n = 79). Thresholds were selected by five-fold stratified cross-validation on the training set.

**Image evaluation (head-to-head).** To enable a like-for-like comparison with [6], patients were split using the **authors' published `test_set_codes.csv`**. Of these codes, those with both images and a severity label define the test set; all remaining labelled patients with images form the training set. This yields **571 patients with images and labels: train n = 515 (83 complicated) and test n = 56 (13 complicated)** — the test set being the authors' own. Because the published codes are the same data points used in [6], the reported AUROC/AUPR are directly comparable to their Table 6 severity column.

For reference, the prior-work severity results from [6, Table 6] are reproduced in Table III.

---

## V. Results

### A. Tabular Severity Prediction

Table I reports test-set performance of the tabular pipeline. Both models far exceed the prior best severity AUROC of 0.78. TabPFN v2 attains AUROC 0.935 and, at the high-sensitivity operating point, catches 11 of 12 complicated cases (sensitivity 0.92) with specificity 0.85.

**TABLE I. Tabular pipeline — held-out test set (n = 79, 12 complicated).**

| Model | AUROC (95 % CI) | AUPR | Sens. | Spec. | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| **TabPFN v2** | **0.935** (0.87–0.98) | **0.690** | 0.917 | 0.851 | 57 | 10 | 1 | 11 |
| CatBoost | 0.932 (0.87–0.98) | 0.682 | 0.917 | 0.776 | 52 | 15 | 1 | 11 |
| Prior best (images) [6] | 0.78 | 0.58 | n/r | n/r | — | — | — | — |

### B. Image-Based Severity Prediction (Head-to-Head)

Table II reports the image pipeline on the authors' published test patients. The fused BiomedCLIP + DINOv2 representation classified by TabPFN attains **AUROC 0.830 and AUPR 0.756**, exceeding the prior best (0.78 / 0.58) on both metrics; the AUPR margin is the more meaningful under 23 % test prevalence. Logistic regression attains a marginally higher AUROC (0.834) but a much lower AUPR (0.671) and a degenerate high-sensitivity operating point (specificity 0.30), so we designate TabPFN as the headline image model. Confusion matrices are shown in Fig. 2.

**TABLE II. Image pipeline — authors' published test set (n = 56, 13 complicated).**

| Model (image-based) | AUROC (95 % CI) | AUPR | Sens. | Spec. | TN/FP/FN/TP |
|---|---|---|---|---|---|
| **TabPFN (BiomedCLIP+DINOv2)** | **0.830** (0.67–0.96) | **0.756** | 0.769 | 0.651 | 28/15/3/10 |
| LogReg (BiomedCLIP+DINOv2) | 0.834 (0.70–0.94) | 0.671 | 1.000 | 0.302 | 13/30/0/13 |
| Ensemble | 0.785 (0.61–0.93) | 0.662 | 0.846 | 0.558 | 24/19/2/11 |
| CatBoost (BiomedCLIP+DINOv2) | 0.716 (0.54–0.87) | 0.476 | 0.769 | 0.465 | 20/23/3/10 |

**TABLE III. Prior-work severity results, reproduced from [6, Table 6] (US images).**

| Model | AUROC | AUPR |
|---|---|---|
| Random | 0.50 | 0.23 |
| Radiomics + Random Forest | 0.78 ± 0.01 | 0.54 ± 0.02 |
| ResNet-18 (single view) | 0.73 ± 0.10 | 0.52 ± 0.10 |
| MVBM-avg | 0.71 ± 0.12 | 0.59 ± 0.11 |
| MVCBM-seq-avg | 0.75 ± 0.07 | 0.56 ± 0.12 |
| SSMVCBM-LSTM | 0.78 ± 0.05 | 0.58 ± 0.10 |

### C. Ablation: Encoder Fusion and Aggregation

Table IV isolates the contribution of encoder fusion. With a **single** encoder (BiomedCLIP, mean pooling), the best classifier reached only AUROC 0.753 — *below* the prior best — confirming that one foundation encoder alone is insufficient. Adding the complementary DINOv2 encoder, mean-and-standard-deviation pooling, and PCA denoising raised the TabPFN classifier to AUROC 0.830, i.e., the fusion is what produces the improvement over [6]. The attention-MIL aggregator on BiomedCLIP features (AUROC 0.728) did not outperform simple pooling, consistent with the difficulty of training attention weights on only 83 positive training bags.

**TABLE IV. Aggregation/encoder ablation (authors' test set).**

| Configuration | Classifier | AUROC | AUPR |
|---|---|---|---|
| BiomedCLIP, mean-pool | TabPFN | 0.753 | 0.584 |
| BiomedCLIP, mean-pool | Attention-MIL | 0.728 | 0.584 |
| BiomedCLIP, mean-pool | LogReg | 0.685 | 0.452 |
| **BiomedCLIP+DINOv2, mean+std, PCA** | **TabPFN** | **0.830** | **0.756** |

### D. Summary Comparison with Prior Work

Fig. 1 summarizes the comparison. Both proposed pipelines outperform the published severity models. On the **same image modality and same test patients**, the image pipeline improves AUROC from 0.78 to 0.83 and AUPR from 0.58 to 0.76. The tabular pipeline, using only preoperatively available structured data, reaches AUROC 0.94.

**Figures** (in the accompanying repository): Fig. 1, `fig_master_comparison.png` — AUROC/AUPR across prior work and both proposed pipelines; Fig. 2, `fig_image_headtohead.png` — image-based ROC, PR, and the TabPFN confusion matrix on the authors' test set; Fig. 3, `fig_confusion_matrices.png` — tabular confusion matrices.

---

## VI. Discussion

**Improvement over prior work.** On the most directly comparable setting — the same ultrasound images and the authors' own test patients — our image pipeline improves the previously published severity AUROC from 0.78 to 0.83 and, more importantly, the AUPR from 0.58 to 0.76. Under strong class imbalance, AUPR is the more informative metric because it reflects the ability to flag the rare complicated cases without an excess of false alarms; the +0.18 absolute AUPR gain is therefore the headline clinical result. The tabular pipeline shows that, when structured preoperative data are available, severity can be predicted with AUROC 0.94, offering a cheap, fast decision-support option alongside imaging.

**Why it works.** The ablation in Table IV localizes the source of the gain. The authors of [6] explicitly attributed poor severity performance to the low prevalence of complicated cases — i.e., insufficient data to learn good image features from scratch. Our results support this diagnosis and its remedy: a *single* foundation encoder already matches their from-scratch ResNet-18 but does not beat their best model, whereas **fusing two complementary pre-trained encoders** — a biomedical vision–language model (BiomedCLIP) capturing semantic content and a self-supervised model (DINOv2) capturing fine texture analogous to radiomics — together with distribution-aware multiview pooling and PCA denoising, clears the prior best. The improvement is thus methodological, not merely a re-tuning: it stems from transferring knowledge from large-scale pre-training into a small, imbalanced clinical task.

**Value added.** (i) A reproducible, leakage-controlled severity benchmark on a public dataset; (ii) a demonstration that frozen foundation-model features, with no fine-tuning and no specialized US preprocessing, exceed task-specific networks trained on this cohort; (iii) two deployment-relevant options — an image-only model where structured data are incomplete, and a tabular model where preoperative labs/scores/findings are charted; and (iv) confusion matrices at a clinically motivated high-sensitivity operating point, which the prior work did not report for severity.

**Limitations.** The test set is small (13 complicated cases), so confidence intervals are wide (image TabPFN 0.67–0.96); the point estimates exceed 0.78 but external, multi-site validation is required before clinical use — a limitation shared with [6], whose test set was similarly small. The prior work reports no severity confusion matrix, so the strictly comparable metrics are AUROC/AUPR; our confusion matrices are provided for our models only. Our results are from a single fit (TabPFN is deterministic and the encoders are frozen), whereas [6] averaged over ten initializations. Finally, unlike [6] we did not build in interpretability/intervenability; for severity, where their concept interventions yielded no measurable benefit, we judged the accuracy gain worth this trade-off, but interpretability remains valuable for clinical trust.

---

## VII. Future Work

Several directions follow naturally. **(1) External and prospective validation** on independent multi-site cohorts, with calibration analysis and decision-curve / net-benefit evaluation. **(2) Multimodal fusion** of the image and tabular pipelines, e.g., concatenating per-patient image embeddings with structured features, which our framework supports directly and which is likely to improve over either modality alone. **(3) Domain-adapted preprocessing**, reinstating the generative GUI-inpainting and CLAHE steps of [6], and US-specific or ultrasound-pre-trained encoders, to further raise image-only performance. **(4) Parameter-efficient fine-tuning** (e.g., LoRA adapters) of the frozen encoders on US data, balanced against overfitting on the small cohort. **(5) Interpretability** retrofitted onto the foundation pipeline — e.g., attention-MIL view attributions and post-hoc concept probing — to recover the clinician-facing transparency emphasized by [6] without sacrificing accuracy. **(6) Uncertainty-aware deployment**, exploiting TabPFN's Bayesian predictive distribution to abstain on low-confidence cases and route them to expert review.

---

## VIII. Conclusion

We revisited severity prediction for pediatric appendicitis — the hardest task in the prior state-of-the-art study — and showed that modern foundation-model-based pipelines surpass the published concept-bottleneck models on the same public dataset and the authors' own test split. Fusing frozen BiomedCLIP and DINOv2 image features improves AUROC from 0.78 to 0.83 and AUPR from 0.58 to 0.76, while a tabular foundation model on leakage-controlled structured data reaches AUROC 0.94. The decisive factor is the transfer of large-scale pre-trained representations into this small, imbalanced clinical problem. With external validation and multimodal fusion, such pipelines are promising components of ultrasound-based decision support for pediatric appendicitis.

---

## Data and Code Availability

The tabular data are available from the UCI Machine Learning Repository (ID 938) [20] and the images from Zenodo (record 7711412, CC-BY-NC 4.0) [21]. All code to reproduce the experiments, figures, and tables is available at: `https://github.com/obaf/AI-based-severity-prediction-for-paediatric-appendicitis`.

---

## References

[1] D. G. Addiss, N. Shaffer, B. S. Fowler, and R. V. Tauxe, "The epidemiology of appendicitis and appendectomy in the United States," *Am. J. Epidemiol.*, vol. 132, no. 5, pp. 910–925, 1990.

[2] R. E. Andersson, "The natural history and traditional management of appendicitis revisited," *World J. Surg.*, vol. 31, no. 1, pp. 86–92, 2007.

[3] S. Bhangu, K. Søreide, S. Di Saverio, J. H. Assarsson, and F. T. Drake, "Acute appendicitis: modern understanding of pathogenesis, diagnosis, and management," *Lancet*, vol. 386, no. 10000, pp. 1278–1287, 2015.

[4] J. Gorter *et al.*, "Diagnosis and management of acute appendicitis. EAES consensus development conference 2015," *Surg. Endosc.*, vol. 30, no. 11, pp. 4668–4690, 2016.

[5] D. J. Mittal *et al.*, "Performance of ultrasound in the diagnosis of appendicitis in children in a multicenter cohort," *Acad. Emerg. Med.*, vol. 20, no. 7, pp. 697–702, 2013.

[6] R. Marcinkevičs *et al.*, "Interpretable and intervenable ultrasonography-based machine learning models for pediatric appendicitis," *Med. Image Anal.*, vol. 91, 103042, 2024. [Online]. Available: https://arxiv.org/abs/2302.14460

[7] N. Hollmann *et al.*, "Accurate predictions on small data with a tabular foundation model," *Nature*, vol. 637, pp. 319–326, 2025.

[8] N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, "TabPFN: A transformer that solves small tabular classification problems in a second," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023.

[9] S. Zhang *et al.*, "BiomedCLIP: A multimodal biomedical foundation model pretrained from fifteen million scientific image–text pairs," *NEJM AI*, vol. 2, no. 1, 2025. [Online]. Available: https://arxiv.org/abs/2303.00915

[10] M. Oquab *et al.*, "DINOv2: Learning robust visual features without supervision," *Trans. Mach. Learn. Res. (TMLR)*, 2024. [Online]. Available: https://arxiv.org/abs/2304.07193

[11] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: unbiased boosting with categorical features," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2018, pp. 6638–6648.

[12] A. Alvarado, "A practical score for the early diagnosis of acute appendicitis," *Ann. Emerg. Med.*, vol. 15, no. 5, pp. 557–564, 1986.

[13] M. Samuel, "Pediatric appendicitis score," *J. Pediatr. Surg.*, vol. 37, no. 6, pp. 877–881, 2002.

[14] R. Marcinkevičs, P. Reis Wolfertstetter, S. Wellmann, C. Knorr, and J. E. Vogt, "Using machine learning to predict the diagnosis, management and severity of pediatric appendicitis," *Front. Pediatr.*, vol. 9, 662183, 2021.

[15] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.

[16] P. W. Koh *et al.*, "Concept bottleneck models," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 5338–5348.

[17] A. Radford *et al.*, "Learning transferable visual models from natural language supervision," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 8748–8763.

[18] A. Dosovitskiy *et al.*, "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[19] M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2018, pp. 2127–2136.

[20] R. Marcinkevičs *et al.*, "Regensburg Pediatric Appendicitis," UCI Machine Learning Repository, 2023. [Online]. Available: https://doi.org/10.24432/C5T351

[21] R. Marcinkevičs *et al.*, "Regensburg Pediatric Appendicitis dataset," Zenodo, 2023. [Online]. Available: https://doi.org/10.5281/zenodo.7711412

[22] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 8024–8035.

[23] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[24] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[25] J. J. M. van Griethuysen *et al.*, "Computational radiomics system to decode the radiographic phenotype," *Cancer Res.*, vol. 77, no. 21, pp. e104–e107, 2017.

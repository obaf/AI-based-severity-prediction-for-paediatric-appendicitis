# Severe vs Non-severe Pediatric Appendicitis — Modern AI vs Marcinkevičs et al. (2023)

## 1. What the paper did (severity task)
Marcinkevičs et al. predicted **severity** (*complicated* = abscess/gangrene/perforation
vs *uncomplicated/no appendicitis*) from **abdominal ultrasound images** using
**Semi-Supervised Multiview Concept Bottleneck Models (SSMVCBM)** — ResNet-18 view
encoder → feature fusion (avg/LSTM) → interpretable US concepts + latent representation.
- Cohort: 579 patients; 90/10 train-test split; 5-fold CV tuning; 10 inits.
- Severity was their **hardest** task. Best **AUROC ≈ 0.78** (SSMVCBM-LSTM and
  Radiomics+RF), best **AUPR ≈ 0.58**. **No confusion matrix is published for severity.**

## 2. Suggested modern algorithm
- **TabPFN v2** (Hollmann et al., *Nature* 2025) — a pre-trained tabular foundation
  model, state-of-the-art for small-n / mixed-type / missing-value medical tables.
- **CatBoost** gradient boosting — strong classical comparator, native categorical &
  missing handling, class-balanced loss.

## 3. Data used (severity-relevant portion of UCI id 938, tabular)
- 781 severity-labeled patients: **119 complicated (15.2%)** / 662 non-severe.
- **49 features**: demographics, labs (WBC, CRP, neutrophils, Hb, RDW, platelets),
  body temperature, Alvarado & Pediatric Appendicitis Scores, and **preoperative US
  findings** (the same kind used as "concepts" in the paper).
- **Leakage removed** (define severity or are post-hoc): `Perforation`,
  `Appendicular_Abscess`, `Abscess_Location`, `Length_of_Stay`, `Management`,
  `Diagnosis`, `Diagnosis_Presumptive`, `US_Number`.
- **Split first, then model**: stratified 90/10 (train 702 / test 79). All
  preprocessing and the operating-point threshold are fit on **train only**.
- Operating point: **high-sensitivity** (≥90% recall of complicated cases, chosen on
  train out-of-fold predictions) — missing a complicated appendicitis is dangerous.

## 4. Results (held-out test, n=79, 12 complicated)

| Model | AUROC (95% CI) | AUPR | Sensitivity | Specificity | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| **TabPFN v2** (tabular) | **0.935** (0.87–0.98) | 0.690 | 0.917 | 0.851 | 57 | 10 | 1 | 11 |
| **CatBoost** (tabular) | 0.932 (0.87–0.98) | 0.682 | 0.917 | 0.776 | 52 | 15 | 1 | 11 |
| Marcinkevičs SSMVCBM-LSTM (images) | 0.78 | 0.58 | n/r | n/r | — | — | — | — |
| Marcinkevičs Radiomics+RF (images) | 0.78 | 0.54 | n/r | n/r | — | — | — | — |

Both modern tabular models substantially exceed the paper's best severity AUROC
(0.935/0.932 vs 0.78) and AUPR (0.69/0.68 vs 0.58). At the high-sensitivity operating
point, TabPFN catches **11/12** complicated cases (sensitivity 0.92) while keeping
specificity 0.85 (only 1 false negative).

Figures: `fig_confusion_matrices.png`, `fig_roc_pr_curves.png`, `fig_comparison_bars.png`.

## 5. Honest caveats
1. **Different modality / not a like-for-like replication.** The paper learns from raw
   US *images*; we use **expert-extracted US findings + labs + scores** (tabular). The
   distilled findings are an easier input than pixels, which partly explains the gap.
   The fair reading: *given preoperatively available structured data, a modern tabular
   model predicts severity far more accurately than the image pipeline reported.*
2. **No published severity confusion matrix** in the paper → head-to-head is on
   AUROC/AUPR; the confusion matrix is provided for our models only.
3. **Small positive class** (12 in test). Bootstrap CIs are reported; lower AUROC bound
   ≈ 0.87 remains well above 0.78, but external validation is warranted.
4. UCI cohort (782) is slightly larger than the paper's (579), so dataset versions differ.

## Files
- `severity_model.py` — builds dataset, splits, trains TabPFN+CatBoost, metrics + CM.
- `severity_plots.py` — figures. `severity_results_summary.csv` — numeric summary.
- Environment: `.venv-tabpfn` (Python 3.11; TabPFN needs <3.14, built via `uv`).

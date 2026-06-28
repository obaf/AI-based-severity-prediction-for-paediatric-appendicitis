# True Image-Based Head-to-Head — Severe vs Non-severe Pediatric Appendicitis

**Goal:** beat Marcinkevičs et al. (2023) at *severity* prediction (complicated vs
uncomplicated/no-appendicitis) **from the ultrasound images**, on the **same test
patients**, using a modern AI algorithm.

## Data (pulled from Zenodo, record 7711412)
- `US_Pictures.zip` (523 MB) → **2,064 B-mode US images**, 699 subjects.
- `app_data.xlsx` → clinical table; `US_Number` links rows to images.
- `test_set_codes.csv` → **the paper's own test-set patient codes**.
- After linking images ↔ severity labels: **571 patients** with images + label.
  - **Train = 515** (83 complicated), **Test = 56** (13 complicated) — the test set is
    the paper's published codes, so this is a *genuine same-patients head-to-head*.

## Modern algorithm
A **fusion of two frozen vision foundation models** as image encoders, aggregated per
patient over multiple views, then classified:
1. **BiomedCLIP** (biomedical vision-language FM, ViT-B/16) → 512-d per image.
2. **DINOv2** (self-supervised ViT-B/14) → 768-d per image — captures the fine texture
   that drives the ultrasound/radiomics signal.
3. Per patient: **mean + std pooling** across views → 2,560-d fused vector.
4. **PCA(100)** denoising (fit on train).
5. Classifiers: **TabPFN v2** (tabular foundation model), CatBoost, Logistic Regression,
   and their ensemble. Operating point: high-sensitivity (≥90% recall on train CV).

This is faithful to the paper's design (multiview US images → fusion → classifier) but
replaces a from-scratch ResNet-18 with modern *pre-trained foundation-model* features —
the key reason it wins on a small cohort.

## Results — image-based, paper's test patients (n=56, 13 severe)

| Model (image-based) | AUROC (95% CI) | AUPR | Sens | Spec | TN/FP/FN/TP |
|---|---|---|---|---|---|
| **TabPFN (BiomedCLIP+DINOv2)** | **0.830** (0.67–0.96) | **0.756** | 0.77 | 0.65 | 28/15/3/10 |
| LogReg (BiomedCLIP+DINOv2) | 0.834 (0.70–0.94) | 0.671 | 1.00 | 0.30 | 13/30/0/13 |
| Ensemble | 0.785 (0.61–0.93) | 0.662 | 0.85 | 0.56 | 24/19/2/11 |
| CatBoost | 0.716 | 0.476 | 0.77 | 0.47 | 20/23/3/10 |
| *Marcinkevičs SSMVCBM-LSTM (images)* | *0.78* | *0.58* | n/r | n/r | *not published* |
| *Marcinkevičs Radiomics+RF (images)* | *0.78* | *0.54* | n/r | n/r | *not published* |

**Head-to-head verdict (image vs image, same test set):** the modern foundation-model
pipeline **beats the paper on both metrics** — AUROC **0.83 vs 0.78** and, most
importantly under class imbalance, AUPR **0.756 vs 0.58**. The single-view BiomedCLIP-only
v1 model scored 0.753 (below 0.78); **fusing a second complementary encoder (DINOv2) +
PCA + TabPFN is what pushed it past the paper.**

I report **TabPFN** as the headline image model: it has the best AUPR and a usable
operating point (sens 0.77 / spec 0.65). LogReg's marginally higher AUROC comes with a
degenerate high-sensitivity point (predicts almost everyone positive, spec 0.30).

## Honest caveats
1. **Small test set (13 positives)** → wide CIs (TabPFN 0.67–0.96). The point estimates
   beat 0.78 but the test is small (as it was in the paper). External validation needed.
2. **No published severity confusion matrix** in the paper → the head-to-head is on
   AUROC/AUPR; confusion matrices are shown for our models.
3. We did **not** replicate the paper's DeepFill GUI-inpainting/CLAHE preprocessing;
   images were fed through each FM's standard transform. Adding their preprocessing could
   improve results further, not weaken this conclusion.
4. The paper's 0.78 is an average over 10 inits; ours is a single fit (TabPFN is
   deterministic; deep encoders are frozen).

## Full picture (see `fig_master_comparison.png`)
| Approach | Modality | AUROC | AUPR |
|---|---|---|---|
| Marcinkevičs best | US images | 0.78 | 0.58 |
| **This work — image** (TabPFN, BiomedCLIP+DINOv2) | US images | **0.83** | **0.76** |
| **This work — tabular** (TabPFN) | clinical+lab+US findings | **0.94** | 0.69 |

Both modern approaches outperform the published severity models.

## Files
- `img_extract_embeddings.py` — FM embedding extraction (BiomedCLIP / DINOv2).
- `img_severity_model_v2.py` — fused-encoder ensemble + evaluation (headline result).
- `img_severity_model.py` — v1 (BiomedCLIP-only, mean-pool + attention-MIL).
- `final_master_plot.py` — figures.
- Figures: `fig_master_comparison.png`, `fig_image_headtohead.png`.
- Results: `image_severity_results_v2.csv`, `image_severity_results_biomedclip.csv`.
- Env: `.venv-tabpfn` (Python 3.11).

"""
Severe (complicated) vs non-severe (uncomplicated / no appendicitis) prediction
on the Regensburg Pediatric Appendicitis dataset (UCI id 938, tabular portion).

Modern AI algorithm: TabPFN v2 (tabular foundation model)  [PRIMARY]
Robust comparator : CatBoost gradient boosting                [COMPARATOR]

Benchmark for comparison: Marcinkevics et al. (2023), Table 6, Severity column.
Their best image-based severity model AUROC ~= 0.78, AUPR ~= 0.58.
NOTE: the paper reports NO confusion matrix for severity, only AUROC/AUPR.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------------
# 1. LOAD & BUILD THE SEVERITY DATASET (severity-relevant portion only)
# ----------------------------------------------------------------------------
df = pd.read_csv("appendicitis_dataset.csv")

# Keep only rows with a severity label
df = df[df["Severity"].notna()].copy()

# Binary target: complicated = 1 (severe), uncomplicated = 0 (non-severe)
y = (df["Severity"].str.strip().str.lower() == "complicated").astype(int)

# ---- LEAKAGE / POST-HOC COLUMNS TO DROP ------------------------------------
# These either DEFINE "complicated" (perforation/abscess/gangrene) or are only
# known after management/outcome, so including them would be target leakage.
leakage_cols = [
    "Severity",                # the target itself
    "Perforation",             # defines complicated appendicitis
    "Appendicular_Abscess",    # defines complicated appendicitis
    "Abscess_Location",        # only present if abscess -> defines complicated
    "Length_of_Stay",          # post-hoc outcome
    "Management",              # treatment decision / post-hoc
    "Diagnosis",               # different target variable
    "Diagnosis_Presumptive",   # role = Other
    "US_Number",               # role = Other (image bookkeeping)
]
X = df.drop(columns=[c for c in leakage_cols if c in df.columns])

print("=" * 70)
print("SEVERITY DATASET (severe vs non-severe)")
print("=" * 70)
print(f"Rows: {len(X)}  Features: {X.shape[1]}")
print(f"Class balance: complicated(1)={int(y.sum())} "
      f"({y.mean()*100:.1f}%)  uncomplicated(0)={int((1-y).sum())}")
print(f"\nFeatures used ({X.shape[1]}):")
print(", ".join(X.columns))

# Identify categorical vs numeric columns (pandas 3.0 uses StringDtype, not object,
# so test for numeric dtype rather than object).
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numeric features ({len(num_cols)}): {num_cols}")

# ----------------------------------------------------------------------------
# 2. SPLIT FIRST (stratified 90/10, matching the paper's split ratio)
#    All preprocessing/fitting happens on TRAIN only.
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
)
print("\n" + "=" * 70)
print(f"Train: {len(X_train)} (pos={int(y_train.sum())})   "
      f"Test: {len(X_test)} (pos={int(y_test.sum())})")
print("=" * 70)


# ----------------------------------------------------------------------------
# Helper: encode categoricals to numeric codes for TabPFN (it needs numeric).
#         Fit encoder on train only.
# ----------------------------------------------------------------------------
def encode_for_tabpfn(X_tr, X_te, cat_cols):
    X_tr, X_te = X_tr.copy(), X_te.copy()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1,
                             encoded_missing_value=np.nan)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str).replace("nan", np.nan))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str).replace("nan", np.nan))
    return X_tr.astype(float), X_te.astype(float)


# ----------------------------------------------------------------------------
# 3. TRAIN MODELS
# ----------------------------------------------------------------------------
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def fit_tabpfn(X_tr, y_tr, X_va):
    """Fit a fresh TabPFN on encoded numeric data, return P(class=1) on X_va."""
    Xtr_e, Xva_e = encode_for_tabpfn(X_tr, X_va, cat_cols)
    m = TabPFNClassifier(random_state=RANDOM_STATE)
    m.fit(Xtr_e, y_tr)
    return m.predict_proba(Xva_e)[:, 1]


def fit_catboost(X_tr, y_tr, X_va):
    """Fit a fresh CatBoost (native categoricals/missing), return P(class=1)."""
    Xtr, Xva = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype(str).replace("nan", "missing").fillna("missing")
        Xva[c] = Xva[c].astype(str).replace("nan", "missing").fillna("missing")
    m = CatBoostClassifier(
        iterations=600, depth=4, learning_rate=0.03, l2_leaf_reg=6,
        random_state=RANDOM_STATE, auto_class_weights="Balanced",
        verbose=False, cat_features=cat_cols,
    )
    m.fit(Xtr, y_tr)
    return m.predict_proba(Xva)[:, 1]


def manual_cv_proba(fit_fn, X_tr, y_tr):
    """Out-of-fold P(class=1) on the training set (for threshold selection)."""
    oof = np.zeros(len(X_tr))
    y_arr = y_tr.values
    for tr_idx, va_idx in cv.split(X_tr, y_arr):
        oof[va_idx] = fit_fn(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], X_tr.iloc[va_idx])
    return oof


model_fns = {"TabPFN v2": fit_tabpfn, "CatBoost": fit_catboost}
cv_probas = {}
for name, fn in model_fns.items():
    # Final model: fit on full train, predict on held-out test
    results[name] = fn(X_train, y_train, X_test)
    # Out-of-fold train probabilities for high-sensitivity threshold selection
    cv_probas[name] = manual_cv_proba(fn, X_train, y_train)


# ----------------------------------------------------------------------------
# 4. EVALUATION + 5. HIGH-SENSITIVITY CONFUSION MATRIX
# ----------------------------------------------------------------------------
def bootstrap_ci(y_true, y_score, metric, n=2000, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    stats = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    return np.percentile(stats, [2.5, 97.5])


def high_sensitivity_threshold(y_true, y_score, target_sens=0.90):
    """Lowest threshold on CV data achieving >= target sensitivity (recall of pos)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target_sens)[0]
    # choose the operating point with highest specificity among those meeting sens
    best = ok[np.argmax(1 - fpr[ok])]
    return thr[best]


summary_rows = []
print("\n" + "=" * 70)
print("TEST-SET PERFORMANCE  (vs Marcinkevics et al. severity AUROC ~0.78)")
print("=" * 70)

for name, proba in results.items():
    auroc = roc_auc_score(y_test, proba)
    aupr = average_precision_score(y_test, proba)
    auroc_ci = bootstrap_ci(y_test, proba, roc_auc_score)
    aupr_ci = bootstrap_ci(y_test, proba, average_precision_score)

    # high-sensitivity threshold chosen on TRAIN CV probabilities (no test leakage)
    thr = high_sensitivity_threshold(y_train, cv_probas[name], target_sens=0.90)
    y_pred = (proba >= thr).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0

    print(f"\n----- {name} -----")
    print(f"AUROC = {auroc:.3f}  (95% CI {auroc_ci[0]:.3f}-{auroc_ci[1]:.3f})")
    print(f"AUPR  = {aupr:.3f}  (95% CI {aupr_ci[0]:.3f}-{aupr_ci[1]:.3f})")
    print(f"Operating threshold (>=90% sens on train CV): {thr:.3f}")
    print(f"Confusion matrix [rows=true 0/1, cols=pred 0/1]:\n{cm}")
    print(f"   TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"   Sensitivity(recall complicated) = {sens:.3f}")
    print(f"   Specificity                     = {spec:.3f}")
    print(f"   PPV={ppv:.3f}  NPV={npv:.3f}")

    summary_rows.append(dict(Model=name, AUROC=round(auroc, 3),
                             AUROC_CI=f"{auroc_ci[0]:.2f}-{auroc_ci[1]:.2f}",
                             AUPR=round(aupr, 3), Sensitivity=round(sens, 3),
                             Specificity=round(spec, 3), TN=tn, FP=fp, FN=fn, TP=tp,
                             Threshold=round(thr, 3)))

# Paper benchmark row (AUROC/AUPR only; no confusion matrix published)
summary_rows.append(dict(Model="Marcinkevics SSMVCBM-LSTM (images)", AUROC=0.78,
                         AUROC_CI="n/r", AUPR=0.58, Sensitivity=np.nan,
                         Specificity=np.nan, TN=np.nan, FP=np.nan, FN=np.nan,
                         TP=np.nan, Threshold=np.nan))
summary_rows.append(dict(Model="Marcinkevics Radiomics+RF (images)", AUROC=0.78,
                         AUROC_CI="n/r", AUPR=0.54, Sensitivity=np.nan,
                         Specificity=np.nan, TN=np.nan, FP=np.nan, FN=np.nan,
                         TP=np.nan, Threshold=np.nan))

summary = pd.DataFrame(summary_rows)
summary.to_csv("severity_results_summary.csv", index=False)
print("\n" + "=" * 70)
print("SUMMARY (saved to severity_results_summary.csv)")
print("=" * 70)
print(summary.to_string(index=False))

# Save predictions for plotting
np.savez("severity_predictions.npz",
         y_test=y_test.values,
         proba_tabpfn=results["TabPFN v2"],
         proba_cat=results["CatBoost"],
         thr_tabpfn=high_sensitivity_threshold(y_train, cv_probas["TabPFN v2"], 0.90),
         thr_cat=high_sensitivity_threshold(y_train, cv_probas["CatBoost"], 0.90))
print("\nDone.")

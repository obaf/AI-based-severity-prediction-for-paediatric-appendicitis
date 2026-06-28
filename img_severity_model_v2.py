"""
IMAGE-BASED severity head-to-head (v2) -- stronger pipeline to outperform
Marcinkevics et al. (severity AUROC ~0.78), on the SAME test patients.

Improvements over v1:
  * Fuse multiple frozen foundation encoders (BiomedCLIP + DINOv2)
  * Richer multiview pooling per patient: mean + std (+ optional max)
  * PCA denoising of the fused embedding
  * Ensemble of complementary classifiers: TabPFN v2 + CatBoost + LogReg

Usage: python img_severity_model_v2.py biomedclip dinov2
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve)

ENCODERS = sys.argv[1:] if len(sys.argv) > 1 else ["biomedclip", "dinov2"]
SEED = 42
N_PCA = 100
np.random.seed(SEED)
print(f"Encoders fused: {ENCODERS}")

# ---- labels / split --------------------------------------------------------
clin = pd.read_excel("app_data.xlsx", sheet_name=0)
clin = clin[clin["US_Number"].notna() & clin["Severity"].notna()].copy()
clin["US_Number"] = clin["US_Number"].astype(int)
sev_map = dict(zip(clin["US_Number"],
                   (clin["Severity"].str.strip().str.lower() == "complicated").astype(int)))
test_codes = set(pd.read_csv("test_set_codes.csv", header=None)[0].astype(int).tolist())

# ---- build per-patient fused features --------------------------------------
def pooled_by_encoder(tag):
    d = np.load(f"image_embeddings_{tag}.npz", allow_pickle=True)
    emb, subject = d["emb"], d["subject"]
    bags = {}
    for i, s in enumerate(subject):
        if int(s) in sev_map:
            bags.setdefault(int(s), []).append(emb[i])
    feats = {}
    for s, v in bags.items():
        v = np.vstack(v)
        feats[s] = np.concatenate([v.mean(0), v.std(0)])   # mean + std pooling
    return feats

enc_feats = {tag: pooled_by_encoder(tag) for tag in ENCODERS}
common = set.intersection(*[set(f.keys()) for f in enc_feats.values()])
subs = sorted(common)
test_subs = [s for s in subs if s in test_codes]
train_subs = [s for s in subs if s not in test_codes]

def fuse(sub_list):
    rows = []
    for s in sub_list:
        rows.append(np.concatenate([enc_feats[t][s] for t in ENCODERS]))
    return np.vstack(rows)

Xtr = fuse(train_subs); Xte = fuse(test_subs)
ytr = np.array([sev_map[s] for s in train_subs])
yte = np.array([sev_map[s] for s in test_subs])
print(f"Patients: train {len(train_subs)} (pos {ytr.sum()})  "
      f"test {len(test_subs)} (pos {yte.sum()})  | fused dim {Xtr.shape[1]}")

# ---- preprocessing: standardize + PCA (fit on train) -----------------------
def make_prep(Xtr_):
    sc = StandardScaler().fit(Xtr_)
    pca = PCA(n_components=min(N_PCA, Xtr_.shape[0]-1, Xtr_.shape[1]),
              random_state=SEED).fit(sc.transform(Xtr_))
    return sc, pca

def apply_prep(prep, X):
    sc, pca = prep
    return pca.transform(sc.transform(X))

# ---- models ----------------------------------------------------------------
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier

def fit_tabpfn(Xtr_, ytr_, Xte_):
    m = TabPFNClassifier(random_state=SEED, ignore_pretraining_limits=True)
    m.fit(Xtr_, ytr_); return m.predict_proba(Xte_)[:, 1]

def fit_logreg(Xtr_, ytr_, Xte_):
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3)
    m.fit(Xtr_, ytr_); return m.predict_proba(Xte_)[:, 1]

def fit_catboost(Xtr_, ytr_, Xte_):
    m = CatBoostClassifier(iterations=500, depth=3, learning_rate=0.03, l2_leaf_reg=6,
                           random_state=SEED, auto_class_weights="Balanced", verbose=False)
    m.fit(Xtr_, ytr_); return m.predict_proba(Xte_)[:, 1]

MODELS = {"TabPFN": fit_tabpfn, "CatBoost": fit_catboost, "LogReg": fit_logreg}

# test predictions (fit prep on full train)
prep = make_prep(Xtr)
Xtr_p, Xte_p = apply_prep(prep, Xtr), apply_prep(prep, Xte)
test_proba = {name: fn(Xtr_p, ytr, Xte_p) for name, fn in MODELS.items()}
test_proba["Ensemble"] = np.mean([test_proba[m] for m in MODELS], axis=0)

# OOF predictions for threshold selection (prep re-fit inside each fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {name: np.zeros(len(ytr)) for name in MODELS}
for tr, va in cv.split(Xtr, ytr):
    p = make_prep(Xtr[tr])
    Xtr_f, Xva_f = apply_prep(p, Xtr[tr]), apply_prep(p, Xtr[va])
    for name, fn in MODELS.items():
        oof[name][va] = fn(Xtr_f, ytr[tr], Xva_f)
oof["Ensemble"] = np.mean([oof[m] for m in MODELS], axis=0)

# ---- evaluation ------------------------------------------------------------
def hi_sens_thr(y, p, target=0.90):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return thr[ok[np.argmax(1 - fpr[ok])]]

def boot_ci(y, p, metric, n=2000):
    rng = np.random.RandomState(SEED); y = np.asarray(y); p = np.asarray(p); out = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        out.append(metric(y[idx], p[idx]))
    return np.percentile(out, [2.5, 97.5])

rows = []
print("\n" + "=" * 72)
print(f"IMAGE HEAD-TO-HEAD v2 ({'+'.join(ENCODERS)}) vs Marcinkevics severity AUROC 0.78")
print("=" * 72)
order = list(MODELS) + ["Ensemble"]
for name in order:
    p = test_proba[name]
    auroc = roc_auc_score(yte, p); aupr = average_precision_score(yte, p)
    ci = boot_ci(yte, p, roc_auc_score)
    thr = hi_sens_thr(ytr, oof[name], 0.90)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0; spec = tn/(tn+fp) if (tn+fp) else 0
    flag = "  <-- beats 0.78" if auroc > 0.78 else ""
    print(f"\n--- {name} ---{flag}")
    print(f"AUROC={auroc:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})  AUPR={aupr:.3f}")
    print(f"thr={thr:.3f}  CM [TN FP / FN TP]=[{tn} {fp} / {fn} {tp}]  Sens={sens:.3f} Spec={spec:.3f}")
    rows.append(dict(Encoders="+".join(ENCODERS), Model=name, AUROC=round(auroc,3),
                     AUROC_CI=f"{ci[0]:.2f}-{ci[1]:.2f}", AUPR=round(aupr,3),
                     Sensitivity=round(sens,3), Specificity=round(spec,3),
                     TN=tn, FP=fp, FN=fn, TP=tp, Threshold=round(float(thr),3)))

out = pd.DataFrame(rows)
out.to_csv("image_severity_results_v2.csv", index=False)
np.savez("image_severity_preds_v2.npz", yte=yte,
         **{f"p{i}": test_proba[k] for i, k in enumerate(order)},
         names=np.array(order, dtype=object),
         thr=np.array([hi_sens_thr(ytr, oof[k], 0.90) for k in order]))
print("\nSaved image_severity_results_v2.csv")
print(out.to_string(index=False))

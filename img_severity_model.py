"""
IMAGE-BASED severe vs non-severe appendicitis prediction -- head-to-head with
Marcinkevics et al. (2023), evaluated on the SAME test patients (test_set_codes.csv).

Frozen vision foundation model embeddings (BiomedCLIP / DINOv2) are aggregated per
patient across multiple US views and classified with modern algorithms:
  (1) mean-pool -> TabPFN v2     (primary)
  (2) mean-pool -> Logistic Reg  (comparator)
  (3) Attention-MIL multiview    (modern multiview aggregation, parallels their fusion)

Paper benchmark (Table 6, severity, image-based): AUROC ~0.78, AUPR ~0.58.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve)

TAG = sys.argv[1] if len(sys.argv) > 1 else "biomedclip"
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ---- Load embeddings + labels ---------------------------------------------
d = np.load(f"image_embeddings_{TAG}.npz", allow_pickle=True)
emb, subject = d["emb"], d["subject"]
D = emb.shape[1]
print(f"[{TAG}] embeddings {emb.shape}, {len(np.unique(subject))} subjects")

clin = pd.read_excel("app_data.xlsx", sheet_name=0)
clin = clin[clin["US_Number"].notna() & clin["Severity"].notna()].copy()
clin["US_Number"] = clin["US_Number"].astype(int)
sev_map = dict(zip(clin["US_Number"],
                   (clin["Severity"].str.strip().str.lower() == "complicated").astype(int)))

test_codes = set(pd.read_csv("test_set_codes.csv", header=None)[0].astype(int).tolist())

# ---- Group view embeddings by patient -------------------------------------
bags = {}                       # subject -> (V, D) array
for i, s in enumerate(subject):
    if s in sev_map:
        bags.setdefault(int(s), []).append(emb[i])
bags = {s: np.vstack(v) for s, v in bags.items()}

subs = sorted(bags.keys())
test_subs = [s for s in subs if s in test_codes]
train_subs = [s for s in subs if s not in test_codes]
print(f"Patients with images+label: {len(subs)} | "
      f"train {len(train_subs)} | test {len(test_subs)}")

def pack(sub_list):
    X_mean = np.vstack([bags[s].mean(0) for s in sub_list])
    y = np.array([sev_map[s] for s in sub_list])
    return X_mean, y

Xtr_mean, ytr = pack(train_subs)
Xte_mean, yte = pack(test_subs)
print(f"Train pos={ytr.sum()}/{len(ytr)}  Test pos={yte.sum()}/{len(yte)}")

# Standardize (fit on train)
scaler = StandardScaler().fit(Xtr_mean)
Xtr_s = scaler.transform(Xtr_mean); Xte_s = scaler.transform(Xte_mean)

results = {}      # name -> test probabilities
cv_oof = {}       # name -> (y_train, oof_proba) for threshold selection
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---- (1) mean-pool -> TabPFN ----------------------------------------------
from tabpfn import TabPFNClassifier
def tabpfn_fit_proba(Xtr, ytr_, Xte):
    m = TabPFNClassifier(random_state=SEED, ignore_pretraining_limits=True)
    m.fit(Xtr, ytr_); return m.predict_proba(Xte)[:, 1]
results["TabPFN (mean-pool)"] = tabpfn_fit_proba(Xtr_s, ytr, Xte_s)
oof = np.zeros(len(ytr))
for tr, va in cv.split(Xtr_s, ytr):
    oof[va] = tabpfn_fit_proba(Xtr_s[tr], ytr[tr], Xtr_s[va])
cv_oof["TabPFN (mean-pool)"] = oof

# ---- (2) mean-pool -> Logistic Regression ---------------------------------
def logreg_fit_proba(Xtr, ytr_, Xte):
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    m.fit(Xtr, ytr_); return m.predict_proba(Xte)[:, 1]
results["LogReg (mean-pool)"] = logreg_fit_proba(Xtr_s, ytr, Xte_s)
oof = np.zeros(len(ytr))
for tr, va in cv.split(Xtr_s, ytr):
    oof[va] = logreg_fit_proba(Xtr_s[tr], ytr[tr], Xtr_s[va])
cv_oof["LogReg (mean-pool)"] = oof

# ---- (3) Attention-MIL (gated attention pooling, Ilse et al. 2018) ---------
class GatedAttnMIL(nn.Module):
    def __init__(self, d, hidden=128, attn=64, p=0.3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p))
        self.V = nn.Linear(hidden, attn); self.U = nn.Linear(hidden, attn)
        self.w = nn.Linear(attn, 1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, bag):               # bag: (V, d)
        h = self.fc(bag)                  # (V, hidden)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (V,1)
        a = torch.softmax(a, dim=0)
        z = (a * h).sum(0, keepdim=True)  # (1, hidden)
        return self.head(z).squeeze(), a

def standardize_bag(s):
    return torch.tensor(scaler.transform(bags[s]), dtype=torch.float32)

def train_mil(train_list, val_list, epochs=150, lr=1e-3, wd=1e-3):
    pos_w = torch.tensor([(len(train_list) - sum(sev_map[s] for s in train_list))
                          / max(1, sum(sev_map[s] for s in train_list))])
    net = GatedAttnMIL(D); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_auc, best_state, patience, bad = -1, None, 25, 0
    for ep in range(epochs):
        net.train(); order = np.random.permutation(train_list)
        for s in order:
            opt.zero_grad()
            logit, _ = net(standardize_bag(s))
            loss = lossf(logit.unsqueeze(0), torch.tensor([float(sev_map[s])]))
            loss.backward(); opt.step()
        # validation AUROC
        net.eval(); vp, vy = [], []
        with torch.no_grad():
            for s in val_list:
                lg, _ = net(standardize_bag(s)); vp.append(torch.sigmoid(lg).item()); vy.append(sev_map[s])
        if len(set(vy)) > 1:
            auc = roc_auc_score(vy, vp)
            if auc > best_auc:
                best_auc, best_state, bad = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience: break
    if best_state: net.load_state_dict(best_state)
    return net

def mil_proba(net, sub_list):
    net.eval(); out = []
    with torch.no_grad():
        for s in sub_list:
            lg, _ = net(standardize_bag(s)); out.append(torch.sigmoid(lg).item())
    return np.array(out)

# inner train/val split for early stopping
from sklearn.model_selection import train_test_split
tr_in, va_in = train_test_split(train_subs, test_size=0.2, stratify=ytr, random_state=SEED)
mil = train_mil(tr_in, va_in)
results["Attention-MIL"] = mil_proba(mil, test_subs)
# OOF for threshold
oof = np.zeros(len(ytr)); idx = {s: i for i, s in enumerate(train_subs)}
for tr, va in cv.split(train_subs, ytr):
    tl = [train_subs[i] for i in tr]; vl = [train_subs[i] for i in va]
    tl_in, tl_va = train_test_split(tl, test_size=0.2,
                                    stratify=[sev_map[s] for s in tl], random_state=SEED)
    net = train_mil(tl_in, tl_va)
    p = mil_proba(net, vl)
    for s, pi in zip(vl, p): oof[idx[s]] = pi
cv_oof["Attention-MIL"] = oof

# ---- Evaluate + high-sensitivity confusion matrices -----------------------
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
print(f"IMAGE-BASED HEAD-TO-HEAD  ({TAG})  vs Marcinkevics severity AUROC 0.78")
print("=" * 72)
for name, p in results.items():
    auroc = roc_auc_score(yte, p); aupr = average_precision_score(yte, p)
    ci = boot_ci(yte, p, roc_auc_score)
    thr = hi_sens_thr(ytr, cv_oof[name], 0.90)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0; spec = tn / (tn + fp) if (tn + fp) else 0
    print(f"\n--- {name} ---")
    print(f"AUROC={auroc:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})  AUPR={aupr:.3f}")
    print(f"thr={thr:.3f}  CM [TN FP / FN TP] = [{tn} {fp} / {fn} {tp}]  "
          f"Sens={sens:.3f} Spec={spec:.3f}")
    rows.append(dict(Encoder=TAG, Model=name, AUROC=round(auroc, 3),
                     AUROC_CI=f"{ci[0]:.2f}-{ci[1]:.2f}", AUPR=round(aupr, 3),
                     Sensitivity=round(sens, 3), Specificity=round(spec, 3),
                     TN=tn, FP=fp, FN=fn, TP=tp, Threshold=round(float(thr), 3)))

out = pd.DataFrame(rows)
out.to_csv(f"image_severity_results_{TAG}.csv", index=False)
np.savez(f"image_severity_preds_{TAG}.npz",
         yte=yte, **{f"p{i}": results[k] for i, k in enumerate(results)},
         names=np.array(list(results.keys()), dtype=object),
         thr=np.array([hi_sens_thr(ytr, cv_oof[k], 0.90) for k in results]))
print("\nSaved", f"image_severity_results_{TAG}.csv")
print(out.to_string(index=False))

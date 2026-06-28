"""Figures for the IMAGE-BASED head-to-head vs Marcinkevics et al. (severity)."""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix)

TAG = sys.argv[1] if len(sys.argv) > 1 else "biomedclip"
d = np.load(f"image_severity_preds_{TAG}.npz", allow_pickle=True)
yte = d["yte"]; names = list(d["names"]); thr = d["thr"]
preds = {names[i]: d[f"p{i}"] for i in range(len(names))}

PAPER_AUROC, PAPER_AUPR = 0.78, 0.58

# pick best model by AUROC for the headline confusion matrix
best = max(names, key=lambda n: roc_auc_score(yte, preds[n]))
best_thr = thr[names.index(best)]

# ---- confusion matrices (all models) --------------------------------------
n = len(names)
fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4.2))
if n == 1: axes = [axes]
for ax, name in zip(axes, names):
    p = preds[name]; t = thr[names.index(name)]
    cm = confusion_matrix(yte, (p >= t).astype(int))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=17,
                    fontweight="bold", color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xticks([0, 1], ["Non-sev", "Severe"]); ax.set_yticks([0, 1], ["Non-sev", "Severe"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name}\nAUROC={roc_auc_score(yte, p):.3f}", fontsize=10)
fig.suptitle(f"Image-based severity ({TAG}) — paper's test patients (n={len(yte)})",
             fontweight="bold")
fig.tight_layout(); fig.savefig(f"fig_img_confusion_{TAG}.png", dpi=150, bbox_inches="tight")
print("saved", f"fig_img_confusion_{TAG}.png")

# ---- ROC + PR -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for name in names:
    p = preds[name]
    fpr, tpr, _ = roc_curve(yte, p)
    ax1.plot(fpr, tpr, lw=2, label=f"{name} ({roc_auc_score(yte, p):.3f})")
    pr, rc, _ = precision_recall_curve(yte, p)
    ax2.plot(rc, pr, lw=2, label=f"{name} ({average_precision_score(yte, p):.3f})")
ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax1.text(0.45, 0.08, f"Marcinkevics best\nAUROC = {PAPER_AUROC}", color="red",
         bbox=dict(boxstyle="round", fc="white", ec="red"), fontsize=10)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR (Sensitivity)"); ax1.set_title(f"ROC — image-based ({TAG})")
ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)
ax2.axhline(PAPER_AUPR, color="red", ls=":", lw=1.2); ax2.text(0.05, PAPER_AUPR+0.02,
            f"Marcinkevics best AUPR={PAPER_AUPR}", color="red", fontsize=9)
ax2.axhline(yte.mean(), color="gray", ls="--", lw=1, label=f"Prevalence ({yte.mean():.2f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title(f"PR — image-based ({TAG})")
ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"fig_img_roc_pr_{TAG}.png", dpi=150, bbox_inches="tight")
print("saved", f"fig_img_roc_pr_{TAG}.png")

print(f"\nBest image model: {best}  AUROC={roc_auc_score(yte, preds[best]):.3f}  "
      f"AUPR={average_precision_score(yte, preds[best]):.3f}")

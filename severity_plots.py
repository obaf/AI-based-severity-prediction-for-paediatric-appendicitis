"""Visualizations: confusion matrices, ROC/PR curves, and AUROC comparison
to Marcinkevics et al. (2023) severity results."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix)

d = np.load("severity_predictions.npz")
y = d["y_test"]
models = {
    "TabPFN v2": (d["proba_tabpfn"], float(d["thr_tabpfn"])),
    "CatBoost":  (d["proba_cat"],    float(d["thr_cat"])),
}
PAPER_AUROC = 0.78  # Marcinkevics best severity (SSMVCBM-LSTM / Radiomics+RF)
PAPER_AUPR = 0.58

# ---------------------------------------------------------------- confusion mx
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (name, (proba, thr)) in zip(axes, models.items()):
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xticks([0, 1], ["Non-severe", "Severe"])
    ax.set_yticks([0, 1], ["Non-severe", "Severe"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    auroc = roc_auc_score(y, proba)
    ax.set_title(f"{name}\nAUROC={auroc:.3f} | high-sensitivity point", fontsize=11)
fig.suptitle("Confusion matrices — Severe vs Non-severe appendicitis (held-out test, n=79)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("fig_confusion_matrices.png", dpi=150, bbox_inches="tight")
print("saved fig_confusion_matrices.png")

# ---------------------------------------------------------------- ROC + PR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = {"TabPFN v2": "#1f77b4", "CatBoost": "#ff7f0e"}
for name, (proba, thr) in models.items():
    fpr, tpr, _ = roc_curve(y, proba)
    ax1.plot(fpr, tpr, color=colors[name], lw=2,
             label=f"{name} (AUROC={roc_auc_score(y, proba):.3f})")
    prec, rec, _ = precision_recall_curve(y, proba)
    ax2.plot(rec, prec, color=colors[name], lw=2,
             label=f"{name} (AUPR={average_precision_score(y, proba):.3f})")
ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance (0.50)")
ax1.axhline(PAPER_AUROC, color="red", ls=":", lw=1.5, alpha=0.0)  # placeholder
# Reference marker: paper's best severity AUROC as a labelled horizontal note
ax1.text(0.55, 0.10, f"Marcinkevics et al. best\nseverity AUROC = {PAPER_AUROC}",
         color="red", fontsize=10,
         bbox=dict(boxstyle="round", fc="white", ec="red"))
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate (Sensitivity)")
ax1.set_title("ROC — Severity"); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

prev = y.mean()
ax2.axhline(prev, color="gray", ls="--", lw=1, label=f"Prevalence ({prev:.2f})")
ax2.text(0.05, PAPER_AUPR+0.02, f"Marcinkevics best AUPR = {PAPER_AUPR}",
         color="red", fontsize=9)
ax2.axhline(PAPER_AUPR, color="red", ls=":", lw=1.2)
ax2.set_xlabel("Recall (Sensitivity)"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall — Severity"); ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_roc_pr_curves.png", dpi=150, bbox_inches="tight")
print("saved fig_roc_pr_curves.png")

# ---------------------------------------------------------------- bar compare
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["Marcinkevics\nRadiomics+RF\n(images)", "Marcinkevics\nSSMVCBM-LSTM\n(images)",
          "CatBoost\n(tabular)", "TabPFN v2\n(tabular)"]
aurocs = [0.78, 0.78, roc_auc_score(y, models["CatBoost"][0]),
          roc_auc_score(y, models["TabPFN v2"][0])]
auprs = [0.54, 0.58, average_precision_score(y, models["CatBoost"][0]),
         average_precision_score(y, models["TabPFN v2"][0])]
xpos = np.arange(len(labels)); w = 0.38
b1 = ax.bar(xpos - w/2, aurocs, w, label="AUROC", color="#4c72b0")
b2 = ax.bar(xpos + w/2, auprs, w, label="AUPR", color="#dd8452")
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f"{b.get_height():.2f}", ha="center", fontsize=9)
ax.axvline(1.5, color="gray", ls="--", alpha=0.5)
ax.text(0.5, 1.02, "Marcinkevics et al. (US images)", ha="center", fontsize=9, color="gray")
ax.text(2.5, 1.02, "This work (tabular, modern AI)", ha="center", fontsize=9, color="gray")
ax.set_xticks(xpos, labels, fontsize=9)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
ax.set_title("Severity prediction: this work vs Marcinkevics et al. (2023)",
             fontweight="bold")
ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("fig_comparison_bars.png", dpi=150, bbox_inches="tight")
print("saved fig_comparison_bars.png")

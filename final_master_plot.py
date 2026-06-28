"""Master comparison figure: Marcinkevics et al. vs this work (image + tabular)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

# ---- load image (v2) preds ----
iv = np.load("image_severity_preds_v2.npz", allow_pickle=True)
inames = list(iv["names"]); ithr = iv["thr"]; iyte = iv["yte"]
ipreds = {inames[i]: iv[f"p{i}"] for i in range(len(inames))}
img_best = "TabPFN"  # headline image model (best AUPR, beats paper on both)
img_p = ipreds[img_best]; img_t = ithr[inames.index(img_best)]

# ---- load tabular preds ----
tv = np.load("severity_predictions.npz")
tab_yte = tv["y_test"]; tab_p = tv["proba_tabpfn"]; tab_t = float(tv["thr_tabpfn"])

PAPER = {  # Marcinkevics et al. Table 6, severity (image-based)
    "Radiomics+RF": (0.78, 0.54),
    "SSMVCBM-LSTM": (0.78, 0.58),
}

# ================= Figure 1: master AUROC/AUPR bars =================
fig, ax = plt.subplots(figsize=(11, 5.5))
labels, aurocs, auprs, groups = [], [], [], []
for k, (a, p) in PAPER.items():
    labels.append(f"{k}\n(images)"); aurocs.append(a); auprs.append(p); groups.append("paper")
labels.append(f"Image: TabPFN\nBiomedCLIP+DINOv2"); aurocs.append(roc_auc_score(iyte, img_p))
auprs.append(average_precision_score(iyte, img_p)); groups.append("img")
labels.append("Tabular: TabPFN\n(clinical+lab+US findings)")
aurocs.append(roc_auc_score(tab_yte, tab_p)); auprs.append(average_precision_score(tab_yte, tab_p))
groups.append("tab")

x = np.arange(len(labels)); w = 0.38
b1 = ax.bar(x - w/2, aurocs, w, label="AUROC", color="#4c72b0")
b2 = ax.bar(x + w/2, auprs, w, label="AUPR", color="#dd8452")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012, f"{b.get_height():.2f}",
            ha="center", fontsize=9)
ax.axhline(0.78, color="red", ls=":", lw=1.2, alpha=0.7)
ax.text(len(labels)-0.5, 0.79, "paper best AUROC 0.78", color="red", ha="right", fontsize=8)
ax.axvline(1.5, color="gray", ls="--", alpha=0.4)
ax.text(0.5, 1.04, "Marcinkevics et al. (images)", ha="center", color="gray", fontsize=9)
ax.text(3.0, 1.04, "This work (modern AI)", ha="center", color="gray", fontsize=9)
ax.set_xticks(x, labels, fontsize=8.5); ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
ax.set_title("Severe vs non-severe appendicitis — severity prediction performance",
             fontweight="bold")
ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("fig_master_comparison.png", dpi=150, bbox_inches="tight")
print("saved fig_master_comparison.png")

# ================= Figure 2: image head-to-head ROC/PR + CM =================
fig = plt.figure(figsize=(15, 4.6))
ax1 = fig.add_subplot(1, 3, 1); ax2 = fig.add_subplot(1, 3, 2); ax3 = fig.add_subplot(1, 3, 3)
# ROC
for name in ["TabPFN", "LogReg", "Ensemble"]:
    p = ipreds[name]; fpr, tpr, _ = roc_curve(iyte, p)
    ax1.plot(fpr, tpr, lw=2, label=f"{name} ({roc_auc_score(iyte,p):.3f})")
ax1.plot([0,1],[0,1],"k--",alpha=0.4)
ax1.text(0.40,0.06,"paper best 0.78",color="red",bbox=dict(boxstyle="round",fc="white",ec="red"),fontsize=9)
ax1.set_xlabel("FPR"); ax1.set_ylabel("Sensitivity"); ax1.set_title("Image-based ROC (paper's test set)")
ax1.legend(loc="lower right", fontsize=9); ax1.grid(alpha=0.3)
# PR
for name in ["TabPFN", "LogReg", "Ensemble"]:
    p = ipreds[name]; pr, rc, _ = precision_recall_curve(iyte, p)
    ax2.plot(rc, pr, lw=2, label=f"{name} ({average_precision_score(iyte,p):.3f})")
ax2.axhline(0.58, color="red", ls=":", lw=1.2); ax2.text(0.05,0.60,"paper best AUPR 0.58",color="red",fontsize=8)
ax2.axhline(iyte.mean(), color="gray", ls="--", lw=1, label=f"prevalence {iyte.mean():.2f}")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Image-based PR")
ax2.legend(loc="upper right", fontsize=9); ax2.grid(alpha=0.3)
# Confusion matrix of best image model
cm = confusion_matrix(iyte, (img_p >= img_t).astype(int))
ax3.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax3.text(j,i,cm[i,j],ha="center",va="center",fontsize=18,fontweight="bold",
                 color="white" if cm[i,j]>cm.max()/2 else "black")
ax3.set_xticks([0,1],["Non-severe","Severe"]); ax3.set_yticks([0,1],["Non-severe","Severe"])
ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")
ax3.set_title(f"Best image model: TabPFN\nAUROC={roc_auc_score(iyte,img_p):.3f} (high-sens point)")
fig.suptitle("Image-based head-to-head vs Marcinkevics et al. — same test patients (n=56, 13 severe)",
             fontweight="bold")
fig.tight_layout(); fig.savefig("fig_image_headtohead.png", dpi=150, bbox_inches="tight")
print("saved fig_image_headtohead.png")

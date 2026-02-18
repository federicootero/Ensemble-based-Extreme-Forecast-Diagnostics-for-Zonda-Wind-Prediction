# -*- coding: utf-8 -*-
"""
heat_maps.py
ROC and performance heatmaps for EFI/SOT/CPF variables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc

# ============================================================
# CONFIGURATION
# ============================================================

# Number of variables and lead times
n_vars = len(variables)      # e.g., ws10i, fg10i, t2i, mx2ti
n_leads = len(lead_times)    # e.g., 24, 48, 72, 96, 120 hours

# Initialize results arrays
AUC_results   = np.zeros((n_vars, n_leads))
Skill_results = np.zeros((n_vars, n_leads))
TPR_results   = np.zeros((n_vars, n_leads))
FAR_results   = np.zeros((n_vars, n_leads))
Thr_results   = np.zeros((n_vars, n_leads))

# EFI/SOT/CPF data for Zonda events
efi_station = efi_station_all  # already loaded previously

# ============================================================
# MAIN ROC LOOP — 100 THRESHOLDS
# ============================================================

for i, var in enumerate(variables):

    for j in range(n_leads):

        # --------------------------
        # 1) Extract Zonda vs Non-Zonda values for EFI/SOT/CPF
        # --------------------------
        zonda_vals = efi_station[i, :, j]        # Zonda days
        nonzonda_vals = efi_data_non_zonda[i, :, j]  # Non-Zonda days

        # Build true labels: 1 for Zonda, 0 for Non-Zonda
        y_true = np.array([1]*len(zonda_vals) + [0]*len(nonzonda_vals))
        y_scores = np.concatenate([zonda_vals, nonzonda_vals])

        # Remove NaN values
        mask = ~np.isnan(y_scores)
        y_true_clean = y_true[mask]
        y_scores_clean = y_scores[mask]

        if len(y_scores_clean) == 0:
            AUC_results[i,j] = np.nan
            Skill_results[i,j] = np.nan
            TPR_results[i,j] = np.nan
            FAR_results[i,j] = np.nan
            Thr_results[i,j] = np.nan
            continue

        # --------------------------
        # 2) Define thresholds (100 values from max → min)
        # --------------------------
        min_val = y_scores_clean.min()
        max_val = y_scores_clean.max()
        thresholds = np.linspace(max_val, min_val, 100)

        tpr_list = []
        fpr_list = []

        # --------------------------
        # 3) Compute ROC manually
        # --------------------------
        for thr in thresholds:
            y_pred = (y_scores_clean >= thr).astype(int)

            tp = np.sum((y_pred == 1) & (y_true_clean == 1))
            fp = np.sum((y_pred == 1) & (y_true_clean == 0))
            tn = np.sum((y_pred == 0) & (y_true_clean == 0))
            fn = np.sum((y_pred == 0) & (y_true_clean == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        tpr_arr = np.array(tpr_list)
        fpr_arr = np.array(fpr_list)

        # --------------------------
        # 4) Compute AUC (Area Under ROC Curve)
        # --------------------------
        roc_auc = auc(fpr_arr, tpr_arr)

        # --------------------------
        # 5) Best threshold using Youden's J statistic
        # --------------------------
        J = tpr_arr - fpr_arr
        idx = np.argmax(J)
        best_thr = thresholds[idx]

        # --------------------------
        # 6) Save results
        # --------------------------
        AUC_results[i,j]   = roc_auc
        Skill_results[i,j] = 2 * roc_auc - 1  # Skill score = 2*AUC - 1
        TPR_results[i,j]   = tpr_arr[idx]
        FAR_results[i,j]   = fpr_arr[idx]
        Thr_results[i,j]   = best_thr

# ============================================================
# DATA FOR ANNOTATIONS (EFI/SOT/CPF)
# ============================================================

AUC_mean   = AUC_results
Skill_mean = Skill_results

df_auc   = pd.DataFrame([[f"{AUC_mean[i,j]:.3f}"   for j in range(n_leads)] for i in range(n_vars)],
                        index=var_names, columns=lead_times)
df_skill = pd.DataFrame([[f"{Skill_mean[i,j]:.3f}" for j in range(n_leads)] for i in range(n_vars)],
                        index=var_names, columns=lead_times)
df_tpr   = pd.DataFrame([[f"{TPR_results[i,j]:.3f}" for j in range(n_leads)] for i in range(n_vars)],
                        index=var_names, columns=lead_times)
df_far   = pd.DataFrame([[f"{FAR_results[i,j]:.3f}" for j in range(n_leads)] for i in range(n_vars)],
                        index=var_names, columns=lead_times)

# ============================================================
# PLOT 2×2 PANEL — EFI/SOT/CPF ROC METRICS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ------------------ AUC ------------------
sns.heatmap(
    AUC_mean, annot=df_auc, fmt="", cmap="YlOrBr",
    xticklabels=lead_times, yticklabels=var_names,
    cbar_kws={'label':'AUC'}, vmin=0.50, vmax=1.00,
    ax=axes[0,0], linewidths=0.5, linecolor='white',
    annot_kws={'color':'black','fontsize':13}
)
axes[0,0].set_title("ROC AUC (EFI/SOT/CPF)", fontsize=14, fontweight='bold')
axes[0,0].set_yticklabels(var_names, rotation=0)

# ------------------ Skill Score ------------------
sns.heatmap(
    Skill_mean, annot=df_skill, fmt="", cmap="PuOr",
    xticklabels=lead_times, yticklabels=var_names,
    cbar_kws={'label':'Skill Score'}, vmin=0, vmax=1,
    ax=axes[0,1], linewidths=0.5, linecolor='white',
    annot_kws={'color':'black','fontsize':13}
)
axes[0,1].set_title("Skill Score (EFI/SOT/CPF)", fontsize=14, fontweight='bold')
axes[0,1].set_yticklabels(var_names, rotation=0)

# ------------------ False Positive Rate ------------------
sns.heatmap(
    FAR_results, annot=df_far, fmt="", cmap="Blues",
    xticklabels=lead_times, yticklabels=var_names,
    cbar_kws={'label':'False Positive Rate'}, vmin=0, vmax=1,
    ax=axes[1,0], linewidths=0.5, linecolor='white',
    annot_kws={'color':'black','fontsize':13}
)
axes[1,0].set_title("False Positive Rate (EFI/SOT/CPF)", fontsize=14, fontweight='bold')
axes[1,0].set_yticklabels(var_names, rotation=0)

# ------------------ True Positive Rate ------------------
sns.heatmap(
    TPR_results, annot=df_tpr, fmt="", cmap="YlGn",
    xticklabels=lead_times, yticklabels=var_names,
    cbar_kws={'label':'True Positive Rate'}, vmin=0, vmax=1,
    ax=axes[1,1], linewidths=0.5, linecolor='white',
    annot_kws={'color':'black','fontsize':13}
)
axes[1,1].set_yticklabels(var_names, rotation=0)
axes[1,1].set_title("True Positive Rate (EFI/SOT/CPF)", fontsize=14, fontweight='bold')

# Common ylabel
for ax in axes.flatten():
    ax.set_ylabel("Variable (EFI/SOT/CPF)", rotation=90, fontsize=14, labelpad=20)

plt.tight_layout()
plt.savefig("Fig8.pdf", dpi=600)
plt.show()

# ============================================================
# NOTE:
# - 'nonzonda_vals' correspond to Non-Zonda dates (cleaned from ±3 days around Zonda events)
# - All metrics are calculated for EFI/SOT/CPF variables
# ============================================================

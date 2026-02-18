# -*- coding: utf-8 -*-
"""
heatmap_comparison.py
Comparison of EFI/SOT/CPF metrics with winner-color scheme
All dates must be the same for this plot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from sklearn.metrics import roc_curve, auc

# ============================================================
# 1. FUNCTION TO COMPUTE EFI/SOT/CPF METRICS
# ============================================================

def compute_metrics(zonda_values, nonzonda_values):
    """
    Compute AUC, Skill, TPR, and FPR using ROC (manual or sklearn)
    """
    y_true = np.concatenate([np.ones(len(zonda_values)), np.zeros(len(nonzonda_values))])
    y_scores = np.concatenate([zonda_values, nonzonda_values])

    # Remove NaN
    mask = ~np.isnan(y_scores)
    y_scores = y_scores[mask]
    y_true = y_true[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    fpr, tpr, thr = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    skill = 2*roc_auc - 1

    J = tpr - fpr
    idx = np.argmax(J)

    return roc_auc, skill, tpr[idx], fpr[idx]

# ============================================================
# 2. CPF: same shape as EFI/SOT, just example placeholder values
# ============================================================

nvar = 4
nlead = 5

# Placeholder CPF values (replace with real CPF if available)
AUC_cpf   = np.full((nvar, nlead), 0.75)
Skill_cpf = np.full((nvar, nlead), 0.50)
TPR_cpf   = np.full((nvar, nlead), 0.75)
FPR_cpf   = np.full((nvar, nlead), 0.25)

# ============================================================
# 3. EFI / SOT / CPF: compute metrics
# ============================================================

AUC_efi   = np.zeros((nvar,nlead))
Skill_efi = np.zeros((nvar,nlead))
TPR_efi   = np.zeros((nvar,nlead))
FPR_efi   = np.zeros((nvar,nlead))

AUC_sot   = np.zeros((nvar,nlead))
Skill_sot = np.zeros((nvar,nlead))
TPR_sot   = np.zeros((nvar,nlead))
FPR_sot   = np.zeros((nvar,nlead))

AUC_cpf   = np.zeros((nvar,nlead))
Skill_cpf = np.zeros((nvar,nlead))
TPR_cpf   = np.zeros((nvar,nlead))
FPR_cpf   = np.zeros((nvar,nlead))

for v in range(nvar):
    for lt in range(nlead):
        AUC_efi[v,lt], Skill_efi[v,lt], TPR_efi[v,lt], FPR_efi[v,lt] = compute_metrics(
            efi_station[v,:,lt], efi_data_non_zonda[v,:,lt])

        AUC_sot[v,lt], Skill_sot[v,lt], TPR_sot[v,lt], FPR_sot[v,lt] = compute_metrics(
            sot_station[v,:,lt], sot_data_non_zonda[v,:,lt])

        AUC_cpf[v,lt], Skill_cpf[v,lt], TPR_cpf[v,lt], FPR_cpf[v,lt] = compute_metrics(
            cpf_station[v,:,lt], cpf_data_non_zonda[v,:,lt])

# ============================================================
# 4. FUNCTION TO DETERMINE WINNER COLORS
# ============================================================

color_EFI  = (0.12, 0.47, 0.71, 0.85)   # blue
color_SOT  = (0.17, 0.63, 0.17, 0.85)   # green
color_CPF  = (1.00, 0.47, 0.00, 0.85)   # orange
color_GRAY = (0.80, 0.80, 0.80, 0.45)   # gray for non-winner

def winner_colors(vEFI, vSOT, vCPF):
    vals = np.array([vEFI, vSOT, vCPF], dtype=float)
    w = np.nanargmax(vals)
    cols = [color_GRAY, color_GRAY, color_GRAY]
    if w == 0: cols[0] = color_EFI
    elif w == 1: cols[1] = color_SOT
    else:       cols[2] = color_CPF
    return cols

# ============================================================
# 5. PLOT 2x2 FINAL PANEL (EFI/SOT/CPF)
# ============================================================

var_names  = ['10m wind', '10m gust', '2m temp', '2m max temp']
lead_times = ['24h','48h','72h','96h','120h']

panel_titles = ["AUC", "Skill Score", "True Positive Rate", "False Positive Rate"]

DATA_EFI = [AUC_efi, Skill_efi, TPR_efi, FPR_efi]
DATA_SOT = [AUC_sot, Skill_sot, TPR_sot, FPR_sot]
DATA_CPF = [AUC_cpf, Skill_cpf, TPR_cpf, FPR_cpf]

fig, axs = plt.subplots(2, 2, figsize=(16, 11))
axs = axs.flatten()

for p in range(4):
    ax = axs[p]
    ax.set_title(panel_titles[p] + " (EFI/SOT/CPF)", fontsize=15)

    ax.set_xticks(np.arange(nlead))
    ax.set_xticklabels(lead_times, fontsize=11)

    # Horizontal separators
    for yi in range(nvar+1):
        ax.axhline(yi-0.5, color='black', linewidth=2)

    ax.set_yticks(np.arange(nvar))
    ax.set_yticklabels(var_names, fontsize=11)

    for i in range(nvar):
        for j in range(nlead):

            vEFI = DATA_EFI[p][i,j]
            vSOT = DATA_SOT[p][i,j]
            vCPF = DATA_CPF[p][i,j]

            cEFI, cSOT, cCPF = winner_colors(vEFI, vSOT, vCPF)
            h = 1/3  # height of each stripe

            # EFI top stripe
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, h,
                                   facecolor=cEFI, edgecolor='black', linewidth=0.3))
            ax.text(j, i-0.5+h*0.5, f"{vEFI:.2f}", ha='center', va='center', fontsize=8)

            # SOT middle stripe
            ax.add_patch(Rectangle((j-0.5, i-0.5+h), 1, h,
                                   facecolor=cSOT, edgecolor='black', linewidth=0.3))
            ax.text(j, i-0.5+h*1.5, f"{vSOT:.2f}", ha='center', va='center', fontsize=8)

            # CPF bottom stripe
            ax.add_patch(Rectangle((j-0.5, i-0.5+2*h), 1, h,
                                   facecolor=cCPF, edgecolor='black', linewidth=0.3))
            ax.text(j, i-0.5+h*2.5, f"{vCPF:.2f}", ha='center', va='center', fontsize=8)

    ax.set_xlim(-0.5, nlead-0.5)
    ax.set_ylim(nvar-0.5, -0.5)
    ax.grid(False)

# ============================================================
# GLOBAL LEGEND (EFI/SOT/CPF + gray)
# ============================================================

legend_elements = [
    Patch(facecolor=color_EFI,  edgecolor='black', label='EFI winner'),
    Patch(facecolor=color_SOT,  edgecolor='black', label='SOT winner'),
    Patch(facecolor=color_CPF,  edgecolor='black', label='CPF winner'),
    Patch(facecolor=color_GRAY, edgecolor='black', label='Not winner')
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=4, fontsize=12, frameon=True)

plt.tight_layout()
plt.savefig("Fig11.pdf", dpi=600, format='pdf', bbox_inches='tight')
plt.show()

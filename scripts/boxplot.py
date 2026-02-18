# -*- coding: utf-8 -*-
"""
Figures 3, 4, 5: Boxplots of EFI/SOT/CPF for Zonda vs Non-Zonda events
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# Variables and plotting order (top → bottom)
# 0 = Wind Speed, 1 = Temperature, 2 = Wind Gust, 3 = Max Temperature
index_order = [0, 1, 2, 3]

var_names = [
    "Wind Speed EFI/SOT/CPF",
    "Temperature EFI/SOT/CPF",
    "Wind Gust EFI/SOT/CPF",
    "Max Temperature EFI/SOT/CPF"
]

# Forecast lead times in hours
lead_times = [24, 48, 72, 96, 120]

# Colors for Zonda vs Non-Zonda events
palette = {
    "Non-Zonda": "lightgray",
    "All Zonda": "steelblue"
}

# -----------------------------------------
# BUILD DATAFRAME FOR PLOTTING
# -----------------------------------------

records = []

# ------------------ ZONDA ------------------
for plot_label, real_idx in zip(var_names, index_order):
    for e in range(efi_station.shape[1]):  # number of Zonda dates
        for lt_idx, lt in enumerate(lead_times):
            value = efi_station[real_idx, e, lt_idx]
            records.append([plot_label, lt, value, "All Zonda"])

# ---------------- NON-ZONDA ----------------
for plot_label, real_idx in zip(var_names, index_order):
    for e in range(efi_data_non_zonda.shape[1]):
        for lt_idx, lt in enumerate(lead_times):
            value = efi_data_non_zonda[real_idx, e, lt_idx]
            records.append([plot_label, lt, value, "Non-Zonda"])

df = pd.DataFrame(records, columns=["Variable", "Forecast Hour", "EFI/SOT/CPF Value", "Event Type"])

# -----------------------------------------
# CREATE FIGURE
# -----------------------------------------

fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(12, 16),
    sharex=True
)

for ax, variable in zip(axes, var_names):
    data_sub = df[df["Variable"] == variable]

    sns.boxplot(
        data=data_sub,
        x="Forecast Hour",
        y="EFI/SOT/CPF Value",
        hue="Event Type",
        palette=palette,
        showfliers=False,
        whis=[5, 95],
        ax=ax
    )

    ax.set_title(variable, fontsize=14, fontweight='normal')
    ax.set_xlabel("")
    ax.set_ylabel("EFI/SOT/CPF Value", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Remove axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_ylim(-0.75, 1)

    # Remove individual legend
    if ax.legend_ is not None:
        ax.legend_.remove()

# -----------------------------------------
# GLOBAL LEGEND AT BOTTOM
# -----------------------------------------

handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    title="Event Type",
    loc="lower center",
    bbox_to_anchor=(0.52, 0.03),
    ncol=2,
    fontsize=13,
    title_fontsize=14,
    frameon=True,
    borderpad=0.3
)
legend.get_frame().set_facecolor(axes[0].get_facecolor())
legend.get_frame().set_edgecolor("gray")
legend.get_frame().set_linewidth(0.9)
legend.get_frame().set_boxstyle("round,pad=0.25")

fig.suptitle(
    "EFI/SOT/CPF Value Distribution: All Zonda vs Non-Zonda Events",
    fontsize=18, fontweight="normal",
    y=0.93
)

plt.tight_layout(rect=(0, 0.08, 1, 0.93))

# -----------------------------------------
# SAVE FIGURE
# -----------------------------------------

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Fig3_4_5.pdf"), dpi=300, bbox_inches='tight', format='pdf')

plt.show()

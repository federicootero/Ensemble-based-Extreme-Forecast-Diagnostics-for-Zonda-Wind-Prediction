# -*- coding: utf-8 -*-
"""
EFI / SOT / CPF distribution by event category
Created on Tue Dec 16 15:06:12 2025
@author: Federico Otero
"""

# ============================================================
#Long-Strong events: peak wind speed > 21 kt and duration > 9 h
#Long events: duration > 9 h only
#Strong events: peak wind speed > 21 kt only
#Note: The observational wind data must be requested from the Argentinian National Weather Service,
#(Servicio Meteorológico Nacional) to reproduce or extend the classification of event types.
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# ============================================================
# CONFIG
# ============================================================
variables = ['ws10i', 'fg10i', 't2i', 'mx2ti']
var_names = ['10m wind speed', '10m wind gust', '2m temperature', '2m max temperature']
forecast_hours = ['24h', '120h']
steps = [0, 4]
letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

datasets = {
    'efi_station_all': {'label': 'EFI: All Zonda events', 'color': '#1f77b4', 'ls': '-'},
    'efi_station_long': {'label': 'EFI: Long events', 'color': '#ff7f0e', 'ls': '--'},
    'efi_station_strong': {'label': 'EFI: Strong events', 'color': '#2ca02c', 'ls': '-.'},
    'efi_station_long_strong': {'label': 'EFI: Long+Strong events', 'color': '#d62728', 'ls': ':'},
    'sot_station_all': {'label': 'SOT: All Zonda events', 'color': '#9467bd', 'ls': '-'},
    'sot_station_long': {'label': 'SOT: Long events', 'color': '#8c564b', 'ls': '--'},
    'sot_station_strong': {'label': 'SOT: Strong events', 'color': '#e377c2', 'ls': '-.'},
    'sot_station_long_strong': {'label': 'SOT: Long+Strong events', 'color': '#7f7f7f', 'ls': ':'},
    'cpf_station_all': {'label': 'CPF: All Zonda events', 'color': '#bcbd22', 'ls': '-'},
    'cpf_station_long': {'label': 'CPF: Long events', 'color': '#17becf', 'ls': '--'},
    'cpf_station_strong': {'label': 'CPF: Strong events', 'color': '#ff9896', 'ls': '-.'},
    'cpf_station_long_strong': {'label': 'CPF: Long+Strong events', 'color': '#c49c94', 'ls': ':'}
}

# --- Fuente y estilos ---
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 18
SUBPLOT_TITLE_FONTSIZE = 16
LETTER_FONTSIZE = 16
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 12

# ============================================================
# MAX DENSITY (para ejes consistentes)
# ============================================================
max_densities = {}
yticks_dict = {}

# All dates must be the same for this plot!
for i, var in enumerate(variables):
    max_density = 0
    for j, step in enumerate(steps):
        for dataset in datasets.keys():
            vals = globals()[dataset][i, :, step]
            fig_tmp, ax_tmp = plt.subplots()
            kde = sns.kdeplot(data=vals, ax=ax_tmp, legend=False)
            y_vals = kde.lines[0].get_data()[1]
            max_density = max(max_density, y_vals.max())
            plt.close(fig_tmp)
    max_densities[var] = max_density
    yticks_dict[var] = np.round(np.linspace(0, max_density*1.1, 5), 2)

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(14, 18), dpi=100)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

for i, (var, var_name) in enumerate(zip(variables, var_names)):
    for j, step in enumerate(steps):
        ax = axes[i, j]
        ax_idx = i*2 + j

        # KDE for each category
        for dataset, config in datasets.items():
            vals = globals()[dataset][i, :, step]
            sns.kdeplot(
                data=vals,
                color=config['color'],
                label=config['label'],
                linestyle=config['ls'],
                linewidth=2,
                ax=ax
            )

        ax.set_xlim(-5, 3)
        ax.set_ylim(0, max_densities[var]*1.1)
        ax.set_yticks(yticks_dict[var])
        ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f'Value ({forecast_hours[j]})', fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

        if j == 0:
            ax.set_ylabel('Density', fontsize=LABEL_FONTSIZE)
            ax.text(-0.2, 0.5, var_name, transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=SUBPLOT_TITLE_FONTSIZE)
        else:
            ax.set_ylabel('')

        ax.text(0.02, 0.95, letters[ax_idx], transform=ax.transAxes,
                fontsize=LETTER_FONTSIZE, fontweight='bold', va='top')

# ============================================================
# Leyenda
# ============================================================
handles = [Line2D([0], [0], color=config['color'], lw=2, linestyle=config['ls'], label=config['label'])
           for config in datasets.values()]

fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=5,
    frameon=False,
    fontsize=LEGEND_FONTSIZE
)

plt.suptitle('EFI / SOT / CPF Distribution by Event Categories', y=0.99, fontsize=TITLE_FONTSIZE)
plt.tight_layout()
plt.show()

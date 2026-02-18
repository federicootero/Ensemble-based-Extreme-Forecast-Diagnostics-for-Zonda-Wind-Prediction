# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:38:46 2026

@author: Federico Otero
"""

# -*- coding: utf-8 -*-
"""
Figure 1: EFI and SOT surface diagnostics for selected Zonda event date

This script loads EFI and SOT diagnostics for a given Zonda event date,
extracts the selected lead times (24, 72, 120 h), and plots a 4x3 panel
of variables: Wind Speed, Wind Gust, Temperature, Max Temperature.

Requirements:
- Python >= 3.9
- Libraries: xarray, numpy, pandas, matplotlib, mpl_toolkits.basemap
- Access to ECMWF ensemble EFI and SOT GRIB files
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from datetime import datetime

# Suppress pcolormesh deprecation warning
matplotlib.rcParams['pcolor.shading'] = 'auto'

# ------------------------------------------------------------------
# USER CONFIGURATION
# ------------------------------------------------------------------

# Relative paths inside the repository
DATA_DIR_EFI = os.path.join("data", "EFI")  # place EFI GRIBs here
DATA_DIR_SOT = os.path.join("data", "SOT")  # place SOT GRIBs here

# Zonda event date (YYYY-MM-DD) and corresponding GRIB file name prefix
FECHA_EVENTO = "2023-12-16"
FILE_PREFIX = "event_20231216_"  # adjust to match your filenames

# Selected lead times (hours)
SELECTED_HOURS = [24, 72, 120]
SELECTED_STEP_INDICES = [0, 2, 4]  # corresponding step indices in GRIB

# Variables to plot
VARIABLES = {
    'ws10i': 'Wind Speed',
    'fg10i': 'Wind Gust',
    't2i': 'Temperature',
    'mx2ti': 'Max Temperature'
}

# Optional: spatial subregion to plot (Mendoza region)
LAT_SLICE = slice(-33.3, -32.3)
LON_SLICE = slice(-68.9, -68.7)

# Output directory relative to repo
OUTPUT_DIR = os.path.join("figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# LOAD EFI AND SOT DATA
# ------------------------------------------------------------------

efi_file = os.path.join(DATA_DIR_EFI, f"{FILE_PREFIX}EFI.grib")
sot_file = os.path.join(DATA_DIR_SOT, f"{FILE_PREFIX}SOT.grib")

efi = xr.open_dataset(efi_file, engine="cfgrib")
sot = xr.open_dataset(sot_file, engine="cfgrib")

# Subset spatial box if defined
efi = efi.sel(latitude=LAT_SLICE, longitude=LON_SLICE)
sot = sot.sel(latitude=LAT_SLICE, longitude=LON_SLICE)

# Extract relevant fields
efi_vars = {key: efi[key] for key in VARIABLES.keys()}
sot_vars = {key: sot[key] for key in VARIABLES.keys()}

# ------------------------------------------------------------------
# INITIALIZE MATRICES FOR SELECTED LEAD TIMES
# ------------------------------------------------------------------

n_vars = len(VARIABLES)
n_leads = len(SELECTED_HOURS)
matriz_efi = np.full((n_vars, n_leads, efi.latitude.size, efi.longitude.size), np.nan)
matriz_sot = np.full((n_vars, n_leads, sot.latitude.size, sot.longitude.size), np.nan)

fecha_evento_dt = pd.Timestamp(FECHA_EVENTO)

# Fill matrices
for step_idx, horas in zip(SELECTED_STEP_INDICES, SELECTED_HOURS):
    for t_idx in range(len(efi.time)):
        fecha_validacion = pd.Timestamp(efi.time[t_idx].values) + efi.step[step_idx].values
        if fecha_validacion.normalize() != fecha_evento_dt.normalize():
            continue
        for var_idx, var_key in enumerate(VARIABLES.keys()):
            matriz_efi[var_idx, SELECTED_HOURS.index(horas), :, :] = efi_vars[var_key][t_idx, step_idx, :, :].values
            matriz_sot[var_idx, SELECTED_HOURS.index(horas), :, :] = sot_vars[var_key][t_idx, step_idx, :, :].values

print("Matriz EFI shape:", matriz_efi.shape)
print("Matriz SOT shape:", matriz_sot.shape)

# ------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------

lon2d, lat2d = np.meshgrid(efi.longitude.values, efi.latitude.values)

cmap0 = LinearSegmentedColormap.from_list('', ['white', 'yellow', 'orange', 'red'])
levels_plot = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
norm = BoundaryNorm(levels_plot, ncolors=cmap0.N, clip=True)

lead_titles = [f"{h}h" for h in SELECTED_HOURS]
row_titles = list(VARIABLES.values())

fig, axs = plt.subplots(nrows=n_vars, ncols=n_leads, figsize=(9, 12))

for j in range(n_leads):
    for i_var in range(n_vars):
        ax = axs[i_var, j]
        m = Basemap(
            projection='cyl', resolution='l',
            llcrnrlon=float(efi.longitude.min()),
            llcrnrlat=float(efi.latitude.min()),
            urcrnrlon=float(efi.longitude.max()),
            urcrnrlat=float(efi.latitude.max()),
            ax=ax
        )
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(float(efi.latitude.min()), float(efi.latitude.max())+1, 1), labels=[1,0,0,0], fontsize=8)
        m.drawmeridians(np.arange(float(efi.longitude.min()), float(efi.longitude.max())+1, 1), labels=[0,0,0,1], fontsize=8)

        # EFI filled
        im = m.pcolormesh(lon2d, lat2d, matriz_efi[i_var, j, :, :], latlon=True, cmap=cmap0, norm=norm)

        # SOT black contours
        m.contour(lon2d, lat2d, matriz_sot[i_var, j, :, :], latlon=True, colors='k', linewidths=1, linestyles='solid', levels=[1, 2, 5, 8])

        # Optional: mark Mendoza station
        m.scatter(-68.8, -32.83, s=30, marker='o', color='g', edgecolors='k', zorder=6)

        ax.set_title(lead_titles[j], fontsize=9)

# Add row labels on the right
for row_idx, title in enumerate(row_titles):
    ax = axs[row_idx, -1]
    pos = ax.get_position()
    y_center = (pos.y0 + pos.y1) / 2
    fig.text(pos.x1 , y_center, title, va='center', ha='left', fontsize=10, rotation=270)

# Adjust layout and add colorbar
fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.25)
cbar_ax = fig.add_axes([0.91, 0.2, 0.015, 0.6])
cbar = fig.colorbar(im, cax=cbar_ax, label='EFI')

# Save figure
save_name = f"4vars_3leadtimes_{fecha_evento_dt.strftime('%Y%m%d')}.png"
plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=100, bbox_inches='tight')
plt.close(fig)

print(f"Figure saved as: {os.path.join(OUTPUT_DIR, save_name)}")

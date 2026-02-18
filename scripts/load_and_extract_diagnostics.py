"""
Load and extract surface EFI, SOT, or CPF diagnostics for Zonda wind analysis.

This script:
1. Loads ECMWF ensemble-based diagnostics (EFI, SOT, CPF) from GRIB files.
2. Consolidates yearly diagnostics into a single xarray Dataset (optional).
3. Extracts spatially averaged values over a fixed box (Mendoza region) for Zonda event dates.
4. Stores output as a NumPy array [variable, event, lead_time].

Notes:
- Raw ECMWF GRIB files are NOT included due to licensing restrictions. 
- Users must provide their own GRIB files. (https://apps.ecmwf.int/mars-catalogue/?class=od or https://www.ecmwf.int/)
- Processed observational Zonda dates must be provided as CSV or Excel.  
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import List

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR = "path_to_your_grib_files"  # User-defined
DIAGNOSTIC = "EFI"  # "EFI", "SOT", or "CPF"
TARGET_VARS = ["ws10i", "t2i", "fg10i", "mx2ti"]

# Mendoza region spatial box
LAT_SLICE = slice(-33.3, -32.3)
LON_SLICE = slice(-68.9, -68.7)

# Years to process (if consolidated yearly GRIBs exist)
YEARS = list(range(2013, 2025))

# Path to Zonda event dates
EVENTS_FILE = "path_to_zonda_event_dates.csv"  # Must contain columns: fechas_zonda, archivos

# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def load_surface_file(file_path: str, variables: List[str]) -> xr.Dataset:
    """Load a single GRIB file and retain selected surface variables."""
    ds = xr.open_dataset(
        file_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
    )
    available_vars = [v for v in variables if v in ds.variables]
    if not available_vars:
        raise ValueError(f"No target variables found in {file_path}")
    return ds[available_vars]


def load_yearly_diagnostics(years: List[int], data_dir: str, diagnostic: str) -> xr.Dataset:
    """Load and concatenate yearly diagnostics into one Dataset."""
    datasets = []
    for year in years:
        file_path = os.path.join(data_dir, f"{year}_{diagnostic}.grib")
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file: {file_path}")
            continue
        try:
            ds = load_surface_file(file_path, TARGET_VARS)
            datasets.append(ds)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    if not datasets:
        raise RuntimeError(f"No {diagnostic} datasets were loaded.")
    return xr.concat(datasets, dim="time")


def extract_box_mean(ds: xr.Dataset) -> xr.Dataset:
    """Spatial mean over predefined latitude–longitude box."""
    return ds.sel(latitude=LAT_SLICE, longitude=LON_SLICE).mean(dim=["latitude", "longitude"])


def extract_diagnostics_for_events(event_dates: pd.Series, file_ids: pd.Series, data_dir: str) -> np.ndarray:
    """Extract diagnostics for Zonda events over all lead times and variables."""
    n_vars = len(TARGET_VARS)
    n_events = len(event_dates)
    # Load one file to get number of steps
    sample_ds = xr.open_dataset(os.path.join(data_dir, f"{file_ids.iloc[0]}{DIAGNOSTIC}.grib"), engine="cfgrib")
    n_leads = sample_ds.sizes["step"]

    data = np.full((n_vars, n_events, n_leads), np.nan)

    for i, (date, fname) in enumerate(zip(event_dates, file_ids)):
        file_path = os.path.join(data_dir, f"{fname}{DIAGNOSTIC}.grib")
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file: {file_path}")
            continue

        ds = xr.open_dataset(file_path, engine="cfgrib")
        ds_box = extract_box_mean(ds)

        for j in range(n_leads):
            for t in range(len(ds.time)):
                valid_time = pd.to_datetime(ds.valid_time.values[t, j])
                if valid_time.normalize() == pd.to_datetime(date).normalize():
                    for v, var in enumerate(TARGET_VARS):
                        if var in ds_box:
                            data[v, i, j] = ds_box[var].isel(time=t, step=j).values

    return data

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {DIAGNOSTIC} diagnostics...")

    # Step 1: Load yearly diagnostics (optional)
    # diag_total = load_yearly_diagnostics(YEARS, DATA_DIR, DIAGNOSTIC)
    # print(diag_total)

    # Step 2: Load Zonda event dates
    if not os.path.exists(EVENTS_FILE):
        raise FileNotFoundError(f"Event file not found: {EVENTS_FILE}")

    zonda_df = pd.read_csv(EVENTS_FILE)  # Ensure columns: fechas_zonda, archivos
    zonda_dates = pd.to_datetime(zonda_df["fechas_zonda"])
    file_ids = zonda_df["archivos"]

    # Step 3: Extract diagnostics for events
    diagnostics = extract_diagnostics_for_events(zonda_dates, file_ids, DATA_DIR)
    print("Extraction completed.")
    print("Diagnostics array shape:", diagnostics.shape)

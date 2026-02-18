# -*- coding: utf-8 -*-
"""
Reproducible loader for EFI, SOT, and CPF for Zonda days.

Output:
    processed_data/efi_station.npy
    processed_data/sot_station.npy
    processed_data/cpf_station.npy

Variables: ws10i, fg10i, t2i, mx2ti
Lead times: 6 steps (24h, 48h, ..., 144h)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# EFI/SOT/CPF data directories (relative to repo)
DATA_DIR_EFI = os.path.join("data", "EFI")
DATA_DIR_SOT = os.path.join("data", "SOT")
DATA_DIR_CPF = os.path.join("data", "CPF")

# Station box (Mendoza example)
LAT_SLICE = slice(-32.3, -33.3)
LON_SLICE = slice(-68.9, -68.7)

# Variables to extract
VAR_KEYS = ['ws10i', 'fg10i', 't2i', 'mx2ti']

# Number of forecast steps
N_LEADS = 5  # 24h, 48h, ..., 120h

# Zonda days
zonda_days = ['20080518','20080520','20080712','20080810','20080815','20080826',
              '20080901','20081011','20081108','20090619','20090707','20090812',
              '20090814','20090905','20091004','20091010','20091012','20091023',
              '20091117','20091119','20091124','20100112','20100116','20100614',
              '20100623','20100711','20100819','20100826','20100828','20101027',
              '20101107','20101202','20101210','20101215','20110116','20110413',
              '20110422','20110714','20110725','20110827','20110829','20110911',
              '20111031','20111108','20120527','20120612','20120620','20120721',
              '20121108','20121224','20130527','20130531','20130627','20130718',
              '20130807','20130910','20131009','20131019','20131022','20131106',
              '20131114','20131116','20140529','20140611','20140702','20140706', 
              '20140802','20140830','20140925','20141017','20141203','20150606', 
              '20150712','20150727','20150805','20150821','20150825','20150828',
              '20150914','20151109','20151118','20160415','20160423','20160816',
              '20161101','20161103','20161104','20161204','20170617','20170715',
              '20171003','20171004','20171017','20171028','20171217','20180610',
              '20181029','20181104','20181213','20190624','20190721','20190829', 
              '20190920','20200611','20200929','20210818','20210821','20210911',
              '20210912','20220424','20220426','20220706','20220907','20221007',
              '20230609','20230626','20230721','20230821','20230909','20230910',
              '20230916','20231028','20231110','20231115','20231117','20231214',
              '20231216','20240429','20240507','20240609','20240613','20240618',
              '20240802','20240902','20240919','20240921','20240930']

zonda_dates = [datetime.strptime(date, "%Y%m%d") for date in zonda_days]

# ------------------------------------------------------------------
# FUNCTION TO LOAD DATA
# ------------------------------------------------------------------

def load_station_data(file_dir, prefix_suffix, lat_slice, lon_slice, var_keys, zonda_dates, n_leads):
    n_vars = len(var_keys)
    n_days = len(zonda_dates)
    arr = np.full((n_vars, n_days, n_leads), np.nan)

    for i, zonda_date in enumerate(zonda_dates):
        file_name = f"{prefix_suffix}_{zonda_date.strftime('%Y%m%d')}.grib"
        file_path = os.path.join(file_dir, file_name)

        if not os.path.exists(file_path):
            print(f"File missing: {file_path}")
            continue

        try:
            ds = xr.open_mfdataset(file_path, engine='cfgrib')
            station_data = ds.sel(latitude=lat_slice, longitude=lon_slice)
            station_data = station_data.mean(dim='latitude').mean(dim='longitude')

            for t_idx in range(len(ds.time)):
                for s_idx in range(len(ds.step)):
                    forecast_date = pd.Timestamp(ds.time[t_idx].values) + ds.step[s_idx].values
                    if forecast_date.normalize() == zonda_date.normalize():
                        for v_idx, var in enumerate(var_keys):
                            arr[v_idx, i, s_idx] = station_data[var][t_idx, s_idx].values

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    return arr

# ------------------------------------------------------------------
# LOAD EFI, SOT, CPF
# ------------------------------------------------------------------

efi_station = load_station_data(DATA_DIR_EFI, "EFI", LAT_SLICE, LON_SLICE, VAR_KEYS, zonda_dates, N_LEADS)
sot_station = load_station_data(DATA_DIR_SOT, "SOT", LAT_SLICE, LON_SLICE, VAR_KEYS, zonda_dates, N_LEADS)
cpf_station = load_station_data(DATA_DIR_CPF, "CPF", LAT_SLICE, LON_SLICE, VAR_KEYS, zonda_dates, N_LEADS)

# ------------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------------

os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/efi_station.npy", efi_station)
np.save("processed_data/sot_station.npy", sot_station)
np.save("processed_data/cpf_station.npy", cpf_station)

print("EFI, SOT, CPF station arrays saved in processed_data/")
print("Shapes:", efi_station.shape, sot_station.shape, cpf_station.shape)

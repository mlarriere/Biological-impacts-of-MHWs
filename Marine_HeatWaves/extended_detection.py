# -*- coding: utf-8 -*-
"""
Created on Tues 10 June 11:29:36 2025

Extending MHWs detection 

@author: Marguerite Larriere (mlarriere)
"""
# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

import time
from tqdm.contrib.concurrent import process_map
from functools import partial

from joblib import Parallel, delayed

#%% Server 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% Figure settings
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif':['Times'],
    "font.size": 10,           
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,   
    "text.latex.preamble": r"\usepackage{mathptmx}",  # to match your Overleaf font
})

# %% -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

path_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'

path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_duration = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/mhw_durations'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'
path_det_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer'
path_combined_thesh= '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/combined_thresholds'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
path_growth = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model'
path_growth_inputs_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs/austral_summer'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold
percentile = 90 

# Handling time
from datetime import datetime, timedelta
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0, 121)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)


# %% ============== LOAD DATA ==============
def merge_valid_events_with_gap(arr, min_duration=5, max_gap=2):
    arr = arr.astype(bool)
    n = len(arr)
    result = np.zeros(n, dtype=bool)

    # Find where arr changes (start/end of runs)
    diff = np.diff(np.concatenate(([0], arr.view(np.int8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Classify True segments
    valid_events = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_duration]
    if not valid_events:
        return result

    # Merge valid events separated by short gaps
    merged = [valid_events[0]]
    for s, e in valid_events[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap:
            merged[-1] = (prev_s, e)  # merge
        else:
            merged.append((s, e))

    for s, e in merged:
        result[s:e] = True

    return result

det5m = xr.open_dataset(os.path.join(path_det,'det_5m.nc')) #boolean dataset with detection 
mhw_duration_5m = xr.open_dataset(os.path.join(path_duration, "mhw_duration_5m.nc")).mhw_durations #dataset - shape (40, 365, 434, 1442)

def test_eta_xi_specific_period_with_duration(ieta=211, ixi=889, varname='mhw_abs_threshold_1_deg'):
    det5m_yr = det5m.isel(eta_rho=ieta)
    data = det5m_yr[varname]  # shape (years, days, xi_rho)
    
    durations_sel = mhw_duration_5m.isel(eta_rho=ieta)  # shape (years, days, xi_rho)
    
    # Select years 37 to 38 (inclusive of 37 and 38)
    years_sel = slice(37, 39)
    days_sel = slice(304, 365)
    
    # Extract detection and duration slices for the same subset
    ts_slice = data.isel(years=years_sel, days=days_sel, xi_rho=ixi).values.astype(bool)
    duration_slice = durations_sel.isel(years=years_sel, days=days_sel, xi_rho=ixi).values
    
    # Flatten years and days into one dimension
    ts_flat = ts_slice.reshape(-1)
    dur_flat = duration_slice.reshape(-1)
    
    print(f"Original detection for eta={ieta}, xi={ixi} days 304-365, years 37-38:")
    print(ts_flat.astype(int))
    
    print(f"Duration values (should be >0 where event exists):")
    print(dur_flat)
    
    # Only gap fill where duration > 0 (valid MHW events)
    mask_for_gap_fill = dur_flat > 0
    ts_to_fill = ts_flat & mask_for_gap_fill
    
    # Apply gap filling to the masked array
    extended_ts = merge_valid_events_with_gap(ts_to_fill, min_duration=5, max_gap=2)
    
    print(f"Extended detection after applying gap filling only where duration > 0:")
    print(extended_ts.astype(int))
    
    print(f"Original True count: {np.sum(ts_flat)}")
    print(f"Masked True count (duration > 0): {np.sum(mask_for_gap_fill)}")
    print(f"Extended True count: {np.sum(extended_ts)}")
    
    return ts_flat, extended_ts

# Run test for your specific case
original, extended = test_eta_xi_specific_period_with_duration()

def extended_detection(ieta, varname):
    # ieta=211 
    # varname='mhw_abs_threshold_1_deg'
    det5m_eta = det5m.isel(eta_rho=ieta)
    data = det5m_eta[varname]  # shape: (years, days, xi_rho)

    # Stack data
    data_stacked = data.stack(time=['years', 'days'])  # shape: (xi_rho, years*days)

    # Apply the gap filling 
    extended_array = np.empty_like(data_stacked)
    extended_array = np.apply_along_axis(merge_valid_events_with_gap, axis=1, arr=data_stacked,
                                        min_duration=5, # Only events ≥5 days
                                        max_gap=2 # Merge if gap ≤2 days
                                        )

    return ieta, extended_array

# --- Initialisation
threshold_vars = ['mhw_abs_threshold_1_deg', 'mhw_abs_threshold_2_deg', 'mhw_abs_threshold_3_deg', 'mhw_abs_threshold_4_deg']
results_dict = {var: np.empty((neta, nxi, nyears*ndays), dtype=bool) for var in threshold_vars}
# extended_det_ds = np.empty((neta, nxi, nyears*ndays), dtype=bool)

# --- Calling function to combine eta (process_map)
for varname in threshold_vars:
    extended_detection_ieta = partial(extended_detection, varname=varname)
    for ieta, extended_det in process_map(extended_detection_ieta, range(neta), max_workers=30, desc=f"Processing {varname}"):
        results_dict[varname][ieta] = extended_det

# extended_detection_ieta = partial(extended_detection)
# for ieta, extended_det  in process_map(extended_detection_ieta, range(neta), max_workers=30, desc="Processing eta"):
#     extended_det_ds[ieta] = extended_det

# --- Reshaping
da_dict = {}
for varname in threshold_vars:
    extended_det_ds = results_dict[varname]

    # Flip eta position
    extended_det_transposed = extended_det_ds.transpose(2, 0, 1)
    
    # To DataArray  
    da = xr.DataArray(
        extended_det_transposed,
        coords={
            'time': det5m[varname].stack(time=['years', 'days']).time,
            'eta_rho': det5m[varname]['eta_rho'],
            'xi_rho': det5m[varname]['xi_rho'],
        },
        dims=['time', 'eta_rho', 'xi_rho']
    )

    # Reshape time
    da_unstacked = da.unstack('time')
    da_dict[varname] = da_unstacked

# --- To DataSet
extended_ds = xr.Dataset(da_dict)
# Rename variables
extended_ds['det_1deg_extended'] = extended_ds['mhw_abs_threshold_1_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_1_deg'])
extended_ds['det_2deg_extended'] = extended_ds['mhw_abs_threshold_2_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_2_deg'])
extended_ds['det_3deg_extended'] = extended_ds['mhw_abs_threshold_3_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_3_deg'])
extended_ds['det_4deg_extended'] = extended_ds['mhw_abs_threshold_4_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_4_deg'])

# Add coordinates
extended_ds = extended_ds.assign_coords(
    lon_rho=(('eta_rho', 'xi_rho'), ds_roms.lon_rho.values),
    lat_rho=(('eta_rho', 'xi_rho'), ds_roms.lat_rho.values),
    years=extended_ds.coords['years']+1980,
    days=extended_ds.coords['days'],
)

# Add description
extended_ds.attrs["description"] = (
    "Gap-filled detections of absolute thresholds. "
    "Gap-filling was only applied when events last at least 5 days, with gaps of 1 or 2 days in between."
)

# --- Save file
output_file = os.path.join(path_det, f"det5m_extended.nc")
if not os.path.exists(output_file):
    extended_ds.to_netcdf(output_file, engine="netcdf4")


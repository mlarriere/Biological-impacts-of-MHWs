# -*- coding: utf-8 -*-
"""
Created on Frid 06 Feb 08:40:06 2025

Calculate MHW durations for seasonal dataset, i.e. defined as 1st November to the 30th April

Note: We define the duration based solely on the condition that a mhw event is when T°C > relative thresholds

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob
import collections

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta
import time
from tqdm.contrib.concurrent import process_map

from joblib import Parallel, delayed


#%% -------------------------------- Server --------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% -------------------------------- Figure settings --------------------------------
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif':['Times'],
    "font.size": 9,           
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,   
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

path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
joel_path ='/home/jwongmeng/work/ROMS/scripts/mhw_krill/' #codes joel
output_path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'
path_fixed_baseline = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'
path_duration = os.path.join(path_fixed_baseline, 'mhw_durations')

# %% Functions
def extract_one_season_pair(args):
    ds_y, ds_y1, y = args
    try:
        days_nov_dec = ds_y.sel(days=slice(304, 365))
        days_jan_apr = ds_y1.sel(days=slice(0, 120))

        # Concatenate days and days_of_yr for new season dimension
        combined_days = np.concatenate([
            days_nov_dec['days'].values,
            days_jan_apr['days'].values
        ])
        combined_doy = np.concatenate([
            days_nov_dec['days_of_yr'].values,
            days_jan_apr['days_of_yr'].values
        ])

        season = xr.concat([days_nov_dec, days_jan_apr],
                           dim=xr.DataArray(combined_days, dims="days", name="days"))
        
        season = season.assign_coords(days_of_yr=("days", combined_doy))

        season = season.expand_dims(season_year=[y])
        return season

    except Exception as e:
        print(f"Skipping year {y}: {e}")
        return None

def define_season_all_years_parallel(ds, max_workers=6):
    from tqdm.contrib.concurrent import process_map

    all_years = ds['years'].values
    all_years = [int(y) for y in all_years if (y + 1) in all_years]

    # Pre-slice only needed years
    ds_by_year = {int(y): ds.sel(years=y) for y in all_years + [all_years[-1] + 1]}

    args = [(ds_by_year[y], ds_by_year[y + 1], y) for y in all_years]

    season_list = process_map(extract_one_season_pair, args, max_workers=max_workers, chunksize=1)

    season_list = [s for s in season_list if s is not None]
    if not season_list:
        raise ValueError("No valid seasons found.")

    return xr.concat(season_list, dim="season_year", combine_attrs="override")


# According to Hobday et al. (2016) - MHW needs to persist for at least five days (5days of TRUE)
def compute_mhw_durations(arr):
    """
    Compute duration of consecutive Trues (MHW) and Falses (non-MHW) in a 1D boolean array (time, ).
    Return two arrays of same shape: mhw_durations, non_mhw_durations.
    """
    # arr= bool_series
    arr = arr.astype(bool)
    n = arr.size
    durations = np.zeros(n, dtype=np.int32)
    
    if n == 0:  # Empty case
        return durations, durations

    # Find run starts and lengths
    change = np.diff(arr.view(np.int8), prepend=arr[0]) != 0 # Detect transitions
    run_id = np.cumsum(change)  # Label runs
    run_len = np.bincount(run_id)  # Length of each run

    # Map lengths back
    durations = run_len[run_id]

    mhw_durations = np.where(arr, durations, 0)
    non_mhw_durations = np.where(~arr, durations, 0)

    return mhw_durations, non_mhw_durations


def apply_hobday_rules(bool_event):
    # test
    # bool_event=bool_event_seasonal[10]
    xi_choice = 1000
    d0=140
    dfin=160
    eta_choice = 210

    # print('Initial: ', bool_event[d0:dfin, eta_choice, xi_choice]) #test

    ntime, neta, nxi = bool_event.shape

    # --- Calculate initial durations
    # Initialization
    reshaped = bool_event.reshape(ntime, neta * nxi) #shape (181, 625828)
    mhw_dur = np.zeros_like(reshaped, dtype=np.int32)
    non_mhw_dur = np.zeros_like(reshaped, dtype=np.int32)

    for i in range(reshaped.shape[1]):
        mhw_dur[:, i], non_mhw_dur[:, i] = compute_mhw_durations(reshaped[:, i])
    # print('First duration calculation: ', mhw_dur[d0:dfin, eta_choice * nxi + xi_choice]) #test

    # Reshape
    mhw_dur = mhw_dur.reshape(ntime, neta, nxi) 
    non_mhw_dur = non_mhw_dur.reshape(ntime, neta, nxi) 

    # --- Apply Hobday rules
    # MHW last at least 5 days
    mhw_event = mhw_dur >= 5 

    # Detecting gaps of 1 day 
    mhw_prev = np.roll(mhw_event, 1, axis=0)
    mhw_next = np.roll(mhw_event, -1, axis=0)
    mhw_prev[0] = False
    mhw_next[-1] = False

    gap_1 = (non_mhw_dur == 1) & mhw_prev & mhw_next 
    # print('Gap 1day: ', gap_1[d0:dfin, eta_choice, xi_choice]) #test

    # Detecting gaps of 2 days 
    mhw_prev2 = np.roll(mhw_event, 2, axis=0)
    mhw_next2 = np.roll(mhw_event, -2, axis=0)
    mhw_prev2[:2] = False
    mhw_next2[-2:] = False

    gap_2 = (non_mhw_dur == 2) & mhw_prev2 & mhw_next2 
    # print('Gap 2days: ', gap_2[d0:dfin, eta_choice, xi_choice]) #test

    # Combine events, i.e. allowing gaps of 1 and 2 days between 2 MHW events lasting more than 5 days
    mhw_combined = mhw_event | gap_1 | gap_2 

    # --- Recompute final durations
    reshaped = mhw_combined.reshape(ntime, neta * nxi)
    mhw_final = np.zeros_like(reshaped, dtype=np.int32)

    for i in range(reshaped.shape[1]):
        mhw_final[:, i], _ = compute_mhw_durations(reshaped[:, i])
    # print('Duration 2nd calculation: ', mhw_final[d0:dfin, eta_choice * nxi + xi_choice]) #test

    return mhw_final.reshape(ntime, neta, nxi)


def hobday_single_season(args):
    y, season_data = args
    return y, apply_hobday_rules(season_data)



def apply_hobday_rules_seasonal_parallel(bool_event_seasonal, max_workers=6):
    nyears, ntime, neta, nxi = bool_event_seasonal.shape
    mhw_out = np.zeros((nyears, ntime, neta, nxi), dtype=np.int32)

    tasks = [(y, bool_event_seasonal[y]) for y in range(nyears)]

    results = process_map(hobday_single_season, tasks, max_workers=max_workers, chunksize=1, desc="Years")
    for y, res in results:
        mhw_out[y] = res

    return mhw_out

# def apply_hobday_rules_seasonal(bool_event_seasonal):
#     # test
#     # bool_event_seasonal = det_rel_thres_season.values

#     nyears, ntime, neta, nxi = bool_event_seasonal.shape
#     mhw_out = np.zeros((nyears, ntime, neta, nxi), dtype=np.int32)

#     for y in range(nyears):
#         # test
#         # y=10
#         print(f"Processing season {y+1}/{nyears}")
#         mhw_out[y] = apply_hobday_rules(bool_event_seasonal[y])

#     return mhw_out

# %% 
ds_det= xr.open_dataset(os.path.join(path_fixed_baseline, f"det_depth/det_5m.nc")) #read each depth

# Relative-only MHWs
det_rel_thres = ds_det.mhw_rel_threshold # shape: (40, 365, 434, 1442). Boolean

# Restrict to latitudes south of 60°S
det_rel_thres_SO = det_rel_thres.where(det_rel_thres['lat_rho'] <= -60, drop=True) #shape: (40, 365, 231, 1442)
days_of_yr = det_rel_thres_SO['days'].values
det_rel_thres_SO = det_rel_thres_SO.assign_coords(days_of_yr=("days", days_of_yr))

# To seasonal dataset
det_rel_thres_season = define_season_all_years_parallel(det_rel_thres_SO, max_workers=6)
print(det_rel_thres_season.shape) #shape (39, 181, 231, 1442)
# Clean coordinate naming
det_rel_thres_season = det_rel_thres_season.rename({'season_year': 'season_year_temp'})
det_rel_thres_season = det_rel_thres_season.rename({'season_year_temp': 'years'})


# MHW durations for seasonal dataset
bool_event_seasonal = det_rel_thres_season.values.astype(bool)
mhw_duration_season = apply_hobday_rules_seasonal_parallel(bool_event_seasonal, max_workers=6) #shape (39, 181, 231, 1442)

# To dataset
mhw_duration_season_ds = xr.Dataset({'duration': (('years', 'days', 'eta_rho', 'xi_rho'), mhw_duration_season)},
                                     coords={'years': det_rel_thres_season['years'], 'days': det_rel_thres_season['days'],
                                             'lat_rho': det_rel_thres_season['lat_rho'], 'lon_rho': det_rel_thres_season['lon_rho'],},
                                     attrs={'Description': 'MHWs duration calcualted on seasonal dataset, i.e. detection of MHW events (90thperc) cropped to seaonal dataset and then duration calculated.\n'\
                                                           'Mhw events follow the rules of Hobday et al., i.e. duration >5days and gaps of 1-2 days allowed.'})

# test
dur = mhw_duration_season_ds.duration.isel(years=10)
duration_test = (dur > 0) & (dur < 5)
print(np.unique(duration_test)) # should be only False, OKAY! 
print('Max duration: ', dur.max().values)  #150days
print('Min duration: ',  dur.where(dur > 0).min().values) #5days

# Save to file 
mhw_duration_season_ds.to_netcdf(os.path.join(path_duration, 'mhw_duration_90thperc_seasonal.nc'))

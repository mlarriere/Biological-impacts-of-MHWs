#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 26 March 10:34:17 2025

Frequency of the 4 different events: SST > (relative + absolute thresholds)

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import matplotlib as mpl
import psutil #retracing memory
import glob
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

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
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0, 121)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

# %% ======================== Load data ========================
# MHW durations
mhw_duration_5m = xr.open_dataset(os.path.join(path_duration, "mhw_duration_5m.nc")).mhw_durations #dataset - shape (40, 365, 434, 1442) -- This has duration (90thperc + 1°C conditions!)

det_combined_ds = xr.open_dataset(os.path.join(path_det, 'det5m_extended.nc')) #boolean shape (40, 365, 434, 1442)

# -------------------------------------- FULL YEAR --------------------------------------
# -- Write or load data
combined_file_FULL = os.path.join(os.path.join(path_det, 'duration_AND_thresh_5mFULL.nc'))

if not os.path.exists(combined_file_FULL):

    # === Select 60°S south extent
    print("Selecting 60°S south extent...")
    south_mask = mhw_duration_5m['lat_rho'] <= -60
    mhw_duration_5m_NEW_60S_south = mhw_duration_5m.where(south_mask, drop=True) #shape (40, 365, 231, 1442)
    det_combined_ds_60S_south = det_combined_ds.where(south_mask, drop=True) #shape (40, 365, 231, 1442)
    det_combined_ds_60S_south = det_combined_ds_60S_south.transpose('years','days','eta_rho','xi_rho')


    # lat_mean_eta = mhw_duration_5m.lat_rho.mean(dim='xi_rho')
    # lat_idx = np.where(lat_mean_eta <= -60)[0]
    # south_mask = mhw_duration_5m['lat_rho'] <= -60
    
    # Duration MHW events defined as T°C > 90th percentile and 1°C
    # mhw_duration_5m_NEW_60S = mhw_duration_5m.isel(eta_rho=lat_idx) #shape (40, 365, 231, 1442)
    # mhw_duration_5m_NEW_60S = mhw_duration_5m_NEW_60S.transpose('years','days','eta_rho','xi_rho')

    # Duration MHW events defined as T°C > 90th percentile only
    # mhw_duration_5m_NEW_60S_90thperc = mhw_duration_5m_90th.isel(eta_rho=lat_idx) #shape (40, 365, 231, 1442)

    # # Detections of events above absolute thresholds
    # det_combined_ds_60S_south = det_combined_ds.isel(eta_rho=lat_idx) #shape (40, 365, 231, 1442)
    # det_combined_ds_60S_south = det_combined_ds_60S_south.transpose('years','days','eta_rho','xi_rho')
    # det_combined_ds_60S_south = det_combined_ds_60S_south.drop_vars(['xi_rho', 'eta_rho'])


    # === Associate each mhw duration with the event threshold and store in Datasets
    print('to Dataset...')
    ds_duration_thresh_FULLyear= xr.Dataset(
        data_vars=dict(
            duration = (["years", "days", "eta_rho" ,"xi_rho"], mhw_duration_5m_NEW_60S_south.data), #shape (40, 365, 231, 1442)
            det_1deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_1deg_extended'].data), #float64 [0, 1] -- ONLY ABSOLUTE
            det_2deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_2deg_extended'].data), #float64 [0, 1]
            det_3deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_3deg_extended'].data), #float64 [0, 1]
            det_4deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_4deg_extended'].data) #float64 [0, 1]
            ),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], mhw_duration_5m_NEW_60S_south.lon_rho.values), #(231, 1442)
            lat_rho=(["eta_rho", "xi_rho"], mhw_duration_5m_NEW_60S_south.lat_rho.values), #(231, 1442)
            days_of_yr=(['days'], mhw_duration_5m_NEW_60S_south.coords['days'].values), # Keeping information on day 
            years=(['years'], mhw_duration_5m_NEW_60S_south.coords['years'].values), # Keeping information on day 
            ),
        attrs = {
                "depth": "5m",
                "duration":"Duration redefined as following the rules of Hobday et al. (2016), based on relative threshold (90thperc).\n"\
                    "Duration calculated for events where T°C > relative threshold and 1°C.",
                "det_ideg": "Detected events where SST > (EXTENDED absolute threshold (i°C) BUT NOT NECESSARILY 90th percentile) , boolean array"
                }                
            )
    # ds_duration_thresh_FULLyear= xr.Dataset(data_vars=dict(duration = mhw_duration_5m_NEW_60S,
    #                                                        det_1deg = det_combined_ds_60S_south['det_1deg_extended'],
    #                                                        det_2deg = det_combined_ds_60S_south['det_2deg_extended'],
    #                                                        det_3deg = det_combined_ds_60S_south['det_3deg_extended'],
    #                                                        det_4deg = det_combined_ds_60S_south['det_4deg_extended']),
    #                                         coords=dict(years = mhw_duration_5m_NEW_60S['years'],
    #                                                     days = mhw_duration_5m_NEW_60S['days'],
    #                                                     lon_rho = mhw_duration_5m_NEW_60S['lon_rho'],
    #                                                     lat_rho = mhw_duration_5m_NEW_60S['lat_rho']),
    #                                         attrs = {"depth": "5m",
    #                                                 "duration":"Duration redefined as following the rules of Hobday et al. (2016), based on relative threshold (90thperc).\n"\
    #                                                            "Duration calculated for events where T°C > relative threshold and 1°C.",
    #                                                 "det_ideg": "Detected events where SST > (EXTENDED absolute threshold (i°C) BUT NOT NECESSARILY 90th percentile) , boolean array"})

    
    # Write to file
    print('Writing to file...')
    ds_duration_thresh_FULLyear.to_netcdf(combined_file_FULL)

else: 
    # Load data
    ds_duration_thresh_FULLyear = xr.open_dataset(combined_file_FULL)

#%%  -------------------------------------- SEASONAL --------------------------------------
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

ds_duration_thresh_FULLyear = xr.open_dataset(os.path.join(path_det, 'duration_AND_thresh_5mFULL.nc')) # shape: (40, 365, 231, 1442)

# -- Write or load data
combined_file = os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))
combined_file_90thperc = os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON_90th.nc'))

if not os.path.exists(combined_file):
    seasonal_vars = {}
    
    for var in ['duration', 'det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']:
        var='duration'
        print(f"Processing {var}")
        # Step 1: Wrap variable as dataset (this avoids sel errors in your parallel logic)
        var_ds = ds_duration_thresh_FULLyear[[var]]

        # Step 2: Extract seasonal slice in parallel -- about 10min computing per variable
        season_ds = define_season_all_years_parallel(var_ds, max_workers=10) #shape: (39, 181, 231, 1442)

        # Step 3: Clean coordinate naming
        season_ds = season_ds.rename({'season_year': 'season_year_temp'})
        if 'years' in season_ds:
            season_ds = season_ds.drop_vars('years')
        season_ds = season_ds.rename({'season_year_temp': 'years'})

        # Step 4: Save result
        seasonal_vars[var] = season_ds[var]  # Extract DataArray again for merge

    # Merge all seasonal DataArrays into one Dataset
    ds_duration_thresh_SEASON = xr.merge(seasonal_vars.values()) #shape: (39, 181, 231, 1442)
     
    # === Select only austral summer and early spring
    # jan_april = ds_duration_thresh_FULLyear.sel(days=slice(0, 120)) # 1 Jan to 30 April (Day 0-119) - last idx excluded
    # jan_april.coords['days'] = jan_april.coords['days'] #keep info on day
    # jan_april.coords['years'] = 1980+ jan_april.coords['years'] #keep info on day
    # nov_dec = ds_duration_thresh_FULLyear.sel(days=slice(304, 365)) # 1 Nov to 31 Dec (Day 304–364) - last idx excluded
    # nov_dec.coords['days'] = np.arange(304, 365) #keep info on day
    # nov_dec.coords['years'] = 1980+ nov_dec.coords['years'] #keep info on day
    # ds_duration_thresh_SEASON = xr.concat([nov_dec, jan_april], dim="days") #181days

    # Write to file
    ds_duration_thresh_SEASON.attrs = ds_duration_thresh_FULLyear.attrs.copy()
    ds_duration_thresh_SEASON.attrs["note"] = "Seasonal dataset (Nov 1 – Apr 30), derived from full-year MHW metrics."
    ds_duration_thresh_SEASON.to_netcdf(combined_file) #shape: (40, 181, 231, 1442)

else: 
    # Load data
    ds_duration_thresh_SEASON = xr.open_dataset(combined_file) #shape: (40, 181, 231, 1442)



# %% ======================== Compute Number of days under MHWs ========================
# -- Write or load data
mean_nb_days_file_FULL = os.path.join(os.path.join(path_det, 'mean_nb_days_underMHWs_5mFULL.nc')) #used after
nb_days_file_FULL = os.path.join(os.path.join(path_det, 'nb_days_underMHWs_5mFULL.nc')) #used for the weighting of length trajectories

if not os.path.exists(nb_days_file_FULL):

    # MHWs - UNION between duration (representing "extended" relative threshold) AND absolute extended thresholds
    mhw1deg = ds_duration_thresh_FULLyear['det_1deg'].where(ds_duration_thresh_FULLyear['duration']!=0) # shape - (years, days, eta, xi)
    mhw2deg = ds_duration_thresh_FULLyear['det_2deg'].where(ds_duration_thresh_FULLyear['duration']!=0)
    mhw3deg = ds_duration_thresh_FULLyear['det_3deg'].where(ds_duration_thresh_FULLyear['duration']!=0)
    mhw4deg = ds_duration_thresh_FULLyear['det_4deg'].where(ds_duration_thresh_FULLyear['duration']!=0)
    non_mhw = ds_duration_thresh_FULLyear['det_1deg'].where(ds_duration_thresh_FULLyear['duration']==0)

    # -- Selecting Growth Season
    def extract_one_season_pair(args):
        ds_y, ds_y1, y = args
        try:
            # Slice data
            days_nov_dec = ds_y.sel(days=slice(304, 365))
            days_jan_apr = ds_y1.sel(days=slice(0, 120))
            # Combine into 1 season
            combined_days = np.concatenate([days_nov_dec['days'].values, days_jan_apr['days'].values])
            season = xr.concat([days_nov_dec, days_jan_apr], dim=xr.DataArray(combined_days, dims="days", name="days"))
            season = season.expand_dims(season_year=[y])
            return season
        except Exception as e:
            print(f"Skipping year {y}: {e}")
            return None
    
    def define_season_all_years_parallel(ds, max_workers=10):
        # Keep only years where both y and y+1 exist --  exclude 2019 (nov-dec) 
        all_years = ds['years'].values
        all_years = [int(y) for y in all_years if (y + 1) in all_years]

        # Pre-slice only needed years
        ds_by_year = {int(y): ds.sel(years=y) for y in all_years + [all_years[-1] + 1]}

        # Extract / Define season by calling previous function
        args = [(ds_by_year[y], ds_by_year[y + 1], y) for y in all_years]
        season_list = process_map(extract_one_season_pair, args, max_workers=max_workers, chunksize=1)

        # Store results
        season_list = [s for s in season_list if s is not None]
        if not season_list:
            raise ValueError("No valid seasons found.")
        return xr.concat(season_list, dim="season_year", combine_attrs="override")

   
    mhw1deg_season = define_season_all_years_parallel(mhw1deg) #shape: (39, 181, 231, 1442)
    mhw1deg_season = mhw1deg_season.rename({'season_year': 'season_year_2'})
    mhw1deg_season = mhw1deg_season.drop_vars('years')
    mhw1deg_season = mhw1deg_season.rename({'season_year_2': 'years'})
    mhw1deg_season['years'] = np.arange(1980, 2019)  
    
    mhw2deg_season = define_season_all_years_parallel(mhw2deg) 
    mhw2deg_season = mhw2deg_season.rename({'season_year': 'season_year_2'})
    mhw2deg_season = mhw2deg_season.drop_vars('years')
    mhw2deg_season = mhw2deg_season.rename({'season_year_2': 'years'})
    mhw2deg_season['years'] = np.arange(1980, 2019)  
    
    mhw3deg_season = define_season_all_years_parallel(mhw3deg) 
    mhw3deg_season = mhw3deg_season.rename({'season_year': 'season_year_2'})
    mhw3deg_season = mhw3deg_season.drop_vars('years')
    mhw3deg_season = mhw3deg_season.rename({'season_year_2': 'years'})
    mhw3deg_season['years'] = np.arange(1980, 2019)  
    
    mhw4deg_season = define_season_all_years_parallel(mhw4deg) 
    mhw4deg_season = mhw4deg_season.rename({'season_year': 'season_year_2'})
    mhw4deg_season = mhw4deg_season.drop_vars('years')
    mhw4deg_season = mhw4deg_season.rename({'season_year_2': 'years'})
    mhw4deg_season['years'] = np.arange(1980, 2019)  
    
    non_mhw_season = define_season_all_years_parallel(non_mhw) 
    non_mhw_season = non_mhw_season.rename({'season_year': 'season_year_2'})
    non_mhw_season = non_mhw_season.drop_vars('years')
    non_mhw_season = non_mhw_season.rename({'season_year_2': 'years'})
    non_mhw_season['years'] = np.arange(1980, 2019)  
    
    # --- Number of MHWs days for each year -- shape (years, eta, xi)
    mhw_days_1deg_season = mhw1deg_season.sum(dim='days', skipna=True)  #max: 181 days -- mean ~2.08 days
    mhw_days_2deg_season = mhw2deg_season.sum(dim='days', skipna=True)  #max: 181 days -- mean ~0.97 days
    mhw_days_3deg_season = mhw3deg_season.sum(dim='days', skipna=True)  #max: 181 days -- mean ~0.507 days
    mhw_days_4deg_season = mhw4deg_season.sum(dim='days', skipna=True)  #max: 181 days -- mean ~0.258 days
    non_mhw_days_season = non_mhw_season.sum(dim='days', skipna=True)  #max: 181 days -- mean ~9.91 days

    # To dataset
    ds_mhw_nb_days_season = xr.Dataset(
            data_vars=dict(nb_days_1deg = (["years", "eta_rho" ,"xi_rho"], mhw_days_1deg_season.data), #shape (39, 231, 1442)
                           nb_days_2deg = (["years", "eta_rho" ,"xi_rho"], mhw_days_2deg_season.data),
                           nb_days_3deg = (["years", "eta_rho" ,"xi_rho"], mhw_days_3deg_season.data),
                           nb_days_4deg = (["years", "eta_rho" ,"xi_rho"], mhw_days_4deg_season.data),
                           nb_days_non_mhw = (["years", "eta_rho" ,"xi_rho"], non_mhw_days_season.data)),
            coords=dict(
                lon_rho=(["eta_rho", "xi_rho"], ds_duration_thresh_FULLyear.lon_rho.values), #(434, 1442)
                lat_rho=(["eta_rho", "xi_rho"], ds_duration_thresh_FULLyear.lat_rho.values), #(434, 1442)
                years = np.arange(1980, 2019)
                ),
            attrs = {"depth": "5m",
                     "nb_days_ideg":"Number of days under MHW exposure of intensity i°C",
                     "nb_days_non_mhw": "Number of days under no MHWs",
                     "time window" : "Growth season (181 days)"
                     }                
                )
    
    # --- Number of MHWs days for each year -- shape (years, eta, xi)
    mhw_days_1deg = mhw1deg.sum(dim='days') #max: 365 days -- mean ~2.9 days
    mhw_days_2deg = mhw2deg.sum(dim='days') #max: 365 days -- mean ~1.46 days
    mhw_days_3deg = mhw3deg.sum(dim='days') #max: 360 days -- mean ~0.74 days
    mhw_days_4deg = mhw4deg.sum(dim='days') #max: 326 days -- mean ~ days
    non_mhw_days = non_mhw.sum(dim='days') #max: 365 days -- mean ~ days

    # --- Mean number of MHWs days per year
    mhw1deg_days_per_year = mhw_days_1deg.mean(dim='years') #max: 74.125 days/yr
    mhw2deg_days_per_year = mhw_days_2deg.mean(dim='years') #max: 68.05 days/yr
    mhw3deg_days_per_year = mhw_days_3deg.mean(dim='years') #max: 68.05 days/yr
    mhw4deg_days_per_year = mhw_days_4deg.mean(dim='years') #max: 55.75 days/yr
    non_mhw_days_per_year = non_mhw_days.mean(dim='years') #max: 340.85 days/yr

    # To dataset
    ds_mhw_daysperyear= xr.Dataset(
            data_vars=dict(
                nb_days_1deg_per_yr = (["eta_rho" ,"xi_rho"], mhw1deg_days_per_year.data), #shape (231, 1442)
                nb_days_2deg_per_yr = (["eta_rho" ,"xi_rho"], mhw2deg_days_per_year.data),
                nb_days_3deg_per_yr = (["eta_rho" ,"xi_rho"], mhw3deg_days_per_year.data),
                nb_days_4deg_per_yr = (["eta_rho" ,"xi_rho"], mhw4deg_days_per_year.data),
                nb_days_nonmhw_per_yr = (["eta_rho" ,"xi_rho"], non_mhw_days_per_year.data)
                ),
            coords=dict(
                lon_rho=(["eta_rho", "xi_rho"], ds_duration_thresh_FULLyear.lon_rho.values), #(434, 1442)
                lat_rho=(["eta_rho", "xi_rho"], ds_duration_thresh_FULLyear.lat_rho.values), #(434, 1442)
                ),
            attrs = {
                    "depth": "5m",
                    "nb_days_ideg_per_yr":"Number of days per year being under MHW of i°C",
                    "time window": "full year (365days)"
                    }                
                )
    
    # Write to file
    ds_mhw_daysperyear.to_netcdf(mean_nb_days_file_FULL)
    ds_mhw_nb_days_season.to_netcdf(nb_days_file_FULL)
    
else: 
    # Load data
    ds_mhw_daysperyear = xr.open_dataset(mean_nb_days_file_FULL)
# %% ======================== Temperature of the water column (100m) under MHWs ========================
# -- Write or load data
avg_temp_FULL = os.path.join(os.path.join(path_clim, 'avg_temp_watercolumn_MHW.nc'))

if not os.path.exists(avg_temp_FULL):
    ds_avg_temp= xr.open_dataset(os.path.join(path_clim, 'avg_temp_watercolumn.nc'))

    # === Select 60°S south extent
    south_mask = ds_avg_temp['lat_rho'] <= -60
    ds_avg_temp_60S_south = ds_avg_temp.where(south_mask, drop=True) #shape (40, 365, 231, 1442)

    # Mask -- only cell under MHWs
    ds_mhw_duration = ds_duration_thresh_FULLyear.assign_coords(years=ds_avg_temp_60S_south['years'])
    ds_avg_temp_1deg = ds_avg_temp_60S_south.where((ds_mhw_duration['det_1deg'] > 0) & (ds_mhw_duration['duration'] != 0))
    ds_avg_temp_2deg = ds_avg_temp_60S_south.where((ds_mhw_duration['det_2deg'] > 0) & (ds_mhw_duration['duration'] != 0))
    ds_avg_temp_3deg = ds_avg_temp_60S_south.where((ds_mhw_duration['det_3deg'] > 0) & (ds_mhw_duration['duration'] != 0))
    ds_avg_temp_4deg = ds_avg_temp_60S_south.where((ds_mhw_duration['det_4deg'] > 0) & (ds_mhw_duration['duration'] != 0))
    
    # To dataset
    ds_avg_temp_mhw= xr.Dataset(
                data_vars=dict(
                    det_1deg = (["years", "days", "eta_rho" ,"xi_rho"], ds_avg_temp_1deg.temp.data),
                    det_2deg = (["years", "days", "eta_rho" ,"xi_rho"], ds_avg_temp_2deg.temp.data),
                    det_3deg = (["years", "days", "eta_rho" ,"xi_rho"], ds_avg_temp_3deg.temp.data),
                    det_4deg = (["years", "days", "eta_rho" ,"xi_rho"], ds_avg_temp_4deg.temp.data)
                    ),
                coords=dict(
                    lon_rho=(["eta_rho", "xi_rho"], ds_mhw_duration.lon_rho.values), #(231, 1442)
                    lat_rho=(["eta_rho", "xi_rho"], ds_mhw_duration.lat_rho.values), #(231, 1442)
                    ),
                attrs = {
                        "det_ideg":"Avg temperature in the water column under MHW of i°C"
                        }                
                    )
    
    # Mean temperature over years
    ds_avg_temp_mhw_meantime = ds_avg_temp_mhw.mean(dim=('years', 'days'))

    # Write to file
    ds_avg_temp_mhw.to_netcdf(avg_temp_FULL)
    ds_avg_temp_mhw_meantime.to_netcdf(os.path.join(path_clim, 'avg_temp_watercolumn_MHW_mean.nc'))

else: 
    # Load data
    ds_avg_temp_mhw = xr.open_dataset(avg_temp_FULL)
    ds_avg_temp_mhw_meantime = xr.open_dataset(os.path.join(path_clim, 'avg_temp_watercolumn_MHW_mean.nc'))



# %% ======================== Plot Avg temperature ========================
# --- Define cmap and norm for temperature (2nd row) ---
vmin, vmax = -3, 3
colors = ["#001219", "#669BBC", "#FFFFFF", "#CA6702", "#AE2012", "#5C0101"]
positions = np.linspace(0, 1, len(colors))
# But you want white exactly in the center (position 0.5)
positions = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
cmap_temp = LinearSegmentedColormap.from_list("temperature_100m_avg", list(zip(positions, colors)), N=256)
# cmap_temp = 'coolwarm'
norm_temp = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
variables_temp = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
titles = ['1°C', '2°C', '3°C', '4°C']

# 4 subplots - 1 for each absolute threshold 
plot='slides' #report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611*0.5
    fig_height = 9.3656988889 #674.33032pt
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(4, 1, hspace=0.15)

else:  # 'slides'
    fig_width = 6.3228348611  # inches = \textwidth
    fig_height = fig_width 
    fig = plt.figure(figsize=(fig_width*5, fig_height))  # wide enough for 4 subplots in a row
    gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)  # 4 columns

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}

# Axes creation
axs = []
for j in range(4):
    idx = (j, 0) if plot == 'report' else (0, j)
    ax = fig.add_subplot(gs[idx], projection=ccrs.SouthPolarStereo())
    axs.append(ax)


for i, var in enumerate(variables_temp):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map features
    lw = 1 if plot == 'slides' else 0.5
    ax.coastlines(color='black', linewidth=lw, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=lw, zorder=5)

    # Gridlines
    lw_grid = 0.7 if plot == 'slides' else 0.3
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=lw_grid, zorder=3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    # Font size settings for gridline labels
    gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot data
    im = ax.pcolormesh(
        ds_avg_temp_mhw_meantime.lon_rho,
        ds_avg_temp_mhw_meantime.lat_rho,
        ds_avg_temp_mhw_meantime[var],
        transform=ccrs.PlateCarree(),
        cmap=cmap_temp,
        norm=norm_temp,
        shading='auto',
        zorder=1,
        rasterized=True
    )

    if plot == 'report':
        ax.text(1.5, 0.5,  # x slightly left of axis (negative), y centered
            rf'MHWs $\ge$ {titles[i]}',
            rotation=-90, va='center', ha='center',
            transform=ax.transAxes, **subtitle_kwargs)
    else:
        ax.set_title(rf'MHWs $\ge$ {titles[i]}', **subtitle_kwargs)

# Common colorbar
tick_positions = np.arange(-3, 3.5, 1) 
if plot == 'report':
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', extend='both', location='bottom',
                        fraction=0.025, pad=0.04, ticks=tick_positions, shrink=0.9)
else:
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', extend='both',
        fraction=0.05,  # smaller fraction => thinner bar
        pad=0.07,
        shrink=1.1,      # >1 means longer than default
        ticks=tick_positions
    )
cbar.set_label('Temperature [°C]', **label_kwargs)
cbar.ax.tick_params(**tick_kwargs)

if plot == 'report':
    suptitle_y = 0.92
else:
    suptitle_y = 1.05
plt.suptitle("Average temperature in the upper 100m of the water column\nduring MHWs events (1980–2019)", **maintitle_kwargs, y=suptitle_y)

# --- Output handling ---    
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave space at bottom for colorbar
if plot == 'report':    
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/temp100m_report.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/temp100m_slides.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()


# %% ======================== Plot number of days under MHWs ========================
bounds = [0, 5, 10, 20, 30, 365]
labels = ['$<$5', '5-10', '10-20', '20-30', '$>$30']
n_bins = len(bounds) - 1
colors_list = [
    # '#FFFFFF',      # 0 = white (blank)
    '#440154',      # <5
    # '#3B528B',      
    '#21908C',      # 5-10
    # '#5DC963',      
    # '#FDE725',      
    "#FEC425",      # 10-20
    "#E99220",      # 20-30
    "#A32017"       # >30 (1month)
]
cmap = ListedColormap(colors_list)
norm = BoundaryNorm(bounds, ncolors=len(colors_list))#, extend='max')

# Parameters
variables = ['nb_days_1deg_per_yr', 'nb_days_2deg_per_yr', 'nb_days_3deg_per_yr', 'nb_days_4deg_per_yr']
titles = ['1°C', '2°C', '3°C', '4°C']

# Mask zeros in dataset
for var in variables:
    ds_mhw_daysperyear[var] = ds_mhw_daysperyear[var].where(ds_mhw_daysperyear[var]!=0)

# 4 subplots - 1 for each absolute threshold 
plot='slides' #slides report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611*0.5
    fig_height = 9.3656988889 #674.33032pt
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(4, 1, hspace=0.15)

else:  # 'slides'
    fig_width = 6.3228348611  # inches = \textwidth
    fig_height = fig_width 
    fig = plt.figure(figsize=(fig_width*5, fig_height))  # wide enough for 4 subplots in a row
    gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)  # 4 columns

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}

axs = []
for j in range(4):
    idx = (j, 0) if plot == 'report' else (0, j)
    ax = fig.add_subplot(gs[idx], projection=ccrs.SouthPolarStereo())
    axs.append(ax)


for i, var in enumerate(variables):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map features
    lw = 1 if plot == 'slides' else 0.5
    ax.coastlines(color='black', linewidth=lw, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=lw, zorder=5)

    # Gridlines
    lw_grid = 0.7 if plot == 'slides' else 0.3
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=lw_grid, zorder=3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot data
    im = ax.pcolormesh(
        ds_mhw_daysperyear.lon_rho,
        ds_mhw_daysperyear.lat_rho,
        ds_mhw_daysperyear[var],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading='auto',
        zorder=1,
        rasterized=True
    )

    if plot == 'report':
        ax.text(1.5, 0.5,  # x slightly left of axis (negative), y centered
            rf'MHWs $\ge$ {titles[i]}',
            rotation=-90, va='center', ha='center',
            transform=ax.transAxes, **subtitle_kwargs)
    else:
        ax.set_title( rf'MHWs $\ge$ {titles[i]}', **subtitle_kwargs)

# Common colorbar
tick_positions = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds)-1)]  # Tick positions at bin centers
# cbar = fig.colorbar(im, ax=axs, orientation='horizontal', extend='max', fraction=0.07, ticks=tick_positions)#aspect=10)
if plot == 'report':
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', extend='max', location='bottom',
                        fraction=0.025, pad=0.04, ticks=tick_positions, shrink=0.9)
else:
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', extend='max',
        fraction=0.05,  # smaller fraction => thinner bar
        pad=0.07,
        shrink=1.1,      # >1 means longer than default
        ticks=tick_positions
    )
cbar.set_label('days per year', **label_kwargs)
cbar.ax.set_xticklabels(labels)

if plot == 'report':
    suptitle_y = 0.92
else:
    suptitle_y = 1.05
plt.suptitle("Number of days per year under MHWs \n1980-2019 period - 5m depth",
    **maintitle_kwargs,
    y=suptitle_y
)

# --- Output handling ---
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave space at bottom for colorbar
if plot == 'report':    
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/nb_of_days_report.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/nb_of_days_slides.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()



# %% ======================== Compute frequency of events ========================
from scipy.ndimage import label

variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
avg_event_counts_dict = {}
duration_thresh = 30  #days

for var in variables:
    # Testing
    # var='det_1deg'

    print(f"------- {var} ------- ")

    # 1st step -- Selecting events lasting more than 30days and exceeding absolute threshold 
    valid_mask = (ds_duration_thresh_FULLyear['duration'] > duration_thresh) & (ds_duration_thresh_FULLyear[var] == 1) #boolean
    valid_mask_np = valid_mask.values  

    years_len, days_len, eta_len, xi_len = valid_mask_np.shape # shape: (40, 365, 231, 1442)
    eta_xi = eta_len * xi_len

    # Reshape
    valid_mask_reshaped = valid_mask_np.transpose(0, 2, 3, 1).reshape(years_len, eta_xi, days_len)

    # 2nd step -- Calculate frequency of these events, i.e. mean number of extreme days per year
    event_counts = np.zeros((years_len, eta_xi), dtype=int)

    # Counts events for each cell and year
    for i_year in range(years_len):
        for i_spatial in range(eta_xi):
            ts = valid_mask_reshaped[i_year, i_spatial]
            if np.any(ts):
                labeled, n_labels = label(ts)
                count = 0
                for i in range(1, n_labels + 1):
                    if np.sum(labeled == i) >= duration_thresh:
                        count += 1
                event_counts[i_year, i_spatial] = count

    # Reshape
    event_counts = event_counts.reshape(years_len, eta_len, xi_len)

    # Into DataArray
    event_counts_da = xr.DataArray(
        event_counts,
        coords={
            'years': ds_duration_thresh_FULLyear['years'],
            'eta_rho': ds_duration_thresh_FULLyear['eta_rho'],
            'xi_rho': ds_duration_thresh_FULLyear['xi_rho']},
        dims=['years', 'eta_rho', 'xi_rho'],
        name=f'mhw_event_counts_{var}'
    )

    # Annual mean for each grid cell
    avg_event_counts = event_counts_da.sum(dim='years') #max 1°C: 29yr, max 2°C: 28yr, max 3°C: 27yr, max 4°C: 24yr  
    avg_event_counts_dict[var] = avg_event_counts
    print(f"Maximum frequency for {var}: {np.max(avg_event_counts).values} years") 

# %% ======================== Find cell with NO MHWs ========================
no_mhw_cells = {}
var_no_mhws= ['det_1deg', 'det_2deg','det_3deg', 'det_4deg']

# -- Write or load data
no_mhw_file = os.path.join(os.path.join(path_det, 'noMHW_5m.nc'))
if not os.path.exists(no_mhw_file):
    # mask_nomhw = ds_duration_thresh_FULLyear['duration'] == 0

    for var in var_no_mhws:
        # var='det_1deg'
        print(f'---- {var} ----')
        
        # 1. Boolean array: where the given threshold was met and duration > 0
        is_mhw = ds_duration_thresh_FULLyear[var].astype(bool) & (ds_duration_thresh_FULLyear['duration'] > 0)

        # 2. Flag cells where that intensity threshold EVER occurred
        ever_had_this_mhw = is_mhw.any(dim=('years', 'days'))  # shape: (231, 1442)

        # 3. Invert to find cells that NEVER experienced it
        no_mhw_cells[var] = ~ever_had_this_mhw

        # Check
        count = no_mhw_cells[var].sum().item()
        print(f"{var}: {count} cells with no MHWs") #1°C: 210997 cells, 2°C: 269736 cells, 3°C: 307961 cells, 4°C: 320439 cells
   
    # Write to file
    import pandas as pd
    ds_no_mhw = xr.Dataset(
        no_mhw_cells,
        attrs={
            'description': 'Boolean masks indicating where no MHWs of specific intensity thresholds ever occurred.',
            'source': 'Derived from ds_duration_thresh_FULLyear.',
            'note': 'True means no MHWs of that intensity ever occurred at that location.',
            'created_on': str(pd.Timestamp.now().date())}
        )
    ds_no_mhw.to_netcdf(no_mhw_file)

else:
    # Load data
    ds_no_mhw = xr.open_dataset(no_mhw_file)

# %% ======================== Find cells with MHWs lasting less than 30days ========================
short_mhw_cells = {}
variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']

# -- Write or load data
short_mhw_file = os.path.join(os.path.join(path_det, 'shortMHW_5m.nc'))
if not os.path.exists(short_mhw_file):

    for var in variables:
        short_mhws = (ds_duration_thresh_FULLyear[var].astype(bool) & 
                      (ds_duration_thresh_FULLyear['duration'] > 0) & 
                      (ds_duration_thresh_FULLyear['duration'] < 30)) # True where MHW occurred & ≥30days
        
        short_mhw_cells[var] = short_mhws.any(dim=('years', 'days'))

    # Write to file
    ds_short_mhw = xr.Dataset(short_mhw_cells)
    ds_short_mhw.to_netcdf(short_mhw_file)

else: 
    # Load data
    ds_short_mhw = xr.open_dataset(short_mhw_file)

# %% ======================== Plot Number of years where cells under MHWs ========================
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba

variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']

bins = [0, 5, 10, 15, 20, 30]
colors = [
    "#8CB369",  # 1-5
    "#588978",  # 5-10
    "#E0BE15",  # 10-15
    # "#e05050",  
    "#D36A0D",  # 15-20
    "#800000",  # more than 20
]

# Create a ListedColormap and a BoundaryNorm for discrete bins
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bins, cmap.N)

long_mhw_ref = None  # To store the first mappable
# titles = ['$>1^\circ C$', '$>2^\circ C$', '$>3^\circ C$', '$>4^\circ C$']
titles = ['1°C', '2°C', '3°C', '4°C']

# 4 subplots - 1 for each absolute threshold 
plot='slides' #slides report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width 
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.05)  # ← control spacing

    axs = []
    for j in range(4):
        row = j // 2
        col = j % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
        axs.append(ax)
else:
    fig_width = 6.3228348611  # inches = \textwidth
    fig_height = fig_width 
    fig = plt.figure(figsize=(fig_width*5, fig_height))  # wide enough for 4 subplots in a row
    gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)  # 4 columns
    
    axs = []
    for j in range(4):
        ax = fig.add_subplot(gs[0, j], projection=ccrs.SouthPolarStereo())
        axs.append(ax)


# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}

long_mhw_ref = None
for i, var in enumerate(variables):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
    ax.set_boundary(circle, transform=ax.transAxes)

    # --- Layers ---
    no_mhw = ax.pcolormesh(ds_no_mhw['lon_rho'], ds_no_mhw['lat_rho'],
                           ds_no_mhw[var].astype(int).isel(xi_rho=slice(0, -1)),
                           cmap=ListedColormap([(0, 0, 0, 0), (1.0, 1.0, 1.0, 1)]),
                           shading='auto', transform=ccrs.PlateCarree(), zorder=2,rasterized=True)


    color_short_mhw = '#CCE3DE'  # pastel teal
    rgba_short_mhw = to_rgba(color_short_mhw)
    short_mhw = ax.pcolormesh(ds_short_mhw['lon_rho'], ds_short_mhw['lat_rho'],
                              ds_short_mhw[var].astype(int).isel(xi_rho=slice(0, -1)),
                              cmap=ListedColormap([(0, 0, 0, 0), rgba_short_mhw]),
                              shading='auto', transform=ccrs.PlateCarree(), zorder=1,rasterized=True)

    long_mhw = ax.pcolormesh(ds_duration_thresh_FULLyear['lon_rho'], ds_duration_thresh_FULLyear['lat_rho'],
                             np.ma.masked_less_equal(avg_event_counts_dict[var].isel(xi_rho=slice(0, -1)), 0),
                             cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
    
    # Features
    lw = 1 if plot == 'slides' else 0.5
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=lw, zorder=6)

    # Gridlines
    lw_grid = 0.7 if plot == 'slides' else 0.3
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=lw_grid, zorder=7)    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    if long_mhw_ref is None:
        long_mhw_ref = long_mhw

    ax.set_title(rf'MHWs $\ge$ {titles[i]}', **subtitle_kwargs)

# Legend
import matplotlib.patches as mpatches
if plot == 'report':
    legend_box = (-0.639, -0.28)
    lw_legend=0.5
else:
    legend_box = (0.02, -0.1)
    lw_legend=1

legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
short_patch = mpatches.Patch(facecolor=rgba_short_mhw, edgecolor='black', label='Short MHWs ($<$30d)', linewidth=lw_legend)
no_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='No MHWs', linewidth=lw_legend)
legend=axs[-1].legend(handles=[short_patch, no_patch], loc='lower center', bbox_to_anchor=legend_box, ncol=1, frameon=True, **legend_kwargs)
# legend.get_frame().set_linewidth(0.5)  # thinner box line

# Create common colorbar with ticks centered on bins
tick_positions = [(bins[j] + bins[j+1]) / 2 for j in range(len(bins)-1)]
if plot == 'report':
    cbar_ax = fig.add_axes([0.2, 0, 0.6, 0.02])  #[left, bottom, width, height]
    cbar = fig.colorbar(
        mappable=long_mhw_ref,
        cax=cbar_ax,
        orientation='horizontal',
        extend='max',
        fraction=0.025,
        pad=1.1,
        ticks=tick_positions,
        location='bottom',
        shrink=0.9
    )

else:
    cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.03])  #[left, bottom, width, height]
    cbar = fig.colorbar(
        mappable=long_mhw_ref,
            cax=cbar_ax,
            fraction=0.04,
            orientation='horizontal',
            ticks=tick_positions,
            extend='max'
        )
cbar.set_label('Years', **label_kwargs)
cbar.ax.set_xticklabels(['1-5', '5-10', '10-15', '15-20', '$>$20'])

# Apply font size from tick_kwargs
for label in cbar.ax.get_xticklabels():
    label.set_fontsize(tick_kwargs['labelsize'])

if plot == 'report':
    suptitle_y = 0.99
else:
    suptitle_y = 1.05
plt.suptitle("Number of years with events lasting more than 30 days\n1980-2019 period - 5m depth",
    **maintitle_kwargs,
    y=suptitle_y
)

# --- Output handling ---
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave space at bottom for colorbar
if plot == 'report':    
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/nb_of_years_report.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/nb_of_years_slides.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()



# %%

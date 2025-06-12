#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thurs 12 June 08:50:32 2025

COmputing growth with satellite observation and comparison with ROMS

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

import copernicusmarine
import cdsapi
from pprint import pprint

import time
from tqdm.contrib.concurrent import process_map

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
path_obs = '/nfs/sea/work/mlarriere/mhw_krill_SO/sat_obs'
path_up_obs = '/nfs/sea/work/datasets/gridded/ocean/3d/obs/chl/cmems_bgc_chl'

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

# %% ============= TEMPERATURE =============
# From Copernicus : https://cds.climate.copernicus.eu/datasets/satellite-sea-surface-temperature?tab=overview
import cdsapi

c = cdsapi.Client()

dataset = "satellite-sea-surface-temperature"
request = {
    "variable": "all",
    "processinglevel": "level_4",
    "sensor_on_satellite": "combined_product",
    "version": "2_1",
    "year": ["2017"],
    "month": ["01"],
    "day": [f"{i:02d}" for i in range(1, 32)],
    "format": "netcdf"
}

file_path = os.path.join(path_obs, "sst_satellite_201701.nc")

# Remove existing file if needed
if os.path.exists(file_path):
    os.remove(file_path)

# Download the data
c.retrieve(dataset, request).download(file_path)
print(f"Downloaded to: {file_path}")

# Unzip files imported
import zipfile
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(path_obs)
print("Extracted files:")
print(os.listdir(path_obs))

# Get list of files - daily
nc_files = sorted([
    os.path.join(path_obs, f)
    for f in os.listdir(path_obs)
    if f.endswith(".nc") and "C3S-L4" in f
])

# Load and subset all datasets
datasets = []
for file in nc_files:
    ds = xr.open_dataset(file)
    ds_south = ds.sel(lat=slice(-90, -60))  # Southern Ocean
    ds_south_deg = ds_south['analysed_sst']-273.15 # Convert kelvin to °C 
    datasets.append(ds_south_deg)  # Extract variable of interest

# Combine into a single dataset along time dimension
combined_ds = xr.concat(datasets, dim='time')

# Save southern ocean SST dataset - for the full month
subset_path = os.path.join(path_obs, "sst_satellite_201701_south60.nc")
combined_ds.to_netcdf(subset_path)
print(f"Saved combined subset dataset to: {subset_path}")

# Remove other netcdf files except subset
for f in os.listdir(path_obs):
    if f.endswith(".nc") and f != os.path.basename(subset_path):
        os.remove(os.path.join(path_obs, f))
print("Removed original daily .nc files except subset.")

temp_obs_2017_jan= xr.open_dataset(subset_path).analysed_sst 

# %% ============= CHLOROPHYLL =============
# Obs from CMEMS (copernicus - included different obs including Sea WIS)
# Get list of files - weekly
path_2017 = os.path.join(path_up_obs, '2017')
files_chl_jan = sorted([
    os.path.join(path_2017, f)
    for f in os.listdir(path_2017)
    if f.endswith(".nc") and "201701" in f
])

# Combine file together
datasets_chla = []
for file in files_chl_jan:
    ds = xr.open_dataset(file)
    ds_south = ds.sel(latitude=slice(-90, -60))  # Southern Ocean
    ds_south_surf = ds_south.isel(depth=0).chl # Select surface data
    datasets_chla.append(ds_south_surf)  # Extract variable of interest

ds_chl_2017_jan_weekly = xr.concat(datasets_chla, dim="time")


# %% Align datasets
# --- Spatially
import xesmf as xe

# Use chlorophyll lat/lon coords as target grid since 0.25° resolution (same as ROMS)
target_lat = ds_chl_2017_jan_weekly.latitude.values
target_lon = ds_chl_2017_jan_weekly.longitude.values

# Target grid 
target_grid = {
    'lat': target_lat,
    'lon': target_lon
}
# Regridding
regridder = xe.Regridder(temp_obs_2017_jan, target_grid, 'bilinear')
temp_obs_2017_jan_025 = regridder(temp_obs_2017_jan)

# --- Temporally
# Take monthly mean
temp_obs_2017_jan_monthly = temp_obs_2017_jan_025.resample(time='1MS').mean() #shape: (1, 89, 1440)
ds_chl_2017_jan_monthly = ds_chl_2017_jan_weekly.resample(time='1MS').mean() #shape: (1, 89, 1440)


# %% ============= Run model with Observations =============
import sys
sys.path.append(working_dir+'Growth_Model') 
from growth_model import growth_Atkison2006  # import growth function

output_file = os.path.join(path_growth, "growth_Atkison2006_obs_201701.nc")
if not os.path.exists(output_file):
    growth_obs_2017_jan = growth_Atkison2006(ds_chl_2017_jan_monthly, temp_obs_2017_jan_monthly)
    # Clean dataset
    ds_clean = growth_obs_2017_jan.drop_vars(['lat', 'lon'], errors='ignore') \
                .squeeze()  # removes size-1 dims like time if needed
    ds_clean = ds_clean.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds_clean = ds_clean.rename({'__xarray_dataarray_variable__': 'growth'})

    # Write to file
    ds_clean.to_netcdf(output_file)

else:
    growth_obs_2017_jan = xr.open_dataset(output_file)
    

# %% ============= ROMS =============
# Resulting growth with ROMS inputs
growth_ROMS = xr.open_dataset(os.path.join(path_growth, 'growth_Atkison2006_fullyr.nc'))

# Select January 2017
growth_ROMS_2017 = growth_ROMS.isel(years=2017-1980)

# Mean growth for this period
growth_ROMS_2017_jan_monthly = growth_ROMS_2017.isel(days=slice(0,30)).mean(dim='days') #shape: (eta_rho, xi_rho) : (231, 1442)
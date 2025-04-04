#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Friday 04 April 16:41:11 2025

Processing chlorophyll data from ROMS

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import time

from joblib import Parallel, delayed

# -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

file_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_surface.nc' # drift and bias corrected temperature files
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434
nxi = 1442


# ROMS chlorophyll in [mg Chla/m3]
def mean_chla(yr):

    start_time = time.time()

    ds_chla = xr.open_dataset(os.path.join(path_chla, f"z_SO_d025_avg_daily_{1980+yr}.nc"))
    ds_chla_100m = ds_chla.isel(time= slice(0,365), depth=slice(0, 14)) #depth from 0 to 100m depth
    da_chla_mean = ds_chla_100m.TOT_CHL.mean(dim='depth')

    # Reformating
    da_chla_mean = da_chla_mean.rename('raw_chla') # Rename variable
    da_chla_mean = da_chla_mean.assign_coords(time=np.arange(1, 366)) # Replace cftime with integer day-of-year: 1 to 365
    ds_chla_mean_yr = xr.Dataset({'raw_chla': da_chla_mean}).expand_dims(year=[1980 + yr]) # To dataset and adding year dimension

    # Write dataset to file
    ds_chla_mean_yr.to_netcdf(path=os.path.join(path_growth, f"chla_avg100m_daily_{1980+yr}.nc"), mode='w')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for {1980+yr}: {elapsed_time:.2f} seconds")

    return ds_chla_mean_yr 
    
# Calling function
chla_mean_yearly = Parallel(n_jobs=30)(delayed(mean_chla)(yr) for yr in range(0, nyears)) 

# Combine year
chla_mean_all = xr.concat(chla_mean_yearly, dim='year')
chla_mean_all.to_netcdf(path=os.path.join(path_growth, "chla_avg100m_yearly.nc"), mode='w')

# %%

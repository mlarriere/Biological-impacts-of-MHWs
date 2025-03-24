#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 19 Feb 16:46:48 2025

Calculate yearly average MHW durations

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

import time

from joblib import Parallel, delayed
from scipy.ndimage import label

# %% -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
output_path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon


# %% Computing averaged durations
ds_mhw_duration_stacked = xr.open_dataset(os.path.join(output_path, f"mhw_durations_extended_all_eta.nc"))

# Initialisation
avg_dur = np.zeros((nyears, neta, nxi))  # Average duration of events per year
nb_mhw = np.zeros((neta,nxi), dtype=int)   # number of event per location

# Calculating yearly avg of mhw events
# def avg_duration_yearly(ieta, ds):
for ieta in range(0, neta):
    # ieta = 200 #for test
    ds = ds_mhw_duration_stacked #for test

    ds_eta = ds.isel(eta_rho=ieta)

    for ixi in range(0, nxi): #Computation time for (1eta, all xi) ~ 3min

        # Read data
        ds = ds_eta.isel(xi_rho=ixi).stack(time=('years', 'days'))

        # ds = ds.isel(eta_rho=ieta, xi_rho=ixi).stack(time=('years','days'))

        # Initialisation count (number of events per year)
        yr_count = np.zeros(nyears) 

        # Get indices of mhw and number of events
        labeled_array, num_features = label(ds.mhw_duration.values)

        # If no MHW existing
        if num_features == 0:
            # print("No mhw this year at this location")
            continue
        for label_id in range(1, num_features + 1):

            idx_events = np.where(labeled_array == label_id)[0]
        
            # Starting and ending days of events
            event_start_days = idx_events[0] #ok
            event_end_days = idx_events[-1] #ok
            # print("Start day: ", event_start_days)
            # print("End day: ", event_end_days)


            # Years associated to starting and ending days
            idx_start_yr = np.ceil(event_start_days/365).astype(int) -1 #ok
            # idx_end_yr = np.ceil(event_end_days/365).astype(int) -1 #ok

            # Sum duration of events in each cell during 1 year
            # duration = event_end_days - event_start_days + 1  #ok
            avg_dur[idx_start_yr, ieta, ixi] +=  event_end_days - event_start_days + 1  #ok

            # Counts
            yr_count[idx_start_yr] += 1 # per year -- ok
            nb_mhw[ieta, ixi] += 1 #per cell -- ok

            # print("Start year: ", idx_start_yr)
            # print("End year: ", idx_end_yr)
            # print("duration: ", duration)
            # print("Count yearly: ", yr_count)
            # print("Count cell: ", nb_mhw[ieta, ixi])
            # print("BEFORE Sum duration: ", avg_dur[:, ieta, ixi])

        # Compute avg duration per year
        valid_years = yr_count > 0
        avg_dur[:, ieta, ixi][valid_years] /= yr_count[valid_years]
        # for iyear in range(nyears):
        #     if yr_count[iyear] > 0:
        #         avg_dur[iyear, ieta, ixi] = np.divide(avg_dur[iyear, ieta, ixi], yr_count[iyear])
        #     else:
        #         avg_dur[iyear, ieta, ixi] = 0  
        # print("AFTER Sum duration: ", avg_dur[:, ieta, ixi])
    
        # print(f'Processing for xi {ixi} done')
    print(f'Processing for eta {ieta} done')





# Into Datasets
ds_avg = xr.Dataset(
    data_vars=dict(
        avg_dur=(["years", "eta_rho", "xi_rho"], avg_dur),
        nb_mhw = (["eta_rho", "xi_rho"], nb_mhw)
    ),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),  # (434, 1442)
    ),
    attrs={
        'avg_dur': "Averaged yearly duration of MHW event (90th perc)",
        'nb_mhw': "Number of events in each grid cell over the 40yrs period"
    }
)
    
# Save output
output_file= os.path.join(output_path, f"mhw_avg_duration_yearly.nc")
if not os.path.exists(output_file):
    ds_avg.to_netcdf(output_file, mode='w')

# %%

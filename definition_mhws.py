#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 17 Feb 09:32:05 2025

Detect MHW in ROMS-SO using absolute threshold

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import sys
import os
import warnings
import copy as cp
import gsw
import xarray as xr
import numpy as np
import seaborn as sns

import pandas as pd
import geopandas as gpd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib
# matplotlib.use("WebAgg")  # Set the backend to TkAgg for separate window plotting
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import datetime
import cftime

import dask
from dask.distributed import Client
from joblib import Parallel, delayed

# %% -------------------------------- SETTINGS --------------------------------
# drift and bias corrected temperature files
path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/'
output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/absolute_temp_threshold/'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'
n_jobs = 30


# Temperature threshold 
temp_threshold = 5 #3.5 

# Choose averaging type
per_month = False
per_year = False
per_season = False

# Handling time
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) #defining months with days within a year
month_names = np.array(['Jan','Feb','Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov','Dec'])

season_bins = np.array([0, 90, 181, 273, 365]) #defining seasons with days within a year
season_names = np.array(['DJF (Summer)', 'MAM (Fall)', 'JJA (Winter)', 'SON (Spring)']) #southern ocean!

# %% -------------------------------- LOAD DATA --------------------------------
# file_path = os.path.join(path_mhw, f"temp_DC_BC_surface.nc") # all years - only surf layer (not chunked per latitude)
# ds = xr.open_dataset(file_path, chunks={"year": 1})[var][1:, 0:365, :, :] # Load one year at a time 
# ds = xr.open_dataset(file_path)[var][1:2, 0:365, :, :]  # only 1980 for test

# ------ PER LATITUDE
def detect_absolute_mhw(ieta, temp_threshold):

    print(f"Processing eta {ieta}...")

    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:,0:365,0:1,:].squeeze(axis=2) #Extracts daily data : only 40yr + consider 365 days per year + only surface
    
    # Deal with NANs values
    ds_original.values[np.isnan(ds_original.values)] = 0 # Set to 0 so that det = False at nans

    # Detect mhw - absolute threshold 
    mhw_events = np.greater(ds_original.values, temp_threshold)

    mhw_events_ds = xr.Dataset(
            data_vars=dict(
                mhw_events=(["years", "days", "xi_rho"], mhw_events), 
                ),
            coords=dict(
                lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
                lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
                ),
            attrs=dict(description="Detected " + f'T>{temp_threshold}°C' + " (boolean array)"),
                ) 
    
    # Save output
    output_file = os.path.join(output_path, f"det_{ieta}.nc")
    if not os.path.exists(output_file):
        mhw_events_ds.to_netcdf(output_file, mode='w')  


    return mhw_events
    
    

test = '/nfs/sea/work/jwongmeng/ROMS/analysis/detect/uptrend_poly2_95/cmhw/md50/det_all_morphed.nc'

results = Parallel(n_jobs=30)(delayed(detect_absolute_mhw)(ieta, temp_threshold) for ieta in range(0, neta)) # detects extremes for each latitude in parallel - results (list)

det = np.full((nyears, ndays, neta, nxi), False, dtype=np.bool_) #dim (40, 365, 1, 434, 1442)
for ieta in range(0,neta):

    det[:,:,ieta,:] = results[ieta]

det_ds = xr.Dataset(
    data_vars = dict(
        detect = (["years","days","eta_rho", "xi_rho"], det),
        ),
    coords = dict(
            lon_rho = (["eta_rho","xi_rho"],ds_roms.lon_rho.values), 
            lat_rho = (["eta_rho","xi_rho"],ds_roms.lat_rho.values)
            ),
    attrs = dict(description = "Detected " + 'T>3.5°C' + " (boolean array)")
)

det_ds.to_netcdf(os.path.join(output_path, "det_all_eta.nc"), mode='w') 

del det, results

# Select only 60°S
# spatial_domain = np.less(np.unique(ds_original.lat_rho.values), -60)
# print(f'for {ieta}: {spatial_domain}')

# if spatial_domain: 


# %% --- Average 

det_ds = xr.open_dataset(os.path.join(output_path, "det_all_eta.nc"))

# nc_files = [f for f in os.listdir(output_path) if f.startswith("det_") and f.endswith(".nc")]
# eta_indices = sorted([int(f.split("_")[1].split(".")[0]) for f in nc_files])

# Compute mean values
# --- YEARLY
mhw_per_year = det_ds['detect'].sum(dim=['days','eta_rho', 'xi_rho']).astype(np.float32)    
# --- MONTHLY
mhw_per_month = np.zeros((nyears, 12), dtype=np.float32)  # (years, months)
for i in range(12):  # Loop over months
    mhw_per_month[:, i] = det_ds['detect'].isel(days=slice(month_days[i], month_days[i+1])).sum(dim=['days', 'eta_rho', 'xi_rho'])

# --- SEASONALLY 
mhw_per_season = np.zeros((nyears, 4), dtype=np.float32)  # (years, seasons)
for i in range(4):  # Loop over seasons
    mhw_per_season[:, i] = det_ds['detect'].isel(days=slice(season_bins[i], season_bins[i+1])).sum(dim=['days', 'eta_rho', 'xi_rho'])

# Plot for each season
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

for i, ax in enumerate(axes.flatten()):
    ax.plot(mhw_per_year.years.values, mhw_per_season[:, i], marker='o', linestyle='-')

    # Labels and title
    ax.set_xlabel("Years")
    ax.set_ylabel("Total MHW Events")
    ax.set_title(f"Marine Heatwave (MHW) Events - {season_names[i]}")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# %% -------------------------------- PLOTS --------------------------------
def plot_map(mhw_data, time_index, time_label):
        """Function to plot the map for a given time index and label."""

        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

        # Circular map boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        # Plot temperature events
        pcolormesh = mhw_data.isel(**{time_label: time_index}).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            x="lon_rho", y="lat_rho",
            add_colorbar=False, 
            cmap=mcolors.ListedColormap(['#BBC6A9', '#9B1C1C']),  # Grey for temp < 3.5°C, Red for temp > 3.5°C
            vmin=0, vmax=1  
        )

        # Add features
        ax.coastlines(color='black', linewidth=1.5, zorder=1)
        ax.add_feature(cfeature.LAND, zorder=2,  facecolor='lightgray')
        ax.set_facecolor('lightgrey')

        # Legend
        legend_elements = [
            Line2D([0], [0], color='#BBC6A9', lw=6, label='Temp < 3.5°C'),
            Line2D([0], [0], color='#BE2323', lw=6, label='Temp > 3.5°C')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=14, borderpad=0.8, frameon=True, bbox_to_anchor=(-0.05, -0.05))
        
        # Convert time index to label
        if time_label == "month":
            time_str = month_names[time_index-1]
        elif time_label == "season":
            time_str = season_names[time_index-1]
        else:
            time_str = time_index  # Keep numerical index for year and day

        # Title
        ax.set_title(f"Temperature above 3.5°C \n{time_str} - {year}", fontsize=20, pad=30)

        plt.tight_layout()
        plt.show()


# Plot based on selection
if per_month:
    plot_map(xr.open_dataset(os.path.join(output_path, f"mhw_events_monthly_{2001}.nc"))[var], 
            time_index=0, time_label="month")
elif per_year:
    plot_map(xr.open_dataset(os.path.join(output_path, f"mhw_events_yearly_{2001}.nc"))[var],
            time_index=0, time_label="year")
elif per_season:
    plot_map(xr.open_dataset(os.path.join(output_path, f"mhw_events_seasonal_{2001}.nc"))[var], 
            time_index=2, time_label="season")
else:
    plot_map(xr.open_dataset(os.path.join(output_path, f"mhw_events_daily_{2001}.nc"))[var], 
            time_index=10, time_label="day")

# %% ---Number of days > 3.5°T
data = xr.open_dataset(os.path.join(output_path, f"mhw_events_daily_{1980}.nc"))[var]
extreme_days_count = data.sum(dim="day")

print(f"Minimum extreme days: {extreme_days_count.min().item()}")
print(f"Maximum extreme days: {extreme_days_count.max().item()}")

# Plot
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Plot number of days
pcolormesh = extreme_days_count.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                                                x="lon_rho", y="lat_rho",
                                                cmap='magma', add_colorbar=True)

# Land features
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2,  facecolor='lightgray')
ax.set_facecolor('lightgrey')

# Title
ax.set_title(f"Number of days with temperature > 3.5°C \n Year {extreme_days_count.year.item()}", fontsize=20, pad=30)

plt.tight_layout()
plt.show()

# %%

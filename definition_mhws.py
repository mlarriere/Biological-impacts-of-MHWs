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
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/joel_codes"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())


# drift and bias corrected temperature files
path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/'
output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/absolute_temp_threshold/'
joel_path ='/home/jwongmeng/work/ROMS/scripts/mhw_krill/'

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
temp_threshold = 3.5 

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

    # -- TEST --
    # mean_temp = ds_original.mean(dim=['xi_rho'])
    # Save the plot
    # plt.figure()
    # mean_temp.plot()
    # plt.title(f"Mean Temperature for eta {ieta}")
    # plt.savefig(f"outputs/mhw_plot_eta_{ieta}.png")  # Save plot as an image
    # plt.close()  # Close the figure to free memory
    # -----------

    # Detect mhw - absolute threshold 
    mhw_events = np.greater(ds_original.values, temp_threshold)

    # Save output
    output_file = os.path.join(output_path, f"det_{ieta}.nc")
    if not os.path.exists(output_file):
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
    
        mhw_events_ds.to_netcdf(output_file, mode='w')  


    return mhw_events
    
    

test = '/nfs/sea/work/jwongmeng/ROMS/analysis/detect/uptrend_poly2_95/cmhw/md50/det_all_morphed.nc'

results = Parallel(n_jobs=30)(delayed(detect_absolute_mhw)(ieta, temp_threshold) for ieta in range(0, neta)) # detects extremes for each latitude in parallel - results (list)

# %% Aggregating the different eta values
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

# Save output
output_file = os.path.join(output_path, f"det_all_eta.nc")
if not os.path.exists(output_file):
    det_ds.to_netcdf(output_file, mode='w') 

del det, results

# %% Spatial Average
det_ds = xr.open_dataset(os.path.join(output_path, "det_all_eta.nc"))
det_ds['years'] = xr.DataArray(np.arange(1980, 2020), dims=["years"], coords={"years": np.arange(1980, 2020)}) #assign years values


# Select only 60°S
spatial_domain = np.less(np.unique(det_ds.lat_rho.values), -60) #shape: (434,), i.e. (eta)
spatial_domain_reshaped = np.repeat(spatial_domain[:, np.newaxis], det_ds.lat_rho.shape[1], axis=1) #shape: (434, 1442), i.e. (eta, xi)
mask_da = xr.DataArray(spatial_domain_reshaped, dims=["eta_rho", "xi_rho"])
detected_south_of_60 = det_ds.detect.where(mask_da, drop=True)

del det_ds, spatial_domain, spatial_domain_reshaped, mask_da

# %% --- Averages
# ------ Keep spatial dimensions ------
# YEARLY
mhw_avg_yr_spatial = detected_south_of_60.sum(dim=['days']).astype(np.float32)   

# MONTHLY
mhw_avg_mth_spatial = xr.DataArray(
    np.zeros((nyears, 12, detected_south_of_60.shape[2], detected_south_of_60.shape[3]), dtype=np.float32),
    dims=["years", "month", "eta_rho", "xi_rho"],
    coords = detected_south_of_60.coords
    )
for i in range(12):  # Loop over months
    mhw_avg_mth_spatial[:, i, :, :] = detected_south_of_60.isel(days=slice(month_days[i], month_days[i+1])).sum(dim=['days'])

# SEASONALLY 
mhw_avg_sn_spatial = xr.DataArray(
    np.zeros((nyears, 4, detected_south_of_60.shape[2], detected_south_of_60.shape[3]), dtype=np.float32),
    dims=["years", "season", "eta_rho", "xi_rho"],
    coords = detected_south_of_60.coords
    )
for i in range(4):  # Loop over seasons
    mhw_avg_sn_spatial[:, i, :, :] = detected_south_of_60.isel(days=slice(season_bins[i], season_bins[i+1])).sum(dim=['days'])

# ------ Remove spatial dimensions ------
# YEARLY
mhw_per_year = mhw_avg_yr_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)    

# MONTHLY
mhw_per_month = xr.DataArray(np.zeros((nyears, 12), dtype=np.float32),
                             dims=["years", "month"]
                            )
for i in range(12):  # Loop over months
    mhw_per_month[:, i] = mhw_avg_mth_spatial.isel(month=i).sum(dim=['eta_rho', 'xi_rho'])

# SEASONALLY
mhw_per_season = xr.DataArray(np.zeros((nyears, 4), dtype=np.float32),
                             dims=["years", "season"]
                            )
for i in range(4):  # Loop over seasons
    mhw_per_season[:, i] = mhw_avg_sn_spatial.isel(season=i).sum(dim=['eta_rho', 'xi_rho'])
mhw_per_season['years'] = xr.DataArray(np.arange(1980, 2020), dims=["years"], coords={"years": np.arange(1980, 2020)}) #assign years values

# %% -------------------------------- PLOTS --------------------------------
# -------------------- SPATIAL
# Define variable to plot
years = 1980
mhw_data_toplot = mhw_avg_sn_spatial.sel(years=years)
time_index = 3
time_label="season" # "month", "years"

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
pcolormesh = mhw_data_toplot.isel(**{time_label: time_index}).plot.pcolormesh(
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
    Line2D([0], [0], color='#BBC6A9', lw=6, label=f'Temp < {temp_threshold}°C'),
    Line2D([0], [0], color='#BE2323', lw=6, label=f'Temp > {temp_threshold}°C')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=14, borderpad=0.8, frameon=True, bbox_to_anchor=(-0.05, -0.05))

# Convert time index to label
if time_label == "month":
    time_str = month_names[time_index-1]
elif time_label == "season":
    time_str = season_names[time_index-1]
else:
    time_str = time_label  # Keep numerical index for year and day

# Title
ax.set_title(f"Temperature above {temp_threshold}°C \n{time_str} - {years}", fontsize=20, pad=30)

plt.tight_layout()
plt.show()


# -------------------- TIME SERIES
# Years
plt.figure(figsize=(10, 5))
plt.plot(mhw_per_year.years.values, mhw_per_year.values, linewidth=2, color='#1B4079')
plt.fill_between(mhw_per_year.years.values, mhw_per_year.values, alpha=0.3, color='#1B4079')
plt.xlabel('Years')
plt.ylabel('Counts')
plt.title(f'Cumulative Events T>{temp_threshold}°C \nTime Series - Yearly')
plt.show()


# Seasons
plt.figure(figsize=(10, 5))
for season in range(4): # Loop over all seasons to plot each one
    mhw_data_toplot = mhw_per_season.sel(season=season)
    plt.plot(mhw_data_toplot.years.values, mhw_data_toplot.values, label=f'{season_names[season]}', linewidth=2) # Plot line 
    plt.fill_between(mhw_data_toplot.years.values, mhw_data_toplot.values, alpha=0.3) # Fill area under the curve

# Labels, title, and legend
plt.xlabel('Years')
plt.ylabel('Counts')
plt.title(f'Cumulative Events T>{temp_threshold}°C \nTime Series - Seasonally')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
plt.show()







# %% ---Number of days > 3.5°T
# data = xr.open_dataset(os.path.join(output_path, f"mhw_events_daily_{1980}.nc"))[var]
# extreme_days_count = data.sum(dim="day")

# print(f"Minimum extreme days: {extreme_days_count.min().item()}")
# print(f"Maximum extreme days: {extreme_days_count.max().item()}")

# # Plot
# plt.figure(figsize=(10, 10))
# ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
# ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# # Circular map boundary
# theta = np.linspace(0, 2 * np.pi, 100)
# center, radius = [0.5, 0.5], 0.5
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * radius + center)
# ax.set_boundary(circle, transform=ax.transAxes)

# # Plot number of days
# pcolormesh = extreme_days_count.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
#                                                 x="lon_rho", y="lat_rho",
#                                                 cmap='magma', add_colorbar=True)

# # Land features
# ax.coastlines(color='black', linewidth=1.5, zorder=1)
# ax.add_feature(cfeature.LAND, zorder=2,  facecolor='lightgray')
# ax.set_facecolor('lightgrey')

# # Title
# ax.set_title(f"Number of days with temperature > 3.5°C \n Year {extreme_days_count.year.item()}", fontsize=20, pad=30)

# plt.tight_layout()
# plt.show()

# %%

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

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
joel_path ='/home/jwongmeng/work/ROMS/scripts/mhw_krill/' #codes joel

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'
n_jobs = 30


# - Choose baseline type
# baseline = 'none' # absolute_threshold
baseline = 'fixed1980' 

if baseline=='none':
    temp_threshold = 3.5
    description = "Detected events: " + f'T>{temp_threshold}°C' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/absolute_temp_threshold/'
if baseline=='fixed1980':
    description = "Detected events" + f'T°C > T1980 (fixed baseline)' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline/'


# -- Handling time
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) #defining months with days within a year
month_names = np.array(['Jan','Feb','Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov','Dec'])

season_bins = np.array([0, 90, 181, 273, 365]) #defining seasons with days within a year
season_names = np.array(['DJF (Summer)', 'MAM (Fall)', 'JJA (Winter)', 'SON (Spring)']) #southern ocean!

# %% -------------------------------- LOAD DATA --------------------------------
# file_path = os.path.join(path_mhw, f"temp_DC_BC_surface.nc") # all years - only surf layer (not chunked per latitude)
# ds = xr.open_dataset(file_path, chunks={"year": 1})[var][1:, 0:365, :, :] # Load one year at a time 
# ds = xr.open_dataset(file_path)[var][1:2, 0:365, :, :]  # only 1980 for test

# ------ PER LATITUDE
def detect_absolute_mhw(ieta, temp_threshold, baseline):

    print(f"Processing eta {ieta}...")

    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:,0:365,0:1,:].squeeze(axis=2) #Extracts daily data : only 40yr + consider 365 days per year + only surface
    print(np.unique(ds_original.lat_rho.values))

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

    # -- Detect mhw 
    if baseline=='none':
        temp_threshold = 3.5 # Absolute threshold  

    if baseline=='fixed1980':
        temp_threshold = ds_original[0, :, :].values  # Year index 0 corresponds to 1980
        temp_threshold[np.isnan(temp_threshold)] = 0  #deal with Nan values
    
    # Boolean - MHW events detection
    mhw_events = np.greater(ds_original.values, temp_threshold) 

    # Compute MHW intensity (temperature anomaly)
    mhw_intensity = ds_original.values - temp_threshold
    mhw_intensity[~mhw_events] = np.nan  # Mask non-MHW values


    # Save output
    output_file = os.path.join(output_path, f"det_{ieta}.nc")
    if not os.path.exists(output_file):
        mhw_events_ds = xr.Dataset(
            data_vars=dict(
                mhw_events=(["years", "days", "xi_rho"], mhw_events), 
                mhw_intensity=(["years", "days", "xi_rho"], mhw_intensity), 
                ),
            coords=dict(
                lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
                lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
                ),
            attrs=dict(description=description),
                ) 
    
        mhw_events_ds.to_netcdf(output_file, mode='w')  


    return mhw_events, mhw_intensity #at the end - list of "shape": eta, years, days, xi
    
# Calling function
mhw_events, mhw_intensity = Parallel(n_jobs=30)(delayed(detect_absolute_mhw)(ieta, temp_threshold, baseline) for ieta in range(0, neta)) # detects extremes for each latitude in parallel - results (list)

# %% Aggregating the different eta values
det = np.full((nyears, ndays, neta, nxi), False, dtype=np.bool_) #dim (40, 365, 1, 434, 1442)
for ieta in range(0,neta):

    det[:,:,ieta,:] = results[ieta]

det_ds = xr.Dataset(
    data_vars = dict(
        detect = (["years","days","eta_rho", "xi_rho"], det),
        mhw_intensity=(["years", "days", "xi_rho"], mhw_intensity)
        ),
    coords = dict(
            lon_rho = (["eta_rho","xi_rho"],ds_roms.lon_rho.values), 
            lat_rho = (["eta_rho","xi_rho"],ds_roms.lat_rho.values)
            ),
    attrs = dict(description = description)
)

# Save output
output_file = os.path.join(output_path, f"det_all_eta.nc")
if not os.path.exists(output_file):
    det_ds.to_netcdf(output_file, mode='w') 

# del det, results

# %% Spatial Average
det_ds = xr.open_dataset(os.path.join(output_path, "det_all_eta.nc"))
det_ds['years'] = xr.DataArray(np.arange(1980, 2020), dims=["years"], coords={"years": np.arange(1980, 2020)}) #assign years values


# Select only 60°S
spatial_domain = np.less(np.unique(det_ds.lat_rho.values), -60) #shape: (434,), i.e. (eta)
spatial_domain_reshaped = np.repeat(spatial_domain[:, np.newaxis], det_ds.lat_rho.shape[1], axis=1) #shape: (434, 1442), i.e. (eta, xi)
mask_da = xr.DataArray(spatial_domain_reshaped, dims=["eta_rho", "xi_rho"])
detected_south_of_60 = det_ds.detect.where(mask_da, drop=True) # long computing

# del det_ds, spatial_domain, spatial_domain_reshaped

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
years = 1981
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
    add_colorbar=True, 
    cmap=mcolors.ListedColormap(['#BBC6A9', '#9B1C1C']),  # Grey for temp < 3.5°C, Red for temp > 3.5°C
    vmin=0, vmax=1  
)

# Add features
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2,  facecolor='lightgray')
ax.set_facecolor('lightgrey')

# # Legend
# legend_elements = [
#     Line2D([0], [0], color='#BBC6A9', lw=6, label=f'Temp < {temp_threshold}°C'),
#     Line2D([0], [0], color='#BE2323', lw=6, label=f'Temp > {temp_threshold}°C')
# ]
# ax.legend(handles=legend_elements, loc='lower left', fontsize=14, borderpad=0.8, frameon=True, bbox_to_anchor=(-0.05, -0.05))

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
# plt.figure(figsize=(10, 5))
# plt.plot(mhw_per_year.years.values, mhw_per_year.values, linewidth=2, color='#1B4079')
# plt.fill_between(mhw_per_year.years.values, mhw_per_year.values, alpha=0.3, color='#1B4079')
# plt.xlabel('Years')
# plt.ylabel('Counts')
# plt.title(f'Cumulative Events T>{temp_threshold}°C \nTime Series - Yearly')
# plt.show()


# Seasons
plt.figure(figsize=(10, 5))
# plt.plot(mhw_per_year.years.values, mhw_per_year.values, linewidth=1, color='#47455F', label='Whole Year')
plt.fill_between(mhw_per_year.years.values, mhw_per_year.values, alpha=0.2, color='#47455F')

season_colors = ['#F39C12', '#E74C3C', '#4A90E2', '#2ECC71']  # summer, fall, winter, spring

for season in range(4):  # Loop over all seasons to plot each one
    mhw_data_toplot = mhw_per_season.sel(season=season)
    plt.plot(mhw_data_toplot.years.values, mhw_data_toplot.values,
             color=season_colors[season],
             linewidth=1)  
    plt.fill_between(mhw_data_toplot.years.values, mhw_data_toplot.values, alpha=0.3, color=season_colors[season])  

# Labels, title, and legend
plt.xlabel('Years')
plt.ylabel('Counts')
plt.xlim((1980, 2019))
plt.ylim((0, np.max(mhw_per_year).values))
plt.title(f'Cumulative Events (T>{temp_threshold}°C) \nSouth of 60°S')

# Custom legend
season_labels = [f'{season_names[i]}' for i in range(4)] + ['Whole Year'] 
handles = [plt.Line2D([0], [0], color=season_colors[i], lw=6) for i in range(4)] + [plt.Line2D([0], [0], color='grey', lw=6)]  # Added handle for 'All Years'
plt.legend(handles=handles, labels=season_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, columnspacing=1)

plt.show()



# %% Area 
area_SO = xr.open_dataset(os.path.join(joel_path, "area.nc")) # Nans values - Land
area_SO_surf = area_SO.isel(z_t=0) #select only surface
area_SO_surf_60S = area_SO_surf.area.where(mask_da, drop=True) # south of 60°S

mhw_normalized_by_area = np.divide(mhw_avg_sn_spatial, area_SO_surf_60S)

total_ocean_area_60S = area_SO_surf_60S.sum(dim=['eta_rho', 'xi_rho']) #total ocean area south of 60°S
mhw_total_area = (mhw_normalized_by_area * area_SO_surf_60S).sum(dim=['eta_rho', 'xi_rho']) # MHW area

# Normalize by total ocean area to get the fraction
mhw_fraction_per_year = (mhw_total_area.sum(dim="season") / total_ocean_area_60S)
mhw_fraction_per_season = mhw_total_area / total_ocean_area_60S


# --- PLOT
year = 2019
season=1  
mhw_data_toplot = mhw_normalized_by_area.sel(years=year, season= season)

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Plot event normalized by area
pcolormesh = mhw_data_toplot.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=True, 
    cmap='plasma',  # Grey for low values, Red for high values
    cbar_kwargs={'label': 'MHW normalized by area'}  # Colorbar title

)
# Features
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgray')
ax.set_facecolor('lightgrey')
ax.set_title(f"Temperature above {temp_threshold}°C \n{season_names[season]} - {years}", fontsize=20, pad=30)

plt.show()

# -- AVG 
# Seasons
plt.figure(figsize=(10, 5))
plt.fill_between(mhw_fraction_per_year.years.values, mhw_fraction_per_year, alpha=0.2, color='#47455F')

season_colors = ['#F39C12', '#E74C3C', '#4A90E2', '#2ECC71']  # summer, fall, winter, spring

for season in range(4):  # Loop over all seasons to plot each one
    mhw_data_toplot = mhw_fraction_per_season.sel(season=season)
    plt.plot(mhw_data_toplot.years.values, mhw_data_toplot.values,
             color=season_colors[season],
             linewidth=1)  
    plt.fill_between(mhw_data_toplot.years.values, mhw_data_toplot.values, alpha=0.3, color=season_colors[season])  

# Labels, title, and legend
plt.xlabel('Years')
plt.ylabel('Area fraction [%]')
plt.xlim((1980, 2019))
plt.ylim((0, np.max(mhw_fraction_per_year).values))
plt.title(f'Cumulative Events (T>{temp_threshold}°C) \nSouth of 60°S')

# Custom legend
season_labels = [f'{season_names[i]}' for i in range(4)] + ['Whole Year'] 
handles = [plt.Line2D([0], [0], color=season_colors[i], lw=6) for i in range(4)] + [plt.Line2D([0], [0], color='grey', lw=6)]  # Added handle for 'All Years'
plt.legend(handles=handles, labels=season_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, columnspacing=1)

plt.show()





# %%

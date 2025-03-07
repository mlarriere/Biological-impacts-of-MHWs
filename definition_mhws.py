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

import matplotlib.cm as cm
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
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
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


# -- Define Climatology - baseline type
# baseline = 'fixed1980' 
baseline = 'fixed30yrs' 

if baseline=='fixed1980':
    description = "Detected events" + f'T°C > T°C 1980' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline1980/'
if baseline=='fixed30yrs':
    description = "Detected events" + f'T°C > climatology (1980-2010)' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold
percentile = 90 
pmod = 'perc' + str(percentile)


# -- Handling time
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) #defining months with days within a year
month_names = np.array(['Jan','Feb','Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov','Dec'])

season_bins = np.array([0, 90, 181, 273, 365]) #defining seasons with days within a year
season_names = np.array(['DJF (Summer)', 'MAM (Fall)', 'JJA (Winter)', 'SON (Spring)']) #southern ocean!
# %% SST plot
file_path = os.path.join(path_mhw, f"temp_DC_BC_surface.nc") # all years - only surf layer (not chunked per latitude)
ds = xr.open_dataset(file_path)[var][1:, 0:365, :, :]  # only 1980 for test
ds_sst = ds.mean(dim=['eta_rho', 'xi_rho'])


fig, ax = plt.subplots(figsize=(10, 5))
cmap = cm.get_cmap("YlOrBr", len(years)) 
for i, year in enumerate(ds_sst.year.values):
    ds_sst.sel(year=year).plot(color=cmap(i / len(years)), label=str(year))

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min(years), vmax=max(years)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="Year")

# Set ticks 
ticks = [year for year in range(min(years), max(years)+1) if (year % 5 == 0) or (year == 2019)]
cbar.set_ticks(ticks)  
cbar.set_ticklabels([str(year) for year in ticks]) 
# Final touches
ax.set_xlabel('Day of the Year')
ax.set_ylabel('SST (°C)')
ax.set_title('Sea Surface Temperature in the Southern Ocean')
plt.tight_layout()
plt.show()



# %% test 90th percentile
fn = path_mhw + file_var + 'eta200.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
ds_original = xr.open_dataset(fn)[var][1:,0:365,0:1,:].squeeze(axis=2) #dim: (year: 40, day: 365, xi_rho: 1442)

climatology_sst = ds_original.sel(year=slice(1980, 2009)).mean(dim="year") # 30-year mean
mask = np.where(np.isnan(ds_original[0,0,:]), False, True) #mask representing non-Nan values (True) nan, shape: xi_rho: 1442

# window_size = 11
# ds_original.rolling(day=window_size, min_periods=1, center=True).construct('window')
threshold=np.full((ndays, nxi),np.nan,dtype=np.float32)

for dy in range(0,ndays-1): #days index going from 0 to 364
    if dy<=4:
        window_sst = climatology_sst.isel(day=np.concatenate([np.arange(360+dy,365,1), np.arange(0, dy+6,1)]))
    elif dy>=360:
        window_sst = climatology_sst.isel(day=np.concatenate([np.arange(dy-5, 365,1), np.arange(0,dy-359,1)]))
    else:
        window_sst = climatology_sst.isel(day=np.arange(dy-5, dy+6, 1))

    threshold[dy,:] = np.percentile(window_sst, 90, axis=(0,1))

threshold = np.where(mask[None,:]==True, threshold, np.nan)

# Climatology
output_path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
def calculate_climSST(ieta, baseline):

    print(f"Processing eta {ieta}...")

    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:31,0:365,:,:] #Extracts daily data : only 40yr + consider 365 days per year
    print(np.unique(ds_original.lat_rho.values))

    # Deal with NANs values
    ds_original.values[np.isnan(ds_original.values)] = 0 # Set to 0 so that det = False at nans

    if baseline == 'fixed1980':
        climatology_sst = ds_original[0, :, :, :]  # Only 1980
    elif baseline == 'fixed30yrs':
        climatology_sst = ds_original.sel(year=slice(1980, 2009)).mean(dim="year") # 30-year mean

    # Save output
    output_file_clim = os.path.join(output_path_clim, f"climatology_sst{ieta}.nc")
    if not os.path.exists(output_file_clim):
        climatology_sst = xr.Dataset(
            data_vars= dict(clim_sst =(["days", "z_rho", "xi_rho"], climatology_sst.values)),
            coords=dict(
                lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
                lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
                ),
            attrs=dict(description='Climatology SST - baseline 1980-2009 (30yrs)'),
                ) 

    climatology_sst.to_netcdf(output_file_clim, mode='w')  

# Calling function
results = Parallel(n_jobs=30)(delayed(calculate_climSST)(ieta, baseline) for ieta in range(0, neta)) # calculate climatology for each latitude in parallel - results (list) - ~12min computing

# %% -------------------------------- LOAD DATA --------------------------------
# ds = xr.open_dataset(file_path, chunks={"year": 1})[var][1:, 0:365, :, :] # Load one year at a time 
# ds = xr.open_dataset(file_path)[var][1:2, 0:365, :, :]  # only 1980 for test

# ------ PER LATITUDE
def detect_absolute_mhw(ieta, baseline):

    print(f"Processing eta {ieta}...")

    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:,0:365,0:1,:].squeeze(axis=2) #Extracts daily data : only 40yr + consider 365 days per year + only surface
    print(np.unique(ds_original.lat_rho.values))

    # Deal with NANs values
    ds_original.values[np.isnan(ds_original.values)] = 0 # Set to 0 so that det = False at nans

    # # -- TEST --
    # mean_temp = ds_original.mean(dim=['xi_rho'])
    # # Save the plot
    # plt.figure()
    # mean_temp.plot()
    # plt.title(f"Mean Temperature for eta {ieta}")
    # plt.savefig(f"outputs/mhw_plot_eta_{ieta}.png")  # Save plot as an image
    # plt.close()  # Close the figure to free memory
    # -----------

    # -- Define Climatology
    if baseline == 'fixed1980':
        climatology_sst = ds_original[0, :, :]  # Only 1980
    elif baseline == 'fixed30yrs':
        climatology_sst = ds_original.sel(year=slice(1980, 2009)).mean(dim="year") # 30-year mean

    # -- Define Thresholds
    absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold

    # -- MHW events detection
    mhw_events = {}
    for thresh in absolute_thresholds:
        mhw_events[thresh] = np.greater(ds_original.values, thresh)  #Boolean

    # -- MHW intensity 
    mhw_intensity = ds_original.values - climatology_sst.values # Anomaly relative to climatology
    # mhw_intensity[~mhw_events] = np.nan  # Mask non-MHW values


    # Save output
    output_file_eta = os.path.join(output_path, f"det_{ieta}.nc")
    if not os.path.exists(output_file_eta):
        mhw_events_ds = xr.Dataset(
            data_vars=dict(
                **{
                    f"mhw_events{thresh}": (["years", "days", "xi_rho"], mhw_events[thresh]) for thresh in absolute_thresholds},
                # mhw_events=(["years", "days", "xi_rho"], mhw_events), 
                mhw_intensity=(["years", "days", "xi_rho"], mhw_intensity), 
                ),
            coords=dict(
                lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
                lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
                ),
            attrs=dict(description=description),
                ) 
    
        mhw_events_ds.to_netcdf(output_file_eta, mode='w')  


    return mhw_events, mhw_intensity #at the end - list of "shape": eta, years, days, xi
    
# Calling function
results = Parallel(n_jobs=30)(delayed(detect_absolute_mhw)(ieta, baseline) for ieta in range(0, neta)) # detects extremes for each latitude in parallel - results (list) - ~12min computing

# %% Aggregating the different eta values
# Extract detection and intensity of mhw
mhw_events, mhw_intensity = zip(*results) 

# Initialization 
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold

# det = np.full((nyears, ndays, neta, nxi), False, dtype=np.bool_) #dim (40, 365, 1, 434, 1442)
det = {thresh: np.full((nyears, ndays, neta, nxi), False, dtype=np.bool_) for thresh in absolute_thresholds}  # Shape: (40, 365, neta, nxi)
det_intensity = np.full((nyears, ndays, neta, nxi), False, dtype=np.float32) #dim (40, 365, 1, 434, 1442)

# Loop over neta and write all eta in same Datatset - aggregation 
for ieta in range(0,neta):
    for thresh in absolute_thresholds:
        det[thresh][:, :, ieta, :] = mhw_events[ieta][thresh]  # Store detection for each threshold
    # det[:,:,ieta,:] = mhw_events[ieta]
    det_intensity[:,:,ieta,:] = mhw_intensity[ieta]

det_ds = xr.Dataset(
    data_vars = dict(
         **{f"mhw_events{thresh}": (["years", "days", "eta_rho", "xi_rho"], det[thresh]) for thresh in absolute_thresholds},
        # mhw_events = (["years","days","eta_rho", "xi_rho"], det),
        mhw_intensity=(["years", "days", "eta_rho", "xi_rho"], det_intensity)
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

# %% ------------------ Spatial Average
det_ds = xr.open_dataset(os.path.join(output_path, "det_all_eta.nc"))
det_ds['years'] = xr.DataArray(np.arange(1980, 2020), dims=["years"], coords={"years": np.arange(1980, 2020)}) #assign years values

variables = [f"mhw_events{thresh}" for thresh in absolute_thresholds] + ['mhw_intensity']

# Select only 60°S
spatial_domain = np.less(np.unique(det_ds.lat_rho.values), -60) #shape: (434,), i.e. (eta)
spatial_domain_reshaped = np.repeat(spatial_domain[:, np.newaxis], det_ds.lat_rho.shape[1], axis=1) #shape: (434, 1442), i.e. (eta, xi)
mask_da = xr.DataArray(spatial_domain_reshaped, dims=["eta_rho", "xi_rho"])

# Absolute threshold = 1°C
det1deg_south_of_60 = det_ds.mhw_events1.where(mask_da, drop=True) # long computing - non Nans values in mhw_evnts and intensity
# Save output
output_file = os.path.join(output_path, f"det1deg.nc")
if not os.path.exists(output_file):
    det1deg_south_of_60.to_netcdf(output_file, mode='w') 

# Absolute threshold = 2°C
det2deg_south_of_60 = det_ds.mhw_events2.where(mask_da, drop=True) # long computing - non Nans values in mhw_evnts and intensity
# Save output
output_file = os.path.join(output_path, f"det2deg.nc")
if not os.path.exists(output_file):
    det2deg_south_of_60.to_netcdf(output_file, mode='w') 

# Absolute threshold = 3°C
det3deg_south_of_60 = det_ds.mhw_events3.where(mask_da, drop=True) # long computing - non Nans values in mhw_evnts and intensity
# Save output
output_file = os.path.join(output_path, f"det3deg.nc")
if not os.path.exists(output_file):
    det3deg_south_of_60.to_netcdf(output_file, mode='w') 

# Intensity
# intensity_south_of_60 = det_ds.mhw_intensity.where(mask_da, drop=True) # long computing - non Nans values in mhw_evnts and intensity




# def apply_spatial_mask(yr):
#     # Read only 1 year
#     ds= det_ds.isel(years=yr)

#     # Masking
#     masked_data = {var: ds[var].where(mask_da, drop=True) for var in variables}

#     # Create a new dataset with the masked values
#     detected_south_of_60 = xr.Dataset(
#         {var: (["days", "eta_rho", "xi_rho"], masked_data[var].data) for var in variables},
#         coords=dict(
#             lon_rho=(["eta_rho", "xi_rho"], masked_data['mhw_intensity'].lon_rho.values), #(231, 1442)
#             lat_rho=(["eta_rho", "xi_rho"], masked_data['mhw_intensity'].lat_rho.values)
#         ),
#         attrs=dict(description='Events detected south of 60°S'),
#     )

#     return detected_south_of_60

# Apply mask in parallel - about 10min 
# results = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr) for yr in range(0, nyears)) # spatial mask for each years in parallel
# combined_results = xr.concat(results, dim='years') # put back to original dimensions
# combined_results['years'] = xr.DataArray(np.arange(1980, 2020), dims=["years"], coords={"years": np.arange(1980, 2020)}) #assign years values


# del det_ds, spatial_domain, spatial_domain_reshaped

# %% ------ Averages
det1deg_south_of_60 = xr.open_dataset(os.path.join(output_path, "det1deg.nc"))
det2deg_south_of_60 = xr.open_dataset(os.path.join(output_path, "det2deg.nc"))
det3deg_south_of_60 = xr.open_dataset(os.path.join(output_path, "det3deg.nc"))

# --- YEARLY
# Absolute threshold = 1°C
# ------ Keep spatial dimensions ------
det1deg_sum_yr_spatial = det1deg_south_of_60.sum(dim=['days']).astype(np.float32) #cumulative MHW events per year and per cell Each grid cell can have between 0 (no MHW days in the year) and 365 (MHW present every day of the year)
det1deg_potential_mhw_spatial = (det1deg_sum_yr_spatial > 0).astype(np.float32) # Potential MHW presence (1 if at least one event occurs, else 0)
# ------ Remove spatial dimensions ------
det1deg_sum_yr = det1deg_sum_yr_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)    
det1deg_potential_mhw = det1deg_potential_mhw_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)    

# Retrieve max and min values
print(f"Maximum MHW Events (for thresh=1°C) in year {int(det1deg_potential_mhw.mhw_events1.idxmax().values)}")
print(f"Minimum MHW Events (for thresh=1°C) in year {int(det1deg_potential_mhw.mhw_events1.idxmin().values)}")

del det1deg_south_of_60

# Absolute threshold = 2°C
# ------ Keep spatial dimensions ------
det2deg_sum_yr_spatial = det2deg_south_of_60.sum(dim=['days']).astype(np.float32) #cumulative MHW events per year and per cell Each grid cell can have between 0 (no MHW days in the year) and 365 (MHW present every day of the year)
det2deg_potential_mhw_spatial = (det2deg_sum_yr_spatial > 0).astype(np.float32) # Potential MHW presence (1 if at least one event occurs, else 0)
# ------ Remove spatial dimensions ------
det2deg_sum_yr = det2deg_sum_yr_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)    
det2deg_potential_mhw = det2deg_potential_mhw_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)    

# Retrieve max and min values
print(f"Maximum MHW Events (for thresh=2°C) in year {int(det2deg_potential_mhw.mhw_events2.idxmax().values)}")
print(f"Minimum MHW Events (for thresh=2°C) in year {int(det2deg_potential_mhw.mhw_events2.idxmin().values)}")
del det2deg_south_of_60

# Absolute threshold = 3°C
# ------ Keep spatial dimensions ------
det3deg_sum_yr_spatial = det3deg_south_of_60.sum(dim=['days']).astype(np.float32) #cumulative MHW events per year and per cell Each grid cell can have between 0 (no MHW days in the year) and 365 (MHW present every day of the year)
det3deg_potential_mhw_spatial = (det3deg_sum_yr_spatial > 0).astype(np.float32) # Potential MHW presence (1 if at least one event occurs, else 0)
# ------ Remove spatial dimensions ------
det3deg_sum_yr = det3deg_sum_yr_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)    
det3deg_potential_mhw = det3deg_potential_mhw_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)    

# Retrieve max and min values
print(f"Maximum MHW Events (for thresh=3°C) in year {int(det3deg_potential_mhw.mhw_events3.idxmax().values)}")
print(f"Minimum MHW Events (for thresh=3°C) in year {int(det3deg_potential_mhw.mhw_events3.idxmin().values)}")

del det3deg_south_of_60


# --- MONTHLY
# mhw_avg_mth_spatial = xr.DataArray(
#     np.zeros((nyears, 12, detected_south_of_60.shape[2], detected_south_of_60.shape[3]), dtype=np.float32),
#     dims=["years", "month", "eta_rho", "xi_rho"],
#     coords = detected_south_of_60.coords
#     )
# for i in range(12):  # Loop over months
#     mhw_avg_mth_spatial[:, i, :, :] = detected_south_of_60.isel(days=slice(month_days[i], month_days[i+1])).sum(dim=['days'])

# --- SEASONALLY 
# season_nb_days = np.diff(season_bins)  # Number of days in each season

# det_sum_sn_spatial = xr.DataArray(
#     np.zeros((nyears, 4, detected_south_of_60.mhw_events.shape[2], detected_south_of_60.mhw_events.shape[3]), dtype=np.float32),
#     dims=["years", "season", "eta_rho", "xi_rho"],
#     coords = detected_south_of_60.coords
#     )
# intensity_avg_sn_spatial = xr.DataArray(
#     np.zeros((nyears, 4, detected_south_of_60.mhw_events.shape[2], detected_south_of_60.mhw_events.shape[3]), dtype=np.float32),
#     dims=["years", "season", "eta_rho", "xi_rho"],
#     coords = detected_south_of_60.coords
#     )

# for i in range(4):  # Loop over seasons
#     det_sum_sn_spatial[:, i, :, :] = combined_results.mhw_events1.isel(days=slice(season_bins[i], season_bins[i+1])).sum(dim=['days']) #cumulative MHW events per season and per cell
#     intensity_avg_sn_spatial[:, i, :, :] = combined_results.mhw_intensity.isel(days=slice(season_bins[i], season_bins[i+1])).sum(dim=['days'])/season_nb_days[i]   # mean MHW intensity per season and per cell
# %% -------------------- TIME SERIES
years = np.arange(1980, 2020)  
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, det1deg_potential_mhw.mhw_events1, marker='o', linestyle='-', color='#5A7854', linewidth=2, label="1°C threshold")
ax.plot(years, det2deg_potential_mhw.mhw_events2, marker='o', linestyle='-', color='#8780C6', linewidth=2, label="2°C threshold")
ax.plot(years, det3deg_potential_mhw.mhw_events3, marker='o', linestyle='-', color='#9B2808', linewidth=2, label="3°C threshold")

# Labels and title
ax.set_xlabel("Year")
ax.set_ylabel("Counts")
ax.set_title("Potential MHW presence - South 60°S", fontsize=14)
fig.text(0.5, 0.93, "Sum of the events: 1 if at least one event occurs in a cell, else 0",
         ha='center', fontsize=10, style='italic')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()


# %% -------------------------------- PLOTS --------------------------------
# -------------------- SPATIAL
# Define variable to plot
keys = ["det1deg_yearly", "det2deg_yearly", "det3deg_yearly"]
titles = ["1°C threshold", "2°C threshold", "3°C threshold"]
threshold_colors = ['#5A7854', '#8780C6', '#9B2808']
time_label="years" # season
time_idx = 3
years = 1986

datasets = {
#             # "intensity_yearly": intensity_avg_yr_spatial, 
            "det1deg_yearly": det1deg_potential_mhw_spatial.mhw_events1,
            "det2deg_yearly": det2deg_potential_mhw_spatial.mhw_events2,
            "det3deg_yearly": det3deg_potential_mhw_spatial.mhw_events3,
#             # "det2deg_yearly": det2deg_sum_yr_spatial,
#             # "det3deg_yearly": det3deg_sum_yr_spatial,
#             # "intensity_season": intensity_avg_sn_spatial, 
#             # "det_season": det_sum_sn_spatial
            }

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 15), 
                         subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)})


for ax, key, title, col in zip(axes, keys, titles, threshold_colors):
    mhw_data_toplot = datasets[key].sel(years=years)
    print(np.max(mhw_data_toplot.lon_rho.values)-np.min(mhw_data_toplot.lon_rho.values))

    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular map boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    pcolormesh = mhw_data_toplot.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            x="lon_rho", y="lat_rho",
            add_colorbar=False, vmin=0, vmax=1,
            cmap=plt.matplotlib.colors.ListedColormap(['lightgray', col])
        )
        
    
    # Add features
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')
 

    # Legend
    threshold = title.split(" ")[0]
    
    # Create a binary legend for the current threshold
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label=f'T°C < {threshold}',
               markerfacecolor='lightgray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label=f'T°C ≥ {threshold}',
               markerfacecolor=col, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=14,
              borderpad=0.8, frameon=True, bbox_to_anchor=(0.5, -0.15))
    
for ax in axes:
    # Atlantic-Pacific boundary (near Drake Passage)
    ax.plot([-70, -70], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Atlantic sector

    # Pacific-Indian boundary
    ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Pacific sector

    # Indian-Atlantic boundary
    ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Indian sector



plt.suptitle('Potential Presence of MHW - Absolute Thresholds', fontsize=20, y=0.8)
plt.tight_layout()
plt.show()

# %% --- Averages

# intensity_avg_yr = intensity_avg_yr_spatial.sum(dim=['eta_rho', 'xi_rho']).astype(np.float32)/365

# MONTHLY
# mhw_per_month = xr.DataArray(np.zeros((nyears, 12), dtype=np.float32),
#                              dims=["years", "month"]
#                             )
# for i in range(12):  # Loop over months
#     mhw_per_month[:, i] = mhw_avg_mth_spatial.isel(month=i).sum(dim=['eta_rho', 'xi_rho'])

# SEASONALLY
# det_sum_sn = xr.DataArray(np.zeros((nyears, 4), dtype=np.float32), dims=["years", "season"])
# intensity_avg_sn = xr.DataArray(np.zeros((nyears, 4), dtype=np.float32), dims=["years", "season"])

# for i in range(4):  # Loop over seasons
#     det_sum_sn[:, i] = det_sum_sn_spatial.isel(season=i).sum(dim=['eta_rho', 'xi_rho'])
#     intensity_avg_sn[:, i] = intensity_avg_sn_spatial.isel(season=i).sum(dim=['eta_rho', 'xi_rho'])/4

# det_sum_sn['years'] = xr.DataArray(np.arange(1980, 2020), dims=["years"], coords={"years": np.arange(1980, 2020)}) #assign years values
# intensity_avg_sn['years'] = xr.DataArray(np.arange(1980, 2020), dims=["years"], coords={"years": np.arange(1980, 2020)}) #assign years values




# %% Area 
area_SO = xr.open_dataset(os.path.join(joel_path, "area.nc")) # Area per cell (eta, xi), (Nans values??)
area_SO_surf = area_SO.area.isel(z_t=0)[:, 1:-1] #select only surface, area (view from top), kick out part overlapping
area_SO_surf_60S = area_SO_surf.where(mask_da, drop=True) # south of 60°S - shape: (eta_rho: 231, xi_rho: 1442)
area_SO_surf_60S.values[np.isnan(area_SO_surf_60S.values)] = 0 # Set to 0 so that det = False at nans

total_ocean_area_60S = area_SO_surf_60S.sum(dim=['eta_rho', 'xi_rho']) #total ocean area south of 60°S = 25 257 618 km2

# Assign area of cell to dataset
array_area = area_SO_surf_60S #shape: (231, 1442)
array_det1deg_mhw = det1deg_potential_mhw_spatial.mhw_events1 #shape: (40, 231, 1442)
array_det2deg_mhw = det2deg_potential_mhw_spatial.mhw_events2 #shape: (40, 231, 1442)
array_det3deg_mhw = det3deg_potential_mhw_spatial.mhw_events3 #shape: (40, 231, 1442)

# array_det_mhw_sn = np.zeros((40, 4, 231, 1442))
# for i in range(4):
#     seasonal_slice = detected_south_of_60.mhw_events.isel(days=slice(season_bins[i], season_bins[i+1])).values  # Shape: (40, 92, 231, 1442)
#     array_det_mhw_sn[:, i, :, :] = np.mean(seasonal_slice, axis=1)      # Take the mean over the days in each season


# Multiply
combine_yr_1deg = np.einsum('ijk,jk -> i', array_det1deg_mhw, array_area) #shape: (40)
combine_yr_2deg = np.einsum('ijk,jk -> i', array_det2deg_mhw.values, array_area.values) #shape: (40)
combine_yr_3deg = np.einsum('ijk,jk -> i', array_det3deg_mhw.values, array_area.values) #shape: (40)
# combine_sn = np.einsum('ijkw,kw', array_det_mhw_sn, array_area.values) #shape: (40)
det1deg_area_frac_yr = np.divide(combine_yr_1deg, total_ocean_area_60S.values)
det2deg_area_frac_yr = np.divide(combine_yr_2deg, total_ocean_area_60S.values)
det3deg_area_frac_yr = np.divide(combine_yr_3deg, total_ocean_area_60S.values)
# combine_div_sn = np.divide(combine_sn, total_ocean_area_60S.values)

# det_area_frac_yr = np.mean(combine_div_sn, axis=1)   

# Spatially averaged MHW cumulative intensity
# mhw_normalized_by_area_sn = np.divide(intensity_avg_sn, total_ocean_area_60S.values) *100 #in %
# mhw_normalized_by_area_yr = np.divide(det_sum_yr, total_ocean_area_60S.values) *100 #in %

# mhw_total_area = (mhw_normalized_by_area * area_SO_surf_60S).sum(dim=['eta_rho', 'xi_rho']) # MHW area

# Normalize by total ocean area to get the fraction
# mhw_fraction = mhw_total_area / total_ocean_area_60S
# mhw_fraction_per_season = mhw_total_area / total_ocean_area_60S


# %% --- PLOT
years = np.arange(1980, 1980 + 40)  # 1980 to 2019
season_colors = ['#F39C12', '#E74C3C', '#4A90E2', '#2ECC71']  # summer, fall, winter, spring
threshold_colors = ['#5A7854', '#8780C6', '#9B2808']  # 1, 2, 3°C
season_names = ['Summer', 'Fall', 'Winter', 'Spring']

# Linear trend line (1st-degree pol)
slope1, intercept1 = np.polyfit(years, det1deg_area_frac_yr, 1)
trend_line1 = slope1 * years + intercept1

slope2, intercept2 = np.polyfit(years, det2deg_area_frac_yr, 1)
trend_line2 = slope2 * years + intercept2

slope3, intercept3 = np.polyfit(years, det3deg_area_frac_yr, 1)
trend_line3 = slope3 * years + intercept3

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
# plt.fill_between(years, det_area_frac_yr, alpha=0.2, color='#47455F', label="Annual Mean Area Fraction")
ax.plot(years, det1deg_area_frac_yr, marker='o', linestyle='-', color=threshold_colors[0], linewidth=2, label="1°C threshold")
ax.plot(years, trend_line1, linestyle="--", color=threshold_colors[0], linewidth=2)

ax.plot(years, det2deg_area_frac_yr, marker='o', linestyle='-', color=threshold_colors[1], linewidth=2, label="2°C threshold")
ax.plot(years, trend_line2, linestyle="--", color=threshold_colors[1], linewidth=2)

ax.plot(years, det3deg_area_frac_yr, marker='o', linestyle='-', color=threshold_colors[2], linewidth=2, label="3°C threshold")
ax.plot(years, trend_line3, linestyle="--", color=threshold_colors[2], linewidth=2)

# for season in range(4):  # Loop over all seasons to plot each one
#     mhw_data_toplot = combine_div_sn[:, season]
#     plt.plot(years, mhw_data_toplot, color=season_colors[season], linewidth=1.5, label=season_names[season])
#     # plt.fill_between(years, mhw_data_toplot, alpha=0.3, color=season_colors[season])

# Custom legend
# handles = [
#     plt.Line2D([0], [0], color=season_colors[i], lw=3) for i in range(4)
# ] + [
#     plt.Line2D([0], [0], color='black', lw=3),  # Annual Mean
#     plt.Line2D([0], [0], linestyle="--", color='r', lw=3)  # Trend Line
# ]
# labels = season_names + ["Annual Mean", "Trend Line"]
# plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# Labels and title
ax.set_xlabel("Year")
ax.set_ylabel('Area fraction')
ax.set_title("Areal Fraction of Potential MHWs \n South of 60°S - 30yrs baseline (1980-2010)")
# plt.grid(True)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.show()




# %%

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Plot area 
pcolormesh = area_SO_surf_60S[:, 1:-1].plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False, 
    # vmin=vmin, vmax=vmax,
    cmap='inferno')
    


# # Colorbar
# if cbar_label:
#     cbar = plt.colorbar(pcolormesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
#     cbar.set_label('km2', fontsize=13)
#     cbar.ax.tick_params(labelsize=12)  

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

# Title
ax.set_title(f"Area Fraction", fontsize=20, pad=30)

plt.tight_layout()
plt.show()
# %%

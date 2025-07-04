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

# %% ======================== Load data ========================
# MHW durations
mhw_duration_5m = xr.open_dataset(os.path.join(path_duration, "mhw_duration_5m.nc")).mhw_durations #dataset - shape (40, 365, 434, 1442)
# print(mhw_duration_5m.isel(eta_rho=224, xi_rho=583, years=38, days=slice(0,30)).values)
cell = mhw_duration_5m.isel(eta_rho=211, xi_rho=889, years=slice(37, 38),  days=slice(304, 365))
print("Durations:")
print(cell.values) #Only 0

det_combined_ds = xr.open_dataset(os.path.join(path_det, 'det5m_extended.nc')) #boolean shape (40, 365, 434, 1442)
# print(det_combined_ds.det_4deg_extended.isel(eta_rho=224, xi_rho=583, years=38, days=slice(0,30)).values)
cell = det_combined_ds.isel(eta_rho=211, xi_rho=889, days=slice(304, 365), years=slice(37, 38))
print("Detection flag (det_1deg):")
print(cell.det_1deg_extended.values) #Only TRUE

# -------------------------------------- FULL YEAR --------------------------------------
# -- Write or load data
combined_file_FULL = os.path.join(os.path.join(path_det, 'duration_AND_thresh_5mFULL.nc'))

if not os.path.exists(combined_file_FULL):

    # === Select 60°S south extent
    south_mask = mhw_duration_5m['lat_rho'] <= -60
    mhw_duration_5m_NEW_60S_south = mhw_duration_5m.where(south_mask, drop=True) #shape (40, 365, 231, 1442)
    det_combined_ds_60S_south = det_combined_ds.where(south_mask, drop=True) #shape (40, 365, 231, 1442)
    det_combined_ds_60S_south = det_combined_ds_60S_south.transpose('years','days','eta_rho','xi_rho')

    # === Associate each mhw duration with the event threshold 
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
                "duration":"Duration redefined as following the rules of Hobday et al. (2016), based on relative threshold (90thperc) - based on the condition that a mhw is when T°C > absolute AND relative thresholds",
                "det_ideg": "Detected events where SST > (EXTENDED absolute threshold (i°C) BUT NOT NECESSARILY 90th percentile) , boolean array"
                }                
            )

    # Write to file
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

if not os.path.exists(combined_file):
    seasonal_vars = {}
    
    for var in ['duration', 'det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']:
        print(f"Processing {var}")
        var='duration'
        # Step 1: Wrap variable as dataset (this avoids sel errors in your parallel logic)
        var_ds = ds_duration_thresh_FULLyear[[var]]

        # Step 2: Extract seasonal slice in parallel -- about 10min computing per variable
        season_ds = define_season_all_years_parallel(var_ds, max_workers=30)

        # Step 3: Clean coordinate naming
        season_ds = season_ds.rename({'season_year': 'season_year_temp'})
        if 'years' in season_ds:
            season_ds = season_ds.drop_vars('years')
        season_ds = season_ds.rename({'season_year_temp': 'years'})

        # Step 4: Save result
        seasonal_vars[var] = season_ds[var]  # Extract DataArray again for merge

    # Merge all seasonal DataArrays into one Dataset
    ds_duration_thresh_SEASON = xr.merge(seasonal_vars.values())
     
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


cell = ds_duration_thresh_SEASON.isel(eta_rho=211, xi_rho=889, years=slice(37, 38))
print("Durations:")
print(cell.duration.values)

print("Detection flag (det_1deg):")
print(cell.det_1deg.values)

# %% ---- Compute Number of days under MHWs
# -- Write or load data
nb_days_file_FULL = os.path.join(os.path.join(path_det, 'nb_days_underMHWs_5mFULL.nc'))

if not os.path.exists(nb_days_file_FULL):

    # MHWs - UNION between duration (representing "extended" relative threshold) AND absolute extended thresholds
    mhw1deg = ds_duration_thresh_FULLyear['det_1deg'].where(ds_duration_thresh_FULLyear['duration']!=0)
    mhw2deg = ds_duration_thresh_FULLyear['det_2deg'].where(ds_duration_thresh_FULLyear['duration']!=0)
    mhw3deg = ds_duration_thresh_FULLyear['det_3deg'].where(ds_duration_thresh_FULLyear['duration']!=0)
    mhw4deg = ds_duration_thresh_FULLyear['det_4deg'].where(ds_duration_thresh_FULLyear['duration']!=0)

    # Number of MHWs days for each year
    mhw_days_1deg = mhw1deg.sum(dim='days') #max: 365 days -- mean ~2.91402844days
    # mhw_days_1deg_BEFORE = ds_mhw_duration['det_1deg'].sum(dim='days') #max: 365 days -- mean ~16.78351751days
    mhw_days_2deg = mhw2deg.sum(dim='days') #max: 365 days
    mhw_days_3deg = mhw3deg.sum(dim='days') #max: 360 days
    mhw_days_4deg = mhw4deg.sum(dim='days') #max: 326 days

    # Number of MHWs days per year
    mhw1deg_days_per_year = mhw_days_1deg.mean(dim='years') #max: 74.125 days/yr
    mhw2deg_days_per_year = mhw_days_2deg.mean(dim='years') #max: 68.05 days/yr
    mhw3deg_days_per_year = mhw_days_3deg.mean(dim='years') #max: 68.05 days/yr
    mhw4deg_days_per_year = mhw_days_4deg.mean(dim='years') #max: 55.75 days/yr

    # To dataset
    ds_mhw_daysperyear= xr.Dataset(
            data_vars=dict(
                nb_days_1deg_per_yr = (["eta_rho" ,"xi_rho"], mhw1deg_days_per_year.data), #shape (231, 1442)
                nb_days_2deg_per_yr = (["eta_rho" ,"xi_rho"], mhw2deg_days_per_year.data),
                nb_days_3deg_per_yr = (["eta_rho" ,"xi_rho"], mhw3deg_days_per_year.data),
                nb_days_4deg_per_yr = (["eta_rho" ,"xi_rho"], mhw4deg_days_per_year.data)
                ),
            coords=dict(
                lon_rho=(["eta_rho", "xi_rho"], ds_duration_thresh_FULLyear.lon_rho.values), #(434, 1442)
                lat_rho=(["eta_rho", "xi_rho"], ds_duration_thresh_FULLyear.lat_rho.values), #(434, 1442)
                ),
            attrs = {
                    "depth": "5m",
                    "nb_days_ideg_per_yr":"Number of days per year being under MHW of i°C"
                    }                
                )
    
    # Write to file
    ds_mhw_daysperyear.to_netcdf(nb_days_file_FULL)
else: 
    # Load data
    ds_mhw_daysperyear = xr.open_dataset(nb_days_file_FULL)

# %% Temperature water column under MHWs
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



# %% Plot Avg temperature
# --- Define cmap and norm for temperature (2nd row) ---
cmap_temp = 'coolwarm'
vmin_temp = -2
vmax_temp = 2
norm_temp = mcolors.TwoSlopeNorm(vmin=vmin_temp, vcenter=0, vmax=vmax_temp)
variables_temp = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
titles = ['1°C', '2°C', '3°C', '4°C']

# 4 subplots - 1 for each absolute threshold 
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width 
fig = plt.figure(figsize=(fig_width*4, fig_height))  # wide enough for 4 subplots in a row
gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)  # 4 columns

axs = []
for j in range(4):
    # ax = fig.add_subplot(gs[j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    ax = fig.add_subplot(gs[0, j], projection=ccrs.SouthPolarStereo())
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
    ax.coastlines(color='black', linewidth=1, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=1, zorder=5)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=2)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
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

    ax.set_title(f'MHW $>$ {titles[i]}', fontsize=16)

# Common colorbar
tick_positions = np.arange(-2, 2.5, 0.5) 
cbar = fig.colorbar(
    im, ax=axs, orientation='horizontal',
    fraction=0.07, pad=0.1, ticks=tick_positions
)
cbar.set_label('Temperature [°C]', fontsize=12)

plt.suptitle("Average temperature of the first 100m under MHWs \n1980-2019", fontsize=24, y=1.05)
plt.tight_layout(rect=[0, 0, 1, 0.99])  # [left, bottom, right, top]
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/nb_of_days.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %% Plot
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm, ListedColormap

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
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width 
fig = plt.figure(figsize=(fig_width*4, fig_height))  # wide enough for 4 subplots in a row
gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)  # 4 columns

axs = []
for j in range(4):
    # ax = fig.add_subplot(gs[j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    ax = fig.add_subplot(gs[0, j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
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
    ax.coastlines(color='black', linewidth=1, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=1, zorder=5)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=2)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
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

    ax.set_title(f'MHW $>$ {titles[i]}', fontsize=16)

# Common colorbar
tick_positions = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds)-1)]  # Tick positions at bin centers
cbar = fig.colorbar(im, ax=axs, orientation='horizontal', extend='max', fraction=0.07, ticks=tick_positions)#aspect=10)
cbar.set_label('days per year', fontsize=12)
cbar.ax.set_xticklabels(labels)

plt.suptitle("Number of days per year under MHWs \n On average over the 1980-2019 period - 5m depth", fontsize=20, y=1.05)

plt.tight_layout(rect=[0, 0, 1, 0.89])  # [left, bottom, right, top]
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/nb_of_days.pdf'), dpi =150, format='pdf', bbox_inches='tight')


# %% ---- Compute  frequency of events
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
    # print(valid_mask_np[38,:,224,583])

    years_len, days_len, eta_len, xi_len = valid_mask_np.shape # shape: (40, 365, 231, 1442)
    eta_xi = eta_len * xi_len

    # Reshape
    valid_mask_reshaped = valid_mask_np.transpose(0, 2, 3, 1).reshape(years_len, eta_xi, days_len)
    # print(valid_mask_reshaped[38, 224 * 1442 + 583])

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
    # print(event_counts[38, 224 * 1442 + 583])

    # Reshape
    event_counts = event_counts.reshape(years_len, eta_len, xi_len)

    # Into DataArray
    event_counts_da = xr.DataArray(
        event_counts,
        coords={
            'years': ds_duration_thresh_FULLyear['years'],
            'eta_rho': ds_duration_thresh_FULLyear['eta_rho'],
            'xi_rho': ds_duration_thresh_FULLyear['xi_rho']
        },
        dims=['years', 'eta_rho', 'xi_rho'],
        name=f'mhw_event_counts_{var}'
    )
    # print(event_counts_da.isel(years=38, eta_rho=224, xi_rho=583))

    # Annual mean for each grid cell
    avg_event_counts = event_counts_da.sum(dim='years') #max 1°C: 29yr, max 2°C: 28yr, max 3°C: 27yr, max 4°C: 24yr  
    avg_event_counts_dict[var] = avg_event_counts
    print(f"Maximum frequency for {var}: {np.max(avg_event_counts).values} years") 

# %% Find cell with NO MHWs - for each tresholds 
no_mhw_cells = {}
var_no_mhws= ['det_1deg', 'det_2deg','det_3deg', 'det_4deg']

# -- Write or load data
no_mhw_file = os.path.join(os.path.join(path_det, 'noMHW_5m.nc'))
if not os.path.exists(no_mhw_file):
    for var in var_no_mhws:
        is_mhw = ds_duration_thresh_FULLyear[var].astype(bool) & (ds_duration_thresh_FULLyear['duration'] != 0)
        # no_mhws_flag= ds_duration_thresh_FULLyear[var].where(ds_duration_thresh_FULLyear['duration']!=0) # 1 where MHW occurred -- shape: (40, 365, 231, 1442)
        no_mhw_cells[var] = ~is_mhw.any(dim=('years', 'days')) # True where NO MHW EVER occurred -- shape: (231, 1442)
        
        # Check
        count = no_mhw_cells[var].sum().item()
        print(f"{var}: {count} cells with no MHWs") #1°C: 210997 cells, 2°C: 269736 cells, 3°C: 307961 cells, 4°C: 320439  cells
   
    # Write to file
    ds_no_mhw = xr.Dataset(no_mhw_cells)
    ds_no_mhw.to_netcdf(no_mhw_file)

else:
    # Load data
    ds_no_mhw = xr.open_dataset(no_mhw_file)

# %% Find cell with  MHWs lasting less than 30days - for each thredhols 
short_mhw_cells = {}
variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']

# -- Write or load data
short_mhw_file = os.path.join(os.path.join(path_det, 'shortMHW_5m.nc'))
if not os.path.exists(short_mhw_file):

    for var in variables:
        mhw_flags = ds_duration_thresh_FULLyear[var].where(ds_duration_thresh_FULLyear['duration']!=0)  # shape: (years, days, eta_rho, xi_rho): (40, 365, 231, 1442)
        ever_mhw = mhw_flags.any(dim=('years', 'days'))   # True where MHW occurred
        long_events = avg_event_counts_dict[var] > 0      # True where ≥30d MHW occurred

        short_mhw_cells[var] = ever_mhw & ~long_events    # True where only short MHWs occurred (<30 days)
    
    # Write to file
    ds_short_mhw = xr.Dataset(short_mhw_cells)
    ds_short_mhw.to_netcdf(short_mhw_file)

else: 
    # Load data
    ds_short_mhw = xr.open_dataset(short_mhw_file)

# %% --- PLOT area SO
from matplotlib.colors import LinearSegmentedColormap
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

report = False 
if report==True:
    figsize=(fig_width*2, fig_height)

else:
    figsize=(fig_width*4, fig_height)

fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)  # 4 columns

axs = []
for j in range(4):
    ax = fig.add_subplot(gs[0, j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
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

    # Map extent and features
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.coastlines(color='black', linewidth=1, zorder=6)
    ax.add_feature(cfeature.LAND, zorder=5, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sectors
    for k in [-90, 0, 120]:
        ax.plot([k, k], [-90, -60], transform=ccrs.PlateCarree(), color="#000000", linestyle='--', linewidth=1, zorder=7)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color="#000000", alpha=0.5, linestyle='--', linewidth=0.7, zorder=4)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Base layer: long events
    long_mhw = ax.pcolormesh(ds_duration_thresh_FULLyear['lon_rho'], ds_duration_thresh_FULLyear['lat_rho'], avg_event_counts_dict[var], 
                                transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading='auto', zorder=1)
    if long_mhw_ref is None:
        long_mhw_ref = long_mhw  # Save the first one for colorbar

    # 2nd layer - short MHWs in grey
    colors = [(0, 0, 0, 0),  # 0 ->  transparent
            (0.75, 0.75, 0.75, 1)]  # 1 -> grey
    cmap_short = mcolors.ListedColormap(colors)
    short_mhw = ax.pcolormesh(ds_duration_thresh_FULLyear['lon_rho'], ds_duration_thresh_FULLyear['lat_rho'], ds_short_mhw[var].astype(int),
                            cmap=cmap_short, shading='auto', transform=ccrs.PlateCarree(), zorder=2)

    # 3rd layer - no mhws 
    colors = [(0, 0, 0, 0),  # False (0) -> transparent
          (1, 1, 1, 1)] # True (1) -> opaque white
    cmap_no = mcolors.ListedColormap(colors)
    no_mhw_plot = ax.pcolormesh(ds_no_mhw['lon_rho'], ds_no_mhw['lat_rho'], ds_no_mhw[var].astype(int),
                                cmap=cmap_no, transform=ccrs.PlateCarree(), zorder=3)

    ax.set_title(f'MHW $>$ {titles[i]}', fontsize=16)

# Legend
import matplotlib.patches as mpatches
short_mhw_patch = mpatches.Patch(facecolor='grey', edgecolor='black', label='Short MHWs ($<$30days)', linewidth=1)
no_mhw_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='No MHWs', linewidth=1)
ax.legend(handles=[short_mhw_patch, no_mhw_patch], 
          loc='lower center', bbox_to_anchor=(0.02, -0.1),
           ncol=1, frameon=True, fontsize=12)


# Create common colorbar with ticks centered on bins
tick_positions = [(bins[j] + bins[j+1]) / 2 for j in range(len(bins)-1)]
cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.03])  #[left, bottom, width, height]
cbar = fig.colorbar(
    mappable=long_mhw_ref,
    cax=cbar_ax,
    fraction=0.04,
    orientation='horizontal',
    ticks=tick_positions,
    extend='max'
)

cbar.ax.set_xticklabels(['1-5', '5-10', '10-15', '15-20', '$>$20'], fontsize=14)
cbar.set_label("Years", fontsize=16)

plt.suptitle("Number of years with events lasting more than 30 days \n1980-2019 period - 5m depth", fontsize=20, y=1.05)
plt.tight_layout(rect=[0, 0, 0, 0.95])
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/nb_of_years.pdf'), dpi =150, format='pdf', bbox_inches='tight')




# %%

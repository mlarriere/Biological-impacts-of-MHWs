#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues 10 March 08:47:34 2025

Growth model according to Atkinson et al. (2006) 

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import matplotlib as mpl
import psutil #retracing memory
import sys
import glob
import re
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
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
    "font.size": 9,           
    "axes.titlesize": 9,
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

sys.path.append(working_dir+'Growth_Model') 
from B_Growth_Model.Atkinson2006_model import growth_Atkinson2006  # import growth function

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

path_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'

path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
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
doy_list = list(range(305, 365)) + list(range(0, 121))
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

# %% ================ Load data ================
# ==== Temperature [°C] -- Weighted averaged temperature of the first 100m - Austral summer - 60S
temp_avg_100m = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (40, 181, 231, 1442)
 
# ==== Chla [mh Chla/m3] -- Surface chla (5m) - Austral summer - 60S
chla_surf_corrected_seasonal = xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 

# == Climatological Drivers -> Mean T°C  (days, eta, xi) period: 1980-2009
temp_clim = temp_avg_100m.isel(years=slice(0,30)).mean(dim=['years'])
chla_clim = chla_surf_corrected_seasonal.isel(years=slice(0,30)).mean(dim=['years'])

# %% ================ Hypothetical Growth ================
# --- 2D grid of all combinations
chla_hyp= np.arange(0, 5, 0.01)
temp_hyp = np.arange(-4, 4, 0.01)

# Calculating growth for all combination
CHLA, TEMP = np.meshgrid(chla_hyp, temp_hyp)
growth_hyp = growth_Atkinson2006(CHLA, TEMP, maturity_stage='juvenile', length=26) #Bahlburg et al. (2023): juvenile 26mm krill

# %% ======================== Climatological growth (1980-2009) for 1 location ========================
# ---- Prepare data for plot
file_chla = os.path.join(path_growth_inputs, "chla_spatial_mean.nc")
file_temp = os.path.join(path_growth_inputs, "temp_spatial_mean.nc")
if not os.path.exists(file_chla):
    chla_mean = chla_surf_corrected_seasonal.mean(dim=['eta_rho', 'xi_rho'])
    chla_mean.to_netcdf(file_chla)
else: 
    chla_mean = xr.open_dataset(file_chla)

if not os.path.exists(file_temp):
    temp_mean = temp_avg_100m.mean(dim=['eta_rho', 'xi_rho'])
    temp_mean.to_netcdf(file_temp)
else: 
    temp_mean = xr.open_dataset(file_temp)

# %% ======================== Areas and volume MPAs ========================
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)

# ---- Fix extent 
# South of 60°S
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km2
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km3

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)

# --- Calculate area and volume of each MPA
mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}


# %% ============ Mask temp and chla for MPAs ============
# load dta
chla_full_extent = chla_surf_corrected_seasonal.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
temp_full_extent = temp_avg_100m.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

# --- Mask and spatial mean per MPA ---
chla_mpa = {}
temp_mpa = {}

for mpa_key, (mpa_name, mpa_mask) in mpa_masks.items():
    # test
    # mpa_key = 'SO'
    # mpa_name = mpa_masks[mpa_key][0]
    # mpa_mask = mpa_masks[mpa_key][1]
        
    file_chla_mpa = os.path.join(path_growth_inputs, f'mpas/chla_seasonal_{mpa_key}')
    file_temp_mpa = os.path.join(path_growth_inputs, f'mpas/temp_seasonal_{mpa_key}')
    if os.path.exists(file_chla_mpa) and os.path.exists(file_temp_mpa):
        # Load from file
        chla_mpa[mpa_key] = xr.open_dataset(file_chla_mpa)['chla']
        temp_mpa[mpa_key] = xr.open_dataset(file_temp_mpa)['avg_temp']
        print(f"{mpa_name} ({mpa_key}): loaded from file")

    else:  
        
        # Apply MPA mask
        chla_masked = chla_full_extent['chla'].where(mpa_mask)
        temp_masked = temp_full_extent['avg_temp'].where(mpa_mask)

        # Spatial mean over eta_rho, xi_rho
        chla_mpa[mpa_key] = chla_masked.mean(dim=['eta_rho', 'xi_rho'])
        temp_mpa[mpa_key] = temp_masked.mean(dim=['eta_rho', 'xi_rho'])

        # Save to file 
        chla_mpa[mpa_key].to_netcdf(file_chla_mpa)
        temp_mpa[mpa_key].to_netcdf(file_temp_mpa)
        print(f"{mpa_name} ({mpa_key}): chla shape = {chla_mpa[mpa_key].shape}, temp shape = {temp_mpa[mpa_key].shape}")

    
#%%  ============ Plot ============
plot='report' #slides report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width*0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
else:  # 'slides'
    fig, ax = plt.subplots(figsize=(10, 6))

# Font size settings
title_kwargs = {'fontsize': 15} if plot == 'slides' else {} #'fontsize': 14
label_kwargs = {'fontsize': 14} if plot == 'slides' else {} #'fontsize': 12
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {} #'labelsize': 10
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {} #'fontsize': 12
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}

# --- Colormap normalization ---
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

# --- Contour levels ---
min_val = -0.2 #np.nanmin(growth_hyp)
max_val = 0.4 # np.nanmax(growth_hyp)
levels = np.arange(np.floor(min_val * 10) / 10, np.ceil(max_val * 10) / 10 + 0.05, 0.05)

# --- Plot 0 contour ---
zero_level = [0]
manual_zero_pos = [(0.5, -0.5)]
zero_contour = ax.contour(CHLA, TEMP, growth_hyp, levels=zero_level, colors='black', linewidths=0.8, linestyles='--', zorder=3)
label_fontsize = 12 if plot == 'slides' else None
ax.clabel(zero_contour, manual=manual_zero_pos, fmt="%.2f", inline=True, fontsize=label_fontsize, colors='black')

# --- Plot selected labeled contours ---
levels_to_plot = [-0.2, -0.1, 0.1, 0.2, 0.3]
if plot == 'report':
    manual_positions = {
        -0.2: (0.33, -1.4), 
        -0.1: (0.4, -0.7),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
         0.3: (1.5, -0.4),
    }
else:  # slides
    manual_positions = {
        -0.2: (0.4, -1.65),
        -0.1: (0.45, -0.9),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
        0.3: (1.5, -0.4),
    }

for lvl in levels_to_plot:
    contour = ax.contour(CHLA, TEMP, growth_hyp, levels=[lvl], colors='white',
                         linewidths=0.8, linestyles='--', zorder=3)
    try:
        ax.clabel(contour, manual=[manual_positions[lvl]], fmt="%.2f", inline=True,
                fontsize=label_fontsize, colors='white')
    except Exception as e:
        print(f"Failed to label level {lvl}: {e}")

# --- Pseudocolor background ---
pcm = ax.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto', 
                    cmap='coolwarm_r', norm=norm, rasterized=True)

# --- One line per year ---
import matplotlib.cm as cm
years = temp_mean.years.values
cmap_years = cm.get_cmap('plasma', len(years))
lw = 1.5 if plot == 'slides' else 1.0

year_to_plot = 2018  # change as needed

chla_yr = chla_mean['chla'].sel(years=year_to_plot).values
temp_yr = temp_mean['avg_temp'].sel(years=year_to_plot).values

valid_mask = ~np.isnan(chla_yr) & ~np.isnan(temp_yr)
ax.plot(chla_yr[valid_mask], temp_yr[valid_mask],
        color='black', linewidth=lw, alpha=0.9, zorder=4, label=str(year_to_plot))
# --- MPA colors ---
mpa_colors = {
    'RS': '#9a031e',  
    'SO': '#F7B538',   
    'EA': '#5f0f40',  
    'WS': '#bb4d00', 
    'AP': '#7c6a0a',
}


# --- Southern Ocean mean (black) ---
ax.plot(chla_yr[valid_mask], temp_yr[valid_mask],
        color='black', linewidth=lw, alpha=0.9, zorder=5, label=f'SO mean {year_to_plot}')

# --- One line per MPA ---
for mpa_key, (mpa_name, _) in mpa_masks.items():
    chla_yr_mpa = chla_mpa[mpa_key].sel(years=year_to_plot).values
    temp_yr_mpa = temp_mpa[mpa_key].sel(years=year_to_plot).values

    valid = ~np.isnan(chla_yr_mpa) & ~np.isnan(temp_yr_mpa)
    if valid.sum() == 0:
        print(f"Skipping {mpa_key}: no valid data")
        continue

    ax.plot(chla_yr_mpa[valid], temp_yr_mpa[valid],
            color=mpa_colors[mpa_key], linewidth=lw, alpha=0.9, zorder=4,
            label=mpa_name)


# --- Legend ---
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='black', lw=2, label=f'Southern Ocean'),
    *[Line2D([0], [0], color=mpa_colors[k], lw=2, label=mpa_masks[k][0])
      for k in mpa_masks]
]

legend = ax.legend(
    handles=custom_lines,
    loc='upper right',
    # bbox_to_anchor=(1.02, 0.5) if plot == 'report' else (0.02, 0.98),
    frameon=True,              # Enable frame
    facecolor='white',         # White background
    framealpha=0.9,            # Slight transparency
    handlelength=1,        # Length lines
    handletextpad=0.8,     # spacing between handle and text (default ~0.8)
    borderaxespad=0.5,     # padding between axes and legend box
    borderpad=0.4,         # padding inside the legend box
    labelspacing=0.6,      # vertical spacing between entries (reduced from 1.2)
    **legend_kwargs        # includes fontsize only for 'slides'
)
legend.get_frame().set_linewidth(0.5)  # Default is ~1.0; reduce for thinner box

# --- Axis labels and title ---
if plot == 'report':
    suptitle_y = 0.99
else:
    suptitle_y = 1

fig.suptitle(f'Krill Growth Dynamic\nSpatial Mean {year_to_plot}', y=suptitle_y, **suptitle_kwargs)
ax.set_xlabel('Chlorophyll-a [mg/m³]', **label_kwargs)
ax.set_ylabel('Temperature [°C]', **label_kwargs)
# ax.set_yticks(np.arange(-3, 4, 1))
ax.tick_params(**tick_kwargs)
ax.set_ylim(-2,2)
ax.set_xlim(0,3)

# --- Colorbar ---
cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04, label='Growth [mm/d]', extend='both')
cbar.ax.yaxis.label.set_size(label_kwargs.get('fontsize', None))
cbar.ax.tick_params(**tick_kwargs)

# --- Final layout ---
plt.tight_layout()
plt.show()
# %% ============ Surrogates (spatial means)============
path_surr = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/surrogates'
# -- Detrended signal (no warming)
temp_detrended = xr.open_dataset(os.path.join(path_surr, 'detrended_signal/temp_detrended_seasonal.nc'))
temp_detrended_mean = temp_detrended.mean(dim=['eta_rho', 'xi_rho'])

# -- No MHWs signal
temp_no_mhw = xr.open_dataset(os.path.join(path_surr, 'detrended_signal/clim_with_trend.nc'))
temp_no_mhw_mean = temp_no_mhw.mean(dim=['eta_rho', 'xi_rho'])

# -- Climatological signal
temp_clim_mean = temp_clim.mean(dim=['eta_rho', 'xi_rho'])
chla_clim_mean = chla_clim.mean(dim=['eta_rho', 'xi_rho'])

# %% ============ Mask temp and chla for MPAs ============
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# -- Load data
temp_detrended_extent = temp_detrended.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
temp_no_mhw_extent = temp_no_mhw.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
temp_clim_extent = temp_clim.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
chla_clim_extent = chla_clim.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))


def process_mpa(mpa_key, mpa_masks, path_surr, temp_detrended_extent, temp_no_mhw_extent, temp_clim_extent, chla_clim_extent):
    mpa_name, mpa_mask = mpa_masks[mpa_key]

    file_temp_detrended_mpa = os.path.join(path_surr, f'detrended_signal/mpas/temp_detrended_{mpa_key}')
    file_temp_no_mhw_mpa = os.path.join(path_surr, f'detrended_signal/mpas/temp_no_mhw_seasonal_{mpa_key}')
    file_temp_clim_mpa = os.path.join(path_surr, f'detrended_signal/mpas/temp_clim_{mpa_key}')
    file_chla_clim_mpa = os.path.join(path_surr, f'detrended_signal/mpas/chla_clim_{mpa_key}')

    if os.path.exists(file_temp_detrended_mpa) and os.path.exists(file_temp_no_mhw_mpa) and os.path.exists(file_temp_clim_mpa) and os.path.exists(file_chla_clim_mpa):
        print(f"{mpa_name} ({mpa_key}): loaded from file")
        return mpa_key, file_temp_detrended_mpa, file_temp_no_mhw_mpa, file_temp_clim_mpa, file_chla_clim_mpa,  'loaded'

    else:
        # Apply MPA mask and spatial mean
        temp_detrended_mpa = temp_detrended_extent['avg_temp'].where(mpa_mask).mean(dim=['eta_rho', 'xi_rho'])
        temp_no_mhw_mpa = temp_no_mhw_extent['avg_temp'].where(mpa_mask).mean(dim=['eta_rho', 'xi_rho'])
        temp_clim_mpa = temp_clim_extent['avg_temp'].where(mpa_mask).mean(dim=['eta_rho', 'xi_rho'])
        chla_clim_mpa = chla_clim_extent['chla'].where(mpa_mask).mean(dim=['eta_rho', 'xi_rho'])

        # Save to file
        temp_detrended_mpa.to_netcdf(file_temp_detrended_mpa)
        temp_no_mhw_mpa.to_netcdf(file_temp_no_mhw_mpa)
        temp_clim_mpa.to_netcdf(file_temp_clim_mpa)
        chla_clim_mpa.to_netcdf(file_chla_clim_mpa)
        print(f"{mpa_name} ({mpa_key}): done.")
        return mpa_key, file_temp_detrended_mpa, file_temp_no_mhw_mpa, file_temp_clim_mpa, file_chla_clim_mpa, 'computed'


# --- Run in parallel ---
mpa_keys = list(mpa_masks.keys())
_process = partial(process_mpa,
                   mpa_masks=mpa_masks,
                   path_surr=path_surr,
                   temp_detrended_extent=temp_detrended_extent,
                   temp_no_mhw_extent=temp_no_mhw_extent,
                   temp_clim_extent=temp_clim_extent,
                   chla_clim_extent=chla_clim_extent)

with ProcessPoolExecutor(max_workers=len(mpa_keys)) as executor:
    results = list(executor.map(_process, mpa_keys))

# --- Load results back into dicts ---
temp_detrended_mpa = {}
temp_no_mhw_mpa = {}
temp_clim_mpa = {}
chla_clim_mpa = {}

for mpa_key, file_det, file_nomhw, file_clim, file_chla_clim, status in results:
    temp_detrended_mpa[mpa_key] = xr.open_dataset(file_det)['avg_temp']
    temp_no_mhw_mpa[mpa_key] = xr.open_dataset(file_nomhw)['avg_temp']
    temp_clim_mpa[mpa_key] = xr.open_dataset(file_clim)['avg_temp']
    chla_clim_mpa[mpa_key] = xr.open_dataset(file_chla_clim)['chla']

# %% ============ Plot Southern Ocean - different surrogates ============
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# --- Year selection ---
year_to_plot = 1980
mpa_choice= 'SO'
plot = 'report'  # slides report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width * 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
else:
    fig, ax = plt.subplots(figsize=(10, 6))

# Font size settings
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}
lw = 1.5 if plot == 'slides' else 1.0

# --- Colormap normalization ---
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

# --- Pseudocolor background ---
pcm = ax.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto',
                    cmap='coolwarm_r', norm=norm, rasterized=True)

# --- Plot 0 contour ---
zero_level = [0]
manual_zero_pos = [(0.5, -0.5)]
zero_contour = ax.contour(CHLA, TEMP, growth_hyp, levels=zero_level, colors='black',
                          linewidths=0.8, linestyles='--', zorder=3)
label_fontsize = 12 if plot == 'slides' else None
ax.clabel(zero_contour, manual=manual_zero_pos, fmt="%.2f", inline=True,
          fontsize=label_fontsize, colors='black')

# --- Plot selected labeled contours ---
levels_to_plot = [-0.2, -0.1, 0.1, 0.2, 0.3]
if plot == 'report':
    manual_positions = {
        -0.2: (0.33, -1.4),
        -0.1: (0.4, -0.7),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
         0.3: (1.5, -0.4),
    }
else:
    manual_positions = {
        -0.2: (0.4, -1.65),
        -0.1: (0.45, -0.9),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
         0.3: (1.5, -0.4),
    }

for lvl in levels_to_plot:
    contour = ax.contour(CHLA, TEMP, growth_hyp, levels=[lvl], colors='white',
                         linewidths=0.8, linestyles='--', zorder=3)
    try:
        ax.clabel(contour, manual=[manual_positions[lvl]], fmt="%.2f", inline=True,
                  fontsize=label_fontsize, colors='white')
    except Exception as e:
        print(f"Failed to label level {lvl}: {e}")


# --- Observed temp ---
chla_yr = chla_mpa[mpa_choice].sel(years=year_to_plot).values
temp_yr = temp_mpa[mpa_choice].sel(years=year_to_plot).values
valid_mask = ~np.isnan(chla_yr) & ~np.isnan(temp_yr)

ax.plot(chla_yr[valid_mask], temp_yr[valid_mask],
        color='#648028', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='-', label=f'Actual signal')

# --- Southern Ocean mean (detrended temp) ---
temp_yr_det = temp_detrended_mpa[mpa_choice].sel(years=year_to_plot).values  # adjust var name if needed
valid_mask_det = ~np.isnan(chla_yr) & ~np.isnan(temp_yr_det)

ax.plot(chla_yr[valid_mask_det], temp_yr_det[valid_mask_det],
        color='#584CBD', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='--', label=f'Signal without warming')

# --- Southern Ocean mean (no MHW temp) ---
temp_yr_nomhw = temp_no_mhw_mpa[mpa_choice].sel(years=year_to_plot).values  # adjust var name if needed
valid_mask_nomhw = ~np.isnan(chla_yr) & ~np.isnan(temp_yr_nomhw)

ax.plot(chla_yr[valid_mask_nomhw], temp_yr_nomhw[valid_mask_nomhw],
        color='#F18701', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='--', label=f'Signal without MHWs')

# --- Southern Ocean mean (climatological signal) ---
temp_clim_val = temp_clim_mpa[mpa_choice].values  # adjust var name if needed
chla_clim_val = chla_clim_mpa[mpa_choice].values  # adjust var name if needed
valid_mask_clim = ~np.isnan(chla_clim_val) & ~np.isnan(temp_clim_val)

ax.plot(chla_clim_val[valid_mask_det], temp_clim_val[valid_mask_det],
        color='black', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='-', label=f'Climatological signal')

# --- Legend ---
custom_lines = [
    Line2D([0], [0], color='#648028', lw=2, linestyle='-',  label=f'Actual signal'),
    Line2D([0], [0], color='#584CBD', lw=2, linestyle='--', label=f'Signal without warming'),
    Line2D([0], [0], color='#F18701', lw=2, linestyle='--', label=f'Signal without MHWs'),
    Line2D([0], [0], color='black', lw=2, linestyle='-', label=f'Climatological signal'),
]

legend = ax.legend(
    handles=custom_lines,
    loc='upper right',
    frameon=True,
    facecolor='white',
    framealpha=0.9,
    handlelength=1.5,
    handletextpad=0.8,
    borderaxespad=0.5,
    borderpad=0.4,
    labelspacing=0.6,
    **legend_kwargs
)
legend.get_frame().set_linewidth(0.5)

# --- Axis labels and title ---
suptitle_y = 0.99 if plot == 'report' else 1.0
fig.suptitle('Krill Growth Dynamic', y=suptitle_y, **suptitle_kwargs)
fig.text(0.5, suptitle_y - 0.08, f'{mpa_choice} mean – Year {year_to_plot}',
         ha='center', style='italic', **label_kwargs)
ax.set_xlabel('Chlorophyll-a [mg/m³]', **label_kwargs)
ax.set_ylabel('Temperature [°C]', **label_kwargs)
ax.set_ylim(-2, 3)
# ax.set_xlim(0, 1)
ax.set_yticks(np.arange(-2, 4, 1))
ax.tick_params(**tick_kwargs)
ax.autoscale(enable=False)

# --- Growth colorbar ---
cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04,
                    label='Growth [mm/d]', extend='both')
cbar.ax.yaxis.label.set_size(label_kwargs.get('fontsize', None))
cbar.ax.tick_params(**tick_kwargs)

# --- Final layout ---
plt.tight_layout()
plt.show()
# %% ============ Find max MHW duration in AP ============
mhw_duration_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/interpolated/duration_AND_thresh_{region}.nc")).duration for region in ['RS', 'SO', 'EA', 'WS', 'AP']}
da_ap = mhw_duration_mpas['AP']

# Find the maximum value and its location
max_val = da_ap.max().compute()
print(f"Max MHW duration in AP: {max_val.values} days")

# Find indices of maximum
max_idx = da_ap.argmax(dim=['years', 'days', 'eta_rho', 'xi_rho'])

year_idx = max_idx['years'].values
day_idx  = max_idx['days'].values
eta_idx  = max_idx['eta_rho'].values
xi_idx   = max_idx['xi_rho'].values

# Convert to actual year (1980 + year_idx)
actual_year = 1980 + year_idx
actual_day  = da_ap.days_of_yr.isel(days=day_idx).values
lat         = da_ap.lat_rho.isel(eta_rho=eta_idx, xi_rho=xi_idx).values
lon         = da_ap.lon_rho.isel(eta_rho=eta_idx, xi_rho=xi_idx).values

print(f"Year:      {actual_year}")
print(f"Day of yr: {actual_day}")
print(f"Lat:       {lat:.2f}°")
print(f"Lon:       {lon:.2f}°")
print(f"Grid idx:  eta_rho={eta_idx}, xi_rho={xi_idx}")

# %% ============ Plot 1 location ============
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# --- Data selection ---
year_to_plot = 2000
xi_plot = 1086
eta_plot = 184
chla_clim_plot = chla_clim_mean.isel(eta_rho=eta_plot, xi_rho=xi_plot).chla.values
temp_clim_plot = temp_clim_mean.isel(eta_rho=eta_plot, xi_rho=xi_plot).avg_temp.values
chla_actual_plot = chla_surf_corrected_seasonal.isel(years=year_to_plot-1980, eta_rho=eta_plot, xi_rho=xi_plot).chla.values
temp_actual_plot = temp_avg_100m.isel(years=year_to_plot-1980, eta_rho=eta_plot, xi_rho=xi_plot).avg_temp.values
temp_nowarming_plot = temp_detrended.isel(years=year_to_plot-1980, eta_rho=eta_plot, xi_rho=xi_plot).avg_temp.values
temp_nomhw_plot = temp_no_mhw.isel(years=year_to_plot-1980, eta_rho=eta_plot, xi_rho=xi_plot).avg_temp.values

plot = 'report'  # slides report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width * 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
else:
    fig, ax = plt.subplots(figsize=(10, 6))

# Font size settings
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}
lw = 1.5 if plot == 'slides' else 1.0

# --- Colormap normalization ---
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

# --- Pseudocolor background ---
pcm = ax.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto',
                    cmap='coolwarm_r', norm=norm, rasterized=True)

# --- Plot 0 contour ---
zero_level = [0]
manual_zero_pos = [(0.5, -0.5)]
zero_contour = ax.contour(CHLA, TEMP, growth_hyp, levels=zero_level, colors='black',
                          linewidths=0.8, linestyles='--', zorder=3)
label_fontsize = 12 if plot == 'slides' else None
ax.clabel(zero_contour, manual=manual_zero_pos, fmt="%.2f", inline=True,
          fontsize=label_fontsize, colors='black')

# --- Plot selected labeled contours ---
levels_to_plot = [-0.2, -0.1, 0.1, 0.2, 0.3]
if plot == 'report':
    manual_positions = {
        -0.2: (0.33, -1.4),
        -0.1: (0.4, -0.7),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
         0.3: (1.5, -0.4),
    }
else:
    manual_positions = {
        -0.2: (0.4, -1.65),
        -0.1: (0.45, -0.9),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
         0.3: (1.5, -0.4),
    }

for lvl in levels_to_plot:
    contour = ax.contour(CHLA, TEMP, growth_hyp, levels=[lvl], colors='white',
                         linewidths=0.8, linestyles='--', zorder=3)
    try:
        ax.clabel(contour, manual=[manual_positions[lvl]], fmt="%.2f", inline=True,
                  fontsize=label_fontsize, colors='white')
    except Exception as e:
        print(f"Failed to label level {lvl}: {e}")

# --- Climatological signal ---
valid_mask = ~np.isnan(chla_clim_plot) & ~np.isnan(temp_clim_plot)
ax.plot(chla_clim_plot[valid_mask], temp_clim_plot[valid_mask],
        color='black', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='-', label=f'Climatological signal')

# --- Observed temp ---
valid_mask = ~np.isnan(chla_actual_plot) & ~np.isnan(temp_actual_plot)
ax.plot(chla_actual_plot[valid_mask], temp_actual_plot[valid_mask],
        color='#648028', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='-', label=f'Actual signal')

# --- No warming, i.e. detrended temp ---
valid_mask_det = ~np.isnan(chla_actual_plot) & ~np.isnan(temp_nowarming_plot)
ax.plot(chla_actual_plot[valid_mask_det], temp_nowarming_plot[valid_mask_det],
        color='#584CBD', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='--', label=f'Signal without warming')

# --- No MHW temp ---
valid_mask_nomhw = ~np.isnan(chla_actual_plot) & ~np.isnan(temp_nomhw_plot)
ax.plot(chla_actual_plot[valid_mask_nomhw], temp_nomhw_plot[valid_mask_nomhw],
        color='#F18701', linewidth=lw, alpha=0.9, zorder=5,
        linestyle='--', label=f'Signal without MHWs')

# --- Legend ---
custom_lines = [
    Line2D([0], [0], color='black', lw=2, linestyle='-',  label=f'Climatological signal'),
    Line2D([0], [0], color='#648028', lw=2, linestyle='-',  label=f'Actual signal'),
    Line2D([0], [0], color='#584CBD', lw=2, linestyle='--', label=f'Signal without warming'),
    Line2D([0], [0], color='#F18701', lw=2, linestyle='--', label=f'Signal without MHWs'),
]

legend = ax.legend(
    handles=custom_lines,
    loc='upper right',
    frameon=True,
    facecolor='white',
    framealpha=0.9,
    handlelength=1.5,
    handletextpad=0.8,
    borderaxespad=0.5,
    borderpad=0.4,
    labelspacing=0.6,
    **legend_kwargs
)
legend.get_frame().set_linewidth(0.5)

# --- Axis labels and title ---
suptitle_y = 0.99 if plot == 'report' else 1.0
fig.suptitle('Krill Growth Dynamic', y=suptitle_y, **suptitle_kwargs)
fig.text(0.5, suptitle_y - 0.08, f'Southern Ocean mean – Year {year_to_plot}',
         ha='center', style='italic', **label_kwargs)
ax.set_xlabel('Chlorophyll-a [mg/m³]', **label_kwargs)
ax.set_ylabel('Temperature [°C]', **label_kwargs)
# ax.set_ylim(-2, 3)
# ax.set_xlim(0, 1)
ax.set_yticks(np.arange(-2, 4, 1))
ax.tick_params(**tick_kwargs)
ax.autoscale(enable=False)

# --- Growth colorbar ---
cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04,
                    label='Growth [mm/d]', extend='both')
cbar.ax.yaxis.label.set_size(label_kwargs.get('fontsize', None))
cbar.ax.tick_params(**tick_kwargs)

# --- Final layout ---
plt.tight_layout()
plt.show()
# %%

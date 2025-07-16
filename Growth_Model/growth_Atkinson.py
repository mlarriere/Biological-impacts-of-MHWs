#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 31 March 09:54:34 2025

1st attempt - Growth model according to Atkinson et al. (2006) 

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


#%% Server 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% Figure settings
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
from growth_model import growth_Atkison2006  # import growth function

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
temp_avg_100m = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 181, 231, 1442)
temp_avg_100m = temp_avg_100m.rename({'year': 'years'})

# ==== Chla [mh Chla/m3] -- Surface chla (5m) - Austral summer - 60S
chla_surf_corrected = xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended.nc')) #chla_surf_allyears.nc
chla_surf_corrected = chla_surf_corrected.drop_dims('time')
chla_surf_corrected = chla_surf_corrected.rename({'year': 'years'})
chla_surf_corrected = chla_surf_corrected.rename({'day': 'days'})
chla_surf = chla_surf_corrected.assign_coords(days=("days", np.arange(365)))  #shape (40, 365, 231, 1442)

# Reformating - stacking time dimension -- shape (231, 1442, 14600)
temp_100m_stack = temp_avg_100m.stack(time= ['years', 'days'])
chla_surf_stack = chla_surf.stack(time= ['years', 'days']) 


#%% Plot ROMS outputs
# norm_chla=mcolors.Normalize(vmin=0, vmax=2)
# from matplotlib.colors import LinearSegmentedColormap
# colors = ["#216869", "#73A942", "#FBB02D"]  # Blue, Green, Yellow
# cmap_chla = LinearSegmentedColormap.from_list("blue_green_yellow", colors, N=256)

# # ---- Figure and Axes ----
# fig_width, fig_height = 6, 5
# fig = plt.figure(figsize=(fig_width *2, fig_height))
# gs = gridspec.GridSpec(1, 2, wspace=0.08, hspace=0.2)

# axs = []
# for j in range(2): 
#     ax = fig.add_subplot(gs[j], projection=ccrs.SouthPolarStereo())
#     axs.append(ax)

# # ---- Plot Definitions ----
# temp_jan2017_trimmed = temp_jan2017.isel(xi_rho=slice(0, -1))
# plot_data = [
#     (temp_jan2017_trimmed.avg_temp, "Mean Temperature (100m)", 'inferno', None),
#     (chla_jan2017.raw_chla, "Chlorophyll-a (5m)", cmap_chla, norm_chla)
# ]

# ims = []
# for i, (data, title, cmap, norm) in enumerate(plot_data):
#     ax = axs[i]
#     ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

#     # Circular boundary
#     import matplotlib.path as mpath
#     theta = np.linspace(0, 2 * np.pi, 100)
#     center, radius = [0.5, 0.5], 0.5
#     verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#     circle = mpath.Path(verts * radius + center)
#     ax.set_boundary(circle, transform=ax.transAxes)

#     # Coastlines and land
#     ax.coastlines(color='black', linewidth=1)
#     ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
#     ax.set_facecolor('lightgrey')

#     # Sector lines
#     for lon_line in [-90, 0, 120]:
#         ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
#                 color="#080808", linestyle='--', linewidth=1, zorder=5)

#     # Gridlines
#     gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=3)
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     gl.xlabel_style = {'size': 10}
#     gl.ylabel_style = {'size': 10}   
#     from matplotlib.ticker import MultipleLocator
#     gl.xlocator = MultipleLocator(60)
#     # gl.ylocator = MultipleLocator(5)  

#     # Plot data if available
#     if data is not None:
#         lon_name = [name for name in data.coords if 'lon' in name][0]
#         lat_name = [name for name in data.coords if 'lat' in name][0]
#         if data[lon_name].ndim == 1 and data[lat_name].ndim == 1:
#             lon2d, lat2d = np.meshgrid(data[lon_name], data[lat_name])
#         else:
#             lon2d, lat2d = data[lon_name], data[lat_name]

#         im = ax.pcolormesh(
#             data[lon_name], data[lat_name], data,
#             transform=ccrs.PlateCarree(),
#             cmap=cmap,
#             norm=norm,
#             shading='auto',
#             rasterized=True
#         )
#         ims.append(im)
#     else:
#         ims.append(None)

#     ax.set_title(title, fontsize=14)

# # ---- Colorbars ----
# # Temperature colorbar
# cbar_ax1 = fig.add_axes([0.13, 0.01, 0.32, 0.03])  # [left, bottom, width, height]
# cbar1 = fig.colorbar(ims[0], cax=cbar_ax1, orientation='horizontal', extend='both')
# cbar1.set_label("°C", fontsize=12)

# # Chlorophyll colorbar
# cbar_ax2 = fig.add_axes([0.56, 0.01, 0.32, 0.03])
# cbar2 = fig.colorbar(ims[1], cax=cbar_ax2, orientation='horizontal', extend='both')
# cbar2.set_label("mg/m³", fontsize=12)

# plt.suptitle("ROMS outputs \n January 2017", fontsize=18, y=1.05)
# plt.show()

# %% ================ Hypothetical Growth ================
# Investigate max/min values
max_obs_chla = chla_surf_stack.raw_chla.max() #5 mg/m3 Chla
min_obs_chla = chla_surf_stack.raw_chla.min() #0 mg/m3
max_obs_temp = temp_100m_stack.avg_temp.max() #7.3°C
min_obs_temp = temp_100m_stack.avg_temp.min() #-5.03°C

# 2D grid of all combinations
# chla_hyp= np.arange(min_obs_chla, 5.05, 0.01)
# temp_hyp = np.arange(min_obs_temp, max_obs_temp, 0.01)
chla_hyp= np.arange(0, 2, 0.01)
temp_hyp = np.arange(-4, 4, 0.01)

# Calcualting growth for all combination
CHLA, TEMP = np.meshgrid(chla_hyp, temp_hyp)
growth_hyp = growth_Atkison2006(CHLA, TEMP) #[mm] array of shape: (1234, 505)

# %% ================ Growth ================
file_growth = os.path.join(path_growth, "growth_Atkison2006_fullyr.nc")

if not os.path.exists(file_growth):
    # Calculating growth with Growth model - Eq. 4 (Atkinson et al. 2006)
    growth_da = growth_Atkison2006(chla_surf_stack.raw_chla, temp_100m_stack.avg_temp) #[mm]
    growth_redimensioned = growth_da.unstack('time')

    # Write to file
    growth_ds =xr.Dataset(
        {"growth": (["eta_rho", "xi_rho", "years", "days"], growth_redimensioned.data)},
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lon_rho.values),
            lat_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lat_rho.values),
            years = (['years'], growth_redimensioned.years.values),
            days = (['days'], growth_redimensioned.days.values),
            depth= '5'),
        attrs={"description": "Growth of krill based on Atkinson et al (2006) equation, model4 (sex and maturity considered) -- FULL YEAR"}
    )

    growth_ds.to_netcdf(file_growth, mode='w')
else:
    # Load data
    growth_redimensioned = xr.open_dataset(file_growth)

#%% ================ Select year of interest ================
target_start_year = 2018  #1980 #2000 #2010 #2018
target_end_year = target_start_year + 1

# %% Defining seasonal extent for growth (austral summer - early spring)
# def defining_season(ds, starting_year):
#     # Get the day-of-year slices
#     days_nov_dec = ds.sel(days=slice(304, 364), years=starting_year)
#     days_jan_apr = ds.sel(days=slice(0, 119), years=starting_year + 1)

#     # Combine while keeping doy as coords
#     ds_season = xr.concat([days_nov_dec, days_jan_apr], dim=xr.DataArray(np.concatenate([days_nov_dec['days'].values, days_jan_apr['days'].values]), dims="days", name="days"))

#     return ds_season

# %% ================ Growth in yearly context ================
# # Target year: start in July 1980 -> so need Jul-Dec 1980 and Jan-Jun 1981
# target_start_year = 2018  #1980 #2000 #2010 #2018
# target_end_year = target_start_year + 1

# # --- Full year data ---
# # Extract data across July–June (full year across two calendar years)
# CHLA_jul_dec = chla_surf.raw_chla.sel(years=target_start_year).isel(days=slice(181, 365)).mean(dim=["eta_rho", "xi_rho"])#, eta_rho=200, xi_rho=1000)
# TEMP_jul_dec = temp_avg_100m.avg_temp.sel(years=target_start_year).isel(days=slice(181, 365)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

# CHLA_jan_jun = chla_surf.raw_chla.sel(years=target_end_year).isel(days=slice(0, 181)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)
# TEMP_jan_jun = temp_avg_100m.avg_temp.sel(years=target_end_year).isel(days=slice(0, 181)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

# # Concatenate to get full July–June year
# CHLA_full = np.concatenate([CHLA_jul_dec, CHLA_jan_jun])
# TEMP_full = np.concatenate([TEMP_jul_dec, TEMP_jan_jun])

# # --- Growth season data ---
# # Identify Nov 1 – Apr 30 for the red line (Nov–Dec from current year, Jan–Apr from next year)
# CHLA_nov_dec = chla_surf.raw_chla.sel(years=target_start_year).isel(days=slice(305, 365)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)
# TEMP_nov_dec = temp_avg_100m.avg_temp.sel(years=target_start_year).isel(days=slice(305, 365)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

# CHLA_jan_apr = chla_surf.raw_chla.sel(years=target_end_year).isel(days=slice(0, 121)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)
# TEMP_jan_apr = temp_avg_100m.avg_temp.sel(years=target_end_year).isel(days=slice(0, 121)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

# CHLA_season = np.concatenate([CHLA_nov_dec.values, CHLA_jan_apr.values])
# TEMP_season = np.concatenate([TEMP_nov_dec.values, TEMP_jan_apr.values])

# #  ============ Plot ============
# plot='report'

# # Define figure size based on output type
# if plot == 'report':
#     fig_width = 6.3228348611
#     fig_height = fig_width*0.5
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
# else:  # 'slides'
#     fig, ax = plt.subplots(figsize=(10, 6))

# # Font size settings
# title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
# label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
# tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
# legend_kwargs = {'fontsize': 12} if plot == 'slides' else {}

# # --- Colormap normalization ---
# norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

# # --- Contour levels ---
# min_val = np.nanmin(growth_hyp)
# max_val = np.nanmax(growth_hyp)
# levels = np.arange(np.floor(min_val * 10) / 10, np.ceil(max_val * 10) / 10 + 0.05, 0.05)

# # --- Plot 0 contour ---
# zero_level = [0]
# manual_zero_pos = [(0.5, -0.5)]
# zero_contour = ax.contour(CHLA, TEMP, growth_hyp, levels=zero_level, colors='black',
#                           linewidths=0.8, linestyles='--', zorder=3)
# label_fontsize = 12 if plot == 'slides' else None
# ax.clabel(zero_contour, manual=manual_zero_pos, fmt="%.2f", inline=True,
#           fontsize=label_fontsize, colors='black')

# # --- Plot selected labeled contours ---
# levels_to_plot = [-0.2, -0.1, 0.1, 0.2]
# if plot == 'report':
#     manual_positions = {
#         -0.2: (0.33, -1.4), 
#         -0.1: (0.4, -0.7),
#          0.1: (0.7, -0.7),
#          0.2: (1.5, -0.4),
#     }
# else:  # slides
#     manual_positions = {
#         -0.2: (0.4, -1.65),
#         -0.1: (0.45, -0.9),
#          0.1: (0.7, -0.7),
#          0.2: (1.5, -0.4),
#     }

# for lvl in levels_to_plot:
#     contour = ax.contour(CHLA, TEMP, growth_hyp, levels=[lvl], colors='white',
#                          linewidths=0.8, linestyles='--', zorder=3)
#     try:
#         ax.clabel(contour, manual=[manual_positions[lvl]], fmt="%.2f", inline=True,
#                 fontsize=label_fontsize, colors='white')
#     except Exception as e:
#         print(f"Failed to label level {lvl}: {e}")

# # --- Pseudocolor background ---
# pcm = ax.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto',
#                     cmap='coolwarm_r', norm=norm, rasterized=True)

# # --- Full path (winter) ---
# valid_mask_full = ~np.isnan(CHLA_full) & ~np.isnan(TEMP_full)
# CHLA_full_clean = CHLA_full[valid_mask_full]
# TEMP_full_clean = TEMP_full[valid_mask_full]
# ax.plot(CHLA_full_clean, TEMP_full_clean, color='#4D7C8A', linewidth=2, zorder=4)

# # --- Seasonal path (summer + early spring) ---
# valid_mask_season = ~np.isnan(CHLA_season) & ~np.isnan(TEMP_season)
# CHLA_season_clean = CHLA_season[valid_mask_season]
# TEMP_season_clean = TEMP_season[valid_mask_season]
# ax.plot(CHLA_season_clean, TEMP_season_clean, color="#643888", linewidth=2.5, zorder=4)

# # --- Start & End markers ---
# from datetime import datetime, timedelta
# jul1 = datetime(target_start_year, 7, 1)
# dates_full = [jul1 + timedelta(days=int(i)) for i in range(len(CHLA_full))]
# dates_valid = np.array(dates_full)[valid_mask_full]
# start_date_str = dates_valid[0].strftime('%d %b')
# end_date_str = dates_valid[-1].strftime('%d %b')

# ax.scatter(CHLA_full_clean[0], TEMP_full_clean[0], facecolor='white', edgecolor='black',
#            s=90, zorder=5)
# ax.scatter(CHLA_full_clean[-1], TEMP_full_clean[-1], facecolor='black', edgecolor='white',
#            s=90, zorder=5)

# # --- Legend ---
# from matplotlib.lines import Line2D

# # Legend handles
# custom_lines = [
#     Line2D([0], [0], linestyle='None', marker='', label='Growth Paths:', color='black'),  # visible label text
#     Line2D([0], [0], color='#4D7C8A', lw=2, label='Winter'),
#     Line2D([0], [0], color='#643888', lw=2, label='Summer, Early Spring'),
# ]

# # Add start/end markers only for slides
# if plot == 'slides':
#     custom_lines += [
#         Line2D([0], [0], marker='o', markersize=7, markerfacecolor='white', markeredgecolor='black',
#                linestyle='None', label=f'Start ({start_date_str})'),
#         Line2D([0], [0], marker='o', markersize=7, markerfacecolor='black', markeredgecolor='white',
#                linestyle='None', label=f'End ({end_date_str})')
#     ]

# # Create legend
# legend = ax.legend(
#     handles=custom_lines,
#     loc='upper right',
#     # bbox_to_anchor=(1.02, 0.5) if plot == 'report' else (0.02, 0.98),
#     frameon=True,              # Enable frame
#     facecolor='white',         # White background
#     framealpha=0.9,            # Slight transparency
#     handlelength=1,        # Length lines
#     handletextpad=0.8,     # spacing between handle and text (default ~0.8)
#     borderaxespad=0.5,     # padding between axes and legend box
#     borderpad=0.4,         # padding inside the legend box
#     labelspacing=0.6,      # vertical spacing between entries (reduced from 1.2)
#     **legend_kwargs        # includes fontsize only for 'slides'
# )

# # --- Axis labels and title ---
# ax.set_title(f'Krill Growth from 1st July {target_start_year} to 30th June {target_end_year}', **title_kwargs)
# ax.set_xlabel('Chlorophyll-a [mg/m³]', **label_kwargs)
# ax.set_ylabel('Temperature [°C]', **label_kwargs)
# ax.set_yticks(np.arange(-4, 5, 1))
# ax.tick_params(**tick_kwargs)

# # --- Colorbar ---
# # Colorbar
# cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
# cbar.set_label('Growth [mm]', **label_kwargs)
# cbar.ax.tick_params(**tick_kwargs)

# # --- Final layout ---
# plt.tight_layout()

# # --- Output handling ---
# if plot == 'report':
#     outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/temp_chla_diagrams')
#     os.makedirs(outdir, exist_ok=True)
#     outfile = f"{target_start_year}_fullyear_{plot}.pdf"
#     # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
#     plt.show()
# else:
#     plt.show()

# %% ======================== Climatological growth (1980-2009) for 1 location ========================
# Seletina location and the climalotogical period
# loc_index = dict(eta_rho=200, xi_rho=1100) #close to peninsula
# loc_index = dict(eta_rho=230, xi_rho=990) #in the MHWs event (hotspot1)
loc_index = dict(eta_rho=220, xi_rho=1200) #in the wedell sea
lat_pt = float(chla_surf.lat_rho.sel(**loc_index).values)
lon_pt = float(chla_surf.lon_rho.sel(**loc_index).values)
region_label = f'Location ({abs(lat_pt):.2f}°S, {(-(lon_pt-360)) % 360:.2f}°W)'
chla_mean = lambda x: x.sel(**loc_index).isel(years=slice(0, 30)).mean(dim='years')
temp_mean = lambda x: x.sel(**loc_index).isel(years=slice(0, 30)).mean(dim='years')

# --- Full year data ---
# Extract data across July–June
CHLA_jul_dec = chla_mean(chla_surf.raw_chla.isel(days=slice(181, 365)))
TEMP_jul_dec = temp_mean(temp_avg_100m.avg_temp.isel(days=slice(181, 365)))

CHLA_jan_jun = chla_mean(chla_surf.raw_chla.isel(days=slice(0, 181)))
TEMP_jan_jun = temp_mean(temp_avg_100m.avg_temp.isel(days=slice(0, 181)))

# Concatenate to get full July–June year -- shape (365, )
CHLA_full = np.concatenate([CHLA_jul_dec, CHLA_jan_jun], axis=0)
TEMP_full = np.concatenate([TEMP_jul_dec, TEMP_jan_jun], axis=0)

# --- Growth season data ---
#  Extract data across Nov 1 – Apr 30 
CHLA_nov_dec = chla_mean(chla_surf.raw_chla.isel(days=slice(305, 365)))
TEMP_nov_dec = temp_mean(temp_avg_100m.avg_temp.isel(days=slice(305, 365)))

CHLA_jan_apr = chla_mean(chla_surf.raw_chla.isel(days=slice(0, 121)))
TEMP_jan_apr = temp_mean(temp_avg_100m.avg_temp.isel(days=slice(0, 121)))

CHLA_season = np.concatenate([CHLA_nov_dec.values, CHLA_jan_apr.values], axis=0)
TEMP_season = np.concatenate([TEMP_nov_dec.values, TEMP_jan_apr.values], axis=0)

# ============ Plot ============
plot='slides' #slides report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width*0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
else:  # 'slides'
    fig, ax = plt.subplots(figsize=(10, 6))

# Font size settings
# Font size settings
title_kwargs = {'fontsize': 15} if plot == 'slides' else {} #'fontsize': 14
label_kwargs = {'fontsize': 14} if plot == 'slides' else {} #'fontsize': 12
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {} #'labelsize': 10
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {} #'fontsize': 12
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}

# --- Colormap normalization ---
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

# --- Contour levels ---
min_val = np.nanmin(growth_hyp)
max_val = np.nanmax(growth_hyp)
levels = np.arange(np.floor(min_val * 10) / 10, np.ceil(max_val * 10) / 10 + 0.05, 0.05)

# --- Plot 0 contour ---
zero_level = [0]
manual_zero_pos = [(0.5, -0.5)]
zero_contour = ax.contour(CHLA, TEMP, growth_hyp, levels=zero_level, colors='black',
                          linewidths=0.8, linestyles='--', zorder=3)
label_fontsize = 12 if plot == 'slides' else None
ax.clabel(zero_contour, manual=manual_zero_pos, fmt="%.2f", inline=True,
          fontsize=label_fontsize, colors='black')

# --- Plot selected labeled contours ---
levels_to_plot = [-0.2, -0.1, 0.1, 0.2]
if plot == 'report':
    manual_positions = {
        -0.2: (0.33, -1.4), 
        -0.1: (0.4, -0.7),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
    }
else:  # slides
    manual_positions = {
        -0.2: (0.4, -1.65),
        -0.1: (0.45, -0.9),
         0.1: (0.7, -0.7),
         0.2: (1.5, -0.4),
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
pcm = ax.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto', cmap='coolwarm_r', norm=norm, rasterized=True)

# --- Full path (winter) ---
lw = 2 if plot == 'slides' else 1.5
valid_mask_full = ~np.isnan(CHLA_full) & ~np.isnan(TEMP_full)
CHLA_full_clean = CHLA_full[valid_mask_full]
TEMP_full_clean = TEMP_full[valid_mask_full]
ax.plot(CHLA_full_clean, TEMP_full_clean, color='#4D7C8A', linewidth=lw, zorder=4)

# --- Seasonal path (summer + early spring) ---
valid_mask_season = ~np.isnan(CHLA_season) & ~np.isnan(TEMP_season)
CHLA_season_clean = CHLA_season[valid_mask_season]
TEMP_season_clean = TEMP_season[valid_mask_season]
ax.plot(CHLA_season_clean, TEMP_season_clean, color="#643888", linewidth=lw, zorder=4)

# --- Start & End markers ---
s=60 if plot=='slides' else 30
from datetime import datetime, timedelta
jul1 = datetime(1980, 7, 1)
dates_full = [jul1 + timedelta(days=int(i)) for i in range(len(CHLA_full))]
dates_valid = np.array(dates_full)[valid_mask_full]
start_date_str = dates_valid[0].strftime('%d %b')
end_date_str = dates_valid[-1].strftime('%d %b')

ax.scatter(CHLA_full_clean[0], TEMP_full_clean[0], facecolor='white', edgecolor='black', s=s, zorder=5)
ax.scatter(CHLA_full_clean[-1], TEMP_full_clean[-1], facecolor='black', edgecolor='white', s=s, zorder=5)

# --- Legend ---
from matplotlib.lines import Line2D

# Legend handles
custom_lines = [
    Line2D([0], [0], linestyle='None', marker='', label='Growth Paths:', color='black'),  # visible label text
    Line2D([0], [0], color='#4D7C8A', lw=2, label='Winter'),
    Line2D([0], [0], color='#643888', lw=2, label='Summer, Early Spring'),
]

# Add start/end markers only for slides
if plot == 'slides':
    custom_lines += [
        Line2D([0], [0], marker='o', markersize=7, markerfacecolor='white', markeredgecolor='black',
               linestyle='None', label=f'Start ({start_date_str})'),
        Line2D([0], [0], marker='o', markersize=7, markerfacecolor='black', markeredgecolor='white',
               linestyle='None', label=f'End ({end_date_str})')
    ]

# Create legend
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
fig.suptitle(f'Climatological Krill Growth Dynamic', y=suptitle_y, **suptitle_kwargs)
fig.text(0.5, suptitle_y - 0.08, region_label, ha='center', **label_kwargs, style='italic')
ax.set_xlabel('Chlorophyll-a [mg/m³]', **label_kwargs)
ax.set_ylabel('Temperature [°C]', **label_kwargs)
ax.set_yticks(np.arange(-4, 6, 2))
ax.tick_params(**tick_kwargs)

# --- Colorbar ---
# Colorbar
cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04, label='Growth [mm/d]', extend='both')
cbar.ax.yaxis.label.set_size(label_kwargs.get('fontsize', None))
cbar.ax.tick_params(**tick_kwargs)

# --- Final layout ---
plt.tight_layout()

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/temp_chla_diagrams')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"climatological_growth_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/temp_chla_diagrams/climatological_growth_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %% ======================== Growth in yearly context - 4 subplots ========================
years = [1980, 2000, 2010, 2018]
plot = 'slides'  # report slides
spatial_mode = '1loc' # '1loc' #'SouthernOcean' #atlantic_sector

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611 
    fig_height = 6.3228348611 /2
else:  # 'slides'
    fig_width = 16
    fig_height = 9

fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height), constrained_layout=True,
                        sharex='col', sharey='row')
axs = axs.flatten()

# Font size settings
title_kwargs = {'fontsize': 15} if plot == 'slides' else {} #'fontsize': 14
label_kwargs = {'fontsize': 14} if plot == 'slides' else {} #'fontsize': 12
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {} #'labelsize': 10
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {} #'fontsize': 12
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}

# Define label for spatial mode
if spatial_mode == 'SouthernOcean':
    region_label = 'Southern Ocean'
elif spatial_mode == 'atlantic_sector':
    region_label = 'Atlantic Sector (270°–360°E)'
else:
    # Seletina location and the climalotogical period
    # loc_index = dict(eta_rho=200, xi_rho=1100) #close to peninsula
    # loc_index = dict(eta_rho=230, xi_rho=990) #in the MHWs event (hotspot1)
    loc_index = dict(eta_rho=220, xi_rho=1200) #in the wedell sea    lat_pt = float(chla_surf.lat_rho.sel(**loc_index).values)
    lon_pt = float(chla_surf.lon_rho.sel(**loc_index).values)
    region_label = f'Location ({abs(lat_pt):.2f}°S, {(-(lon_pt-360)) % 360:.2f}°W)'

for i, target_start_year in enumerate(years):
    ax = axs[i]

    # Define mean or select index depending on full_years
    if spatial_mode == 'SouthernOcean':
        chla_mean = lambda x: x.mean(dim=["eta_rho", "xi_rho"])
        temp_mean = lambda x: x.mean(dim=["eta_rho", "xi_rho"])

    elif spatial_mode == 'atlantic_sector':
        # Define bounds in 0–360 convention
        lon_min, lon_max = 270, 360

        # Get lon/lat mask for Atlantic sector
        lon_vals = chla_surf.lon_rho
        lat_vals = chla_surf.lat_rho

        sector_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)

        def chla_mean(x):
            return x.where(sector_mask).mean(dim=["eta_rho", "xi_rho"], skipna=True)

        def temp_mean(x):
            return x.where(sector_mask).mean(dim=["eta_rho", "xi_rho"], skipna=True)

    else:
        chla_mean = lambda x: x.sel(**loc_index)
        temp_mean = lambda x: x.sel(**loc_index)


    # --- Extract data for this year ---
    target_end_year = target_start_year + 1

    CHLA_jul_dec = chla_mean(chla_surf.raw_chla.sel(years=target_start_year).isel(days=slice(181, 365)))
    TEMP_jul_dec = temp_mean(temp_avg_100m.avg_temp.sel(years=target_start_year).isel(days=slice(181, 365)))

    CHLA_jan_jun = chla_mean(chla_surf.raw_chla.sel(years=target_end_year).isel(days=slice(0, 181)))
    TEMP_jan_jun = temp_mean(temp_avg_100m.avg_temp.sel(years=target_end_year).isel(days=slice(0, 181)))

    CHLA_full = np.concatenate([CHLA_jul_dec, CHLA_jan_jun])
    TEMP_full = np.concatenate([TEMP_jul_dec, TEMP_jan_jun])

    CHLA_nov_dec = chla_mean(chla_surf.raw_chla.sel(years=target_start_year).isel(days=slice(305, 365)))
    TEMP_nov_dec = temp_mean(temp_avg_100m.avg_temp.sel(years=target_start_year).isel(days=slice(305, 365)))

    CHLA_jan_apr = chla_mean(chla_surf.raw_chla.sel(years=target_end_year).isel(days=slice(0, 121)))
    TEMP_jan_apr = temp_mean(temp_avg_100m.avg_temp.sel(years=target_end_year).isel(days=slice(0, 121)))

    CHLA_season = np.concatenate([CHLA_nov_dec.values, CHLA_jan_apr.values])
    TEMP_season = np.concatenate([TEMP_nov_dec.values, TEMP_jan_apr.values])

    # --- Colormap normalization (example - adjust if growth_hyp changes per year) ---
    norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

    # --- Plot zero contour ---
    zero_level = [0]
    manual_zero_pos = [(0.5, -0.5)]
    zero_contour = ax.contour(CHLA, TEMP, growth_hyp, levels=zero_level, colors='black',
                              linewidths=0.8, linestyles='--', zorder=3)
    label_fontsize = 12 if plot == 'slides' else None
    ax.clabel(zero_contour, manual=manual_zero_pos, fmt="%.2f", inline=True,
              fontsize=label_fontsize, colors='black')

    # --- Plot selected labeled contours ---
    levels_to_plot = [-0.2, -0.1, 0.1, 0.2]
    if plot == 'report':
        manual_positions = {
            -0.2: (0.33, -1.4), 
            -0.1: (0.4, -0.7),
             0.1: (0.7, -0.7),
             0.2: (1.5, -0.4),
        }
    else:  # slides
        manual_positions = {
            -0.2: (0.36, -1.65),
            -0.1: (0.43, -0.9),
             0.1: (0.7, -0.7),
             0.2: (1.5, -0.4),
        }

    for lvl in levels_to_plot:
        contour = ax.contour(CHLA, TEMP, growth_hyp, levels=[lvl], colors='white',
                             linewidths=0.8, linestyles='--', zorder=3)
        try:
            ax.clabel(contour, manual=[manual_positions[lvl]], fmt="%.2f", inline=True,
                      fontsize=label_fontsize, colors='white')
        except Exception as e:
            print(f"Failed to label level {lvl}: {e}")

    # --- Pcolormesh background ---
    pcm = ax.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto',
                        cmap='coolwarm_r', norm=norm, rasterized=True)

    # --- Plot full and seasonal paths ---
    lw = 2 if plot == 'slides' else 1

    valid_mask_full = ~np.isnan(CHLA_full) & ~np.isnan(TEMP_full)
    ax.plot(CHLA_full[valid_mask_full], TEMP_full[valid_mask_full], color='#4D7C8A', linewidth=lw, zorder=4)

    valid_mask_season = ~np.isnan(CHLA_season) & ~np.isnan(TEMP_season)
    ax.plot(CHLA_season[valid_mask_season], TEMP_season[valid_mask_season], color="#643888", linewidth=lw, zorder=4)

    # --- Start & End markers ---
    s=60 if plot=='slides' else 30
    jul1 = datetime(target_start_year, 7, 1)
    dates_full = [jul1 + timedelta(days=int(i)) for i in range(len(CHLA_full))]
    dates_valid = np.array(dates_full)[valid_mask_full]
    start_date_str = dates_valid[0].strftime('%d %b')
    end_date_str = dates_valid[-1].strftime('%d %b')
    ax.scatter(CHLA_full[valid_mask_full][0], TEMP_full[valid_mask_full][0], facecolor='white', edgecolor='black', s=s, zorder=5)
    ax.scatter(CHLA_full[valid_mask_full][-1], TEMP_full[valid_mask_full][-1], facecolor='black', edgecolor='white', s=s, zorder=5)

    # --- Legend ---
    custom_lines = [
        Line2D([0], [0], linestyle='None', marker='', label='Growth Paths:', color='black'),
        Line2D([0], [0], color='#4D7C8A', lw=2, label='Winter'),
        Line2D([0], [0], color='#643888', lw=2, label='Summer, Early Spring'),
    ]
    if plot == 'slides':
        custom_lines += [
            Line2D([0], [0], marker='o', markersize=7, markerfacecolor='white', markeredgecolor='black',
                   linestyle='None', label=f'Start ({start_date_str})'),
            Line2D([0], [0], marker='o', markersize=7, markerfacecolor='black', markeredgecolor='white',
                   linestyle='None', label=f'End ({end_date_str})')
        ]

    # Add legend outside the plot
    if plot == 'report':
        legend_box = (0.45, -0.1)
    else:
        legend_box = (0.5, -0.07)

    legend=fig.legend(handles=custom_lines, loc='lower center',
           bbox_to_anchor=legend_box,  # centered, slightly below the figure
           ncol=len(custom_lines),       # all legend items in one row
           frameon=True, facecolor='white', framealpha=0.9,
           handlelength=1, handletextpad=0.8, borderaxespad=0.5, borderpad=0.4,
           labelspacing=0.6, **legend_kwargs)
    legend.get_frame().set_linewidth(0.5)  # Default is ~1.0; reduce for thinner box

    
    # --- Labels and title ---
    ax.set_title(f'July {target_start_year} to June {target_end_year}', **title_kwargs)
    ax.set_yticks(np.arange(-4, 6, 2))
    ax.tick_params(**tick_kwargs)
    if i % 2 == 0:  
        ax.set_ylabel('Temperature [°C]', **label_kwargs)
    else:
        ax.set_ylabel('')  # No label on right column

    # Only set x-label on bottom row (row 1)
    if i // 2 == 1:
        ax.set_xlabel('Chlorophyll-a [mg/m³]', **label_kwargs)
    else:
        ax.set_xlabel('')  # No label on top row


if plot == 'report':
    suptitle_y = 1.15
else:
    suptitle_y = 1.05
fig.suptitle(f'Annual Krill Growth Dynamics', y=suptitle_y, **suptitle_kwargs)
fig.text(0.5, suptitle_y - 0.1, region_label, ha='center', **label_kwargs, style='italic')

# --- Shared colorbar ---
cbar = fig.colorbar(pcm, ax=axs, orientation='vertical', fraction=0.04, pad=0.04, label='Growth [mm/d]', extend='both')
cbar.ax.yaxis.label.set_size(label_kwargs.get('fontsize', None))

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/temp_chla_diagrams')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"4years_fullyear_{spatial_mode}_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    plt.show()

# %% =============== Plot selected location on the map ===============
# --- Preformatting data ---
num_days = 365
start_day = 180  # July 1 (0-based index)

# --- Select year and day ---
year = 2018
day = 14  # 0-based index → Jan 15
plot_date = datetime.date(int(year), 1, 15)

chla_slice = chla_surf.raw_chla.sel(years=year).isel(days=day) # Extract data
lon = chla_surf.lon_rho
lat = chla_surf.lat_rho

# Location index
loc_index = dict(eta_rho=220, xi_rho=1200)
loc_lon = chla_surf.lon_rho.sel(**loc_index).values
loc_lat = chla_surf.lat_rho.sel(**loc_index).values


plot = 'report'  # report slides

# --- Layout Setup ---
if plot == 'report':
    fig_width = 6.3228348611  # half textwidth in inches
    fig_height = fig_width/2     # in inches
    fig = plt.figure(figsize=(fig_width, fig_height))
else:  # slides
    fig_width = 10  # full width
    fig_height = 5    # square figure
    fig = plt.figure(figsize=(fig_width, fig_height))

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

# --- Font size settings ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
ticklabelsize = 13 if plot == 'slides' else 9
gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}

# --- 1. TIME SERIES PLOT ---
ax_ts = fig.add_subplot(gs[0])

years_to_plot = chla_surf_corrected.years.values[:30]
norm = mcolors.Normalize(vmin=years_to_plot.min(), vmax=years_to_plot.max())
cmap = plt.cm.viridis

# --- Plot data ---
x_vals = np.arange(num_days)
for y in range(30):
    chla_raw = chla_surf_corrected['raw_chla'].isel(years=y, eta_rho=220, xi_rho=1200).values
    valid_mask = ~np.isnan(chla_raw)
    chla_shifted = np.concatenate([chla_raw[start_day:], chla_raw[:start_day]])
    valid_mask_shifted = np.concatenate([valid_mask[start_day:], valid_mask[:start_day]])

    label = 'Sea Ice Period' if y == 0 else None
    ax_ts.fill_between(x_vals, 0, 10, where=~valid_mask_shifted, color='#ADE8F4', alpha=0.2, label=label)
    ax_ts.plot(x_vals, chla_shifted, color=cmap(norm(chla_surf_corrected.years.values[y])))


# --- Labels and legend ---
days_in_month = [31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30]
cumulative_days = np.cumsum([0] + days_in_month[:-1])
month_names = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
ax_ts.set_xticks(cumulative_days)
ax_ts.set_xticklabels(month_names)
ax_ts.tick_params(axis='x', labelsize=ticklabelsize)
for label in ax_ts.get_xticklabels():
    label.set_rotation(60)

ax_ts.set_xlabel('Months', **label_kwargs)
ax_ts.set_ylim(0, 3)

ax_ts.set_ylabel('Chla [mg/m³]', **label_kwargs)
ax_ts.set_xlim(0, num_days - 1)

ax_ts.tick_params(axis='y', labelsize=ticklabelsize)
ax_ts.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=True)

# --- Titla ---
ax_ts.set_title('Time Series over the Climatological Period', **subtitle_kwargs)

# --- Colorbar ---
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ts = fig.colorbar(sm, ax=ax_ts, orientation='vertical', pad=0.02)
cbar_ts.set_label('Year', **label_kwargs)
ax_ts.set_aspect('auto') 

# ------------ 2. MAP PLOT ------------
ax_map = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
ax_map.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.5 + 0.5)
ax_map.set_boundary(circle, transform=ax_map.transAxes)

# Features
ax_map.coastlines(color='black', linewidth=1, zorder=4)
ax_map.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
ax_map.set_facecolor('lightgrey')

for lon_line in [-90, 0, 120]:
    ax_map.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(), color="#080808", linestyle='--', linewidth=1, zorder=5)

gl = ax_map.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=3)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# CHLA Colormap
vmin, vmax = 0, 1
colors = ["#0E1B11", "#4A8956", "#73A942", "#E7D20D", "#FBB02D"]
cmap_var = LinearSegmentedColormap.from_list("chla_colormap", list(zip(np.linspace(0, 1, len(colors)), colors)), N=256)
norm = mcolors.Normalize(vmin=0, vmax=2)

pcm = ax_map.pcolormesh(lon, lat, chla_slice, transform=ccrs.PlateCarree(), rasterized=True,
                        cmap=cmap_var, norm= norm, shading='auto', zorder=1)

# Star at location
marker_kwargs = {'markersize': 12} if plot == 'slides' else {'markersize': 5}
ax_map.plot(loc_lon, loc_lat, marker='*', color='red', **marker_kwargs, transform=ccrs.PlateCarree(), zorder=6)

cbar_map = fig.colorbar(pcm, ax=ax_map, orientation='vertical', shrink=0.6, pad=0.09, extend='max')
cbar_map.set_label('Chla [mg/m³]', **label_kwargs)

# Convert the date into a readable string
ax_map.set_title(f'Snapshot on  {plot_date.strftime("%d %B")} {year}', fontweight='bold', **subtitle_kwargs)

suptitle_y = 0.98 if plot == 'slides' else 1.02
fig.suptitle(f"Surface Chla at ({abs(loc_lat):.2f}°S, {(-(loc_lon-360)) % 360:.2f}°W)", fontweight='bold', y=suptitle_y, **maintitle_kwargs)


# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/temp_chla_diagrams')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"chla_ts_loc_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/temp_chla_diagrams/chla_ts_loc_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()



# %% Select growth season - all years
def extract_one_season_pair(args):
    ds_y, ds_y1, y = args
    try:
        days_nov_dec = ds_y.sel(days=slice(304, 364))
        days_jan_apr = ds_y1.sel(days=slice(0, 119))

        combined_days = np.concatenate(
            [days_nov_dec['days'].values, days_jan_apr['days'].values]
        )
        season = xr.concat([days_nov_dec, days_jan_apr],
                           dim=xr.DataArray(combined_days, dims="days", name="days"))
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


growth_season_file = os.path.join(path_growth, "growth_Atkison2006_seasonal.nc")
if not os.path.exists(growth_season_file):
    # growth_seasons = define_season_all_years(growth_redimensioned.growth) 
    season_ds = define_season_all_years_parallel(growth_redimensioned.growth, max_workers=30)
    season_ds = season_ds.rename({'season_year': 'season_year_temp'})
    season_ds = season_ds.drop_vars('years')
    season_ds = season_ds.rename({'season_year_temp': 'years'})
    season_ds.attrs['description'] = ("Krill growth estimates during the growth season (Nov 1 – Apr 30) based on Atkinson et al. (2006), model 4")
    growth_seasons_ds = season_ds.to_dataset(name="growth") #to dataset
    growth_seasons_ds.to_netcdf(growth_season_file)
else:
    growth_seasons=xr.open_dataset(growth_season_file) 

chla_filtered_seasons_file = os.path.join(path_growth_inputs, "chla_surf_allyears_detrended_seasonal.nc")
if not os.path.exists(chla_filtered_seasons_file):
    chla_filtered_seasons = define_season_all_years_parallel(chla_surf.raw_chla, max_workers=6)
    chla_filtered_seasons = chla_filtered_seasons.rename({'season_year': 'season_year_temp'})
    chla_filtered_seasons = chla_filtered_seasons.drop_vars('years')
    chla_filtered_seasons = chla_filtered_seasons.rename({'season_year_temp': 'years'})
    chla_filtered_seasons.attrs['description'] = ("Surface Chla (detrented) during the growth season (Nov 1 – Apr 30)")
    chla_filtered_seasons_ds = chla_filtered_seasons.to_dataset(name="chla") #to dataset
    chla_filtered_seasons_ds.to_netcdf(chla_filtered_seasons_file)
else:
    chla_filtered_seasons=xr.open_dataset(chla_filtered_seasons_file) 

temp_avg_100m_seasons_file = os.path.join(path_growth_inputs, "temp_avg100m_allyears_seasonal.nc")
if not os.path.exists(temp_avg_100m_seasons_file):
    temp_avg_100m_seasons = define_season_all_years_parallel(temp_avg_100m.avg_temp, max_workers=6)
    temp_avg_100m_seasons = temp_avg_100m_seasons.rename({'season_year': 'season_year_temp'})
    temp_avg_100m_seasons = temp_avg_100m_seasons.drop_vars('years')
    temp_avg_100m_seasons = temp_avg_100m_seasons.rename({'season_year_temp': 'years'})
    temp_avg_100m_seasons.attrs['description'] = ("Avg temperature (first 100m) during the growth season (Nov 1 – Apr 30)")
    temp_avg_100m_seasons_ds = temp_avg_100m_seasons.to_dataset(name="avg_temp") #to dataset
    temp_avg_100m_seasons_ds.to_netcdf(temp_avg_100m_seasons_file)
else:
    temp_avg_100m_seasons=xr.open_dataset(temp_avg_100m_seasons_file) 

#  %% Plot growth season for 1 year
# growth_seasons2017 = growth_seasons.isel(years=37)

# # Select January
# growth_ROMS_2017_jan = growth_seasons2017.sel(days=slice(0,30)).mean(dim='days') #shape: (eta_rho, xi_rho) : (231, 1442)

# # Setup figure
# fig_width = 6.3228348611  # inches = \textwidth
# fig_height = fig_width
# fig = plt.figure(figsize=(fig_width, fig_height))  # Only 1 plot now
# gs = gridspec.GridSpec(1, 1)

# axs = []
# ax = fig.add_subplot(gs[0, 0], projection=ccrs.SouthPolarStereo())
# axs.append(ax)

# # Plot configuration
# growth_ROMS_2017_jan_trimmed = growth_ROMS_2017_jan.isel(xi_rho=slice(0, -1))
# plot_data = [
#     (growth_ROMS_2017_jan_trimmed.growth, "Mean Growth")
# ]

# # Color normalization
# vmin, vmax = -0.2, 0.2
# norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
# cmap = 'PuOr_r'

# # -----------------------------
# # Plotting
# # -----------------------------
# ims = []
# for data, title in plot_data:
#     ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

#     # Circular boundary
#     theta = np.linspace(0, 2 * np.pi, 100)
#     center, radius = [0.5, 0.5], 0.5
#     verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#     circle = mpath.Path(verts * radius + center)
#     ax.set_boundary(circle, transform=ax.transAxes)

#     # Map features
#     ax.coastlines(color='black', linewidth=1, zorder=4)
#     ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
#     ax.set_facecolor('lightgrey')

#     # Sector lines
#     for lon_line in [-90, 0, 120]:
#         ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
#                 color="#080808", linestyle='--', linewidth=1, zorder=5)

#     # Gridlines
#     gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=2)
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     gl.xlabel_style = {'size': 10}
#     gl.ylabel_style = {'size': 10}
#     gl.xformatter = LongitudeFormatter()
#     gl.yformatter = LatitudeFormatter()

#     # Plot data
#     im = ax.pcolormesh(
#         growth_seasons2017.lon_rho,
#         growth_seasons2017.lat_rho,
#         data,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap,
#         norm=norm,
#         shading='auto',
#         zorder=1,
#         rasterized=True
#     )
#     ims.append(im)
#     # ax.set_title(title, fontsize=15)

# # -----------------------------
# # Colorbar
# # -----------------------------
# cbar_ax = fig.add_axes([0.25, 0.01, 0.5, 0.03])  # [left, bottom, width, height]
# tick_positions = [-0.2, -0.1, 0.0, 0.1, 0.2]
# cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', ticks=tick_positions, extend='both')
# cbar.set_label("Growth [mm]", fontsize=14)
# cbar.ax.tick_params(labelsize=13)

# plt.suptitle("Growth with ROMS - January 2017", fontsize=18, y=0.98)
# plt.show()

#%% ==== Periods comparison ====
import matplotlib.colors as mcolors
def plot_comparison(varname, ds, cmap_var=None, ticks=None, cbar_label='', plot='slides'):
    # === Layout config ===
    if plot == 'report':
        fig_width = 6.3228348611
        fig_height = 9.3656988889
        fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})
        # fig.subplots_adjust(hspace=0.01) 
    else:
        fig_width = 16
        fig_height = 8
        fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})

    title_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 11}
    label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
    tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 10}
    suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 13, 'fontweight': 'bold'}

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    # Time periods
    data_1980_2009 = ds.isel(years=slice(0, 30)).mean(dim=('years', 'days'))
    data_2010_2019 = ds.isel(years=slice(30, 40)).mean(dim=('years', 'days'))
    data_diff = data_2010_2019 - data_1980_2009

    # Set normalization
    if varname == 'growth':
        norm_main = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
        extend_var ='both'
    elif varname == 'temp':
        norm_main = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
        extend_var ='both'
    elif varname == 'chla':
        vmax = data_1980_2009.max()  # Max value for chla colormap
        norm_main = mcolors.Normalize(vmin=0, vmax=1)
        extend_var ='max'
    else:
        norm_main = None

    # Difference normalization (centered at zero for difference)
    abs_diff_max = np.max(np.abs(data_diff))
    norm_diff = mcolors.TwoSlopeNorm(vmin=-abs_diff_max, vcenter=0, vmax=abs_diff_max)
    cmap_diff = plt.cm.RdBu_r

    # Titles and datasets
    if plot == 'report':
        font_size_macro = r'\small'
    else:  # slides
        font_size_macro = r'\large'

    plot_data = [
        (data_1980_2009, f"Climatological Growth\n{font_size_macro}{{(1980\\,\\textendash\\,2009)}}", norm_main, cmap_var),
        (data_2010_2019, f"Recent Growth\n{font_size_macro}{{(2010\\,\\textendash\\,2018)}}",norm_main, cmap_var),
        (data_diff, f"Difference\n{font_size_macro}{{(recent\\,\\textendash\\,climatological)}}", norm_diff, cmap_diff),
        ]



    # Plotting data
    scs = []  # List to hold the scatter plot objects for each subplot
    for ax, (data, title, norm, cmap_used) in zip(axs, plot_data):
        sc = data.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            x="lon_rho", y="lat_rho",
            add_colorbar=False, cmap=cmap_used, norm=norm, zorder=1, rasterized=True)
        scs.append(sc)  # Store the plot object
        
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Draw the land feature after the pcolormesh
        ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
        ax.coastlines(color='black', linewidth=1)
        ax.set_facecolor('#F6F6F3')
        
        # Sector boundaries
        for lon in [-90, 120, 0]:
            ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808',
                    linestyle='--', linewidth=1)

        # Gridlines
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
        gl.xlabel_style = gridlabel_kwargs
        gl.ylabel_style = gridlabel_kwargs
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        if plot == 'report':
            ax.set_title(title, x=0.5, y=1.05, ha='center', **title_kwargs)
        else:
            ax.set_title(title, y=1.1, **title_kwargs)
    

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.4)

    # Colorbar for the first and second subplots
    if varname in ['chla', 'growth', 'temp']:
        ticks = ticks
    else:
        ticks = np.arange(np.floor(norm.vmin)-1, np.ceil(norm.vmax) + 1, 1)
    
    if plot == 'report':
        # Colorbar for top 2 plots (ax[0] and ax[1])
        pos1 = axs[0].get_position()
        pos2 = axs[1].get_position()
        # Combine colorbar height across both axes
        cbar_ax1 = fig.add_axes([pos2.x1 + 0.05, pos2.y0, 0.015, pos1.y1 - pos2.y0])
        cbar1 = fig.colorbar(scs[0], cax=cbar_ax1, cmap=cmap_var, ticks=ticks, extend=extend_var)
        cbar1.set_label(cbar_label, **label_kwargs)
        cbar1.ax.tick_params(**tick_kwargs)

        # Colorbar for third (difference) plot
        pos3 = axs[2].get_position()
        cbar_ax2 = fig.add_axes([pos3.x1 + 0.05, pos3.y0, 0.015, pos3.height ])
        cbar2 = fig.colorbar(scs[2], cax=cbar_ax2, cmap=cmap_diff, extend='both')
        cbar2.set_label("$\Delta$ Growth [mm/d]", **label_kwargs)
        cbar2.ax.tick_params(**tick_kwargs)
    else:
        pos1 = axs[1].get_position()
        cbar_ax1 = fig.add_axes([pos1.x1 + 0.01, pos1.y0, 0.01, pos1.height])
        cbar1 = fig.colorbar(scs[0], cax=cbar_ax1, cmap=cmap_var, ticks=ticks, extend=extend_var)
        cbar1.set_label(cbar_label, **label_kwargs)
        cbar1.ax.tick_params(**tick_kwargs)

        # Second colorbar (for the difference)
        pos2 = axs[2].get_position()
        cbar_ax2 = fig.add_axes([pos2.x1 + 0.045, pos2.y0, 0.01, pos2.height])
        cbar2 = fig.colorbar(scs[2], cax=cbar_ax2, cmap=cmap_diff, extend='both')
        cbar2.set_label("$\Delta$ Growth [mm/d]", **label_kwargs)
        cbar2.ax.tick_params(**tick_kwargs)

    # --- Axis labels and title ---
    if plot == 'report':
        suptitle_y = 1
        fig.suptitle('Antarctic Krill Growth - Southern Ocean', x=0.55, y=suptitle_y, **suptitle_kwargs)
        fig.text(0.55, suptitle_y - 0.03, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', **label_kwargs, style='italic')

    else:
        suptitle_y = 0.9
        fig.suptitle(f'Antarctic Krill Growth - Southern Ocean', y=suptitle_y, **suptitle_kwargs)
        fig.text(0.5, suptitle_y - 0.055, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', **label_kwargs, style='italic')

    # --- Output handling ---
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/eq_decomposition')
    os.makedirs(outdir, exist_ok=True)
    if plot == 'report':
        outfile = f"{varname}_diff_{plot}.pdf"
        # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
        plt.show()
    else:    
        # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/eq_decomposition/{varname}_diff_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
        plt.show()

# Choose variable to plot
variable = 'growth'  #'growth', 'temp', 'chla'

if variable == 'growth':
    ds = growth_seasons.isel(xi_rho=slice(0, -1)).growth
    cmap_var = 'PuOr_r'
    label = 'Growth [mm/d]'
    ticks = [-0.2, -0.1, 0, 0.1, 0.2]

elif variable == 'temp':
    ds = temp_avg_100m_seasons.isel(xi_rho=slice(0, -1)).avg_temp
    vmin, vmax = -4, 4  # Symmetric -- centered at 0
    colors = ["#0A3647", "#669BBC", "#FFFFFF", "#EE9B00", "#AE2012"] #BB3E03
    color_positions = np.linspace(vmin, vmax, len(colors))
    normalized_positions = (color_positions - vmin) / (vmax - vmin)  # Normalize to [0, 1]
    cmap_var = LinearSegmentedColormap.from_list("thermal_centered", list(zip(normalized_positions, colors)), N=256)
    label = 'Temperature [°C]'
    ticks = [-2, -1, 0, 1, 2]

elif variable == 'chla':
    ds = chla_filtered_seasons.isel(xi_rho=slice(0, -1)).chla    
    vmin, vmax = 0, 1
    colors = ["#0E1B11", "#4A8956", "#73A942", "#E7D20D", "#FBB02D"]
    color_positions = np.linspace(vmin, vmax, len(colors))
    normalized_positions = (color_positions - vmin) / (vmax - vmin)  # Normalize to [0, 1]
    cmap_var = LinearSegmentedColormap.from_list("blue_green_yellow_buffered", list(zip(normalized_positions, colors)), N=256)
    label = 'Chla [mg/m³]'
    ticks =  [0, 0.5, 1] # 2.5, 3, 3.5, 4, 4.5 ,5

plot_comparison(variable, ds, cmap_var=cmap_var, ticks=ticks, cbar_label=label, plot='report') #report slides

# %% =========================== Growth during MHW events ===========================
# ==== MHWs events detected -- only surface
mhw_det = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))) #shape: (39, 181, 231, 1442)
growth_seasons= xr.open_dataset(os.path.join(path_growth, "growth_Atkison2006_seasonal.nc")) #shape: (39, 181, 231, 1442)

# --------- MHWs ---------
# Write or load
growth_mhw_file= os.path.join(path_growth, "growth_Atkison2006_mhw.nc")

if not os.path.exists(growth_mhw_file):

    variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
    growth_var_names = ['growth_1deg', 'growth_2deg', 'growth_3deg', 'growth_4deg']

    growth_mhw_dict = {}

    for mhw_var, growth_name in zip(variables, growth_var_names):
        print(f'------------{mhw_var}------------')
        # Testing
        # mhw_var='det_1deg'
        # growth_name = 'growth_1deg'

        # MHW mask for current threshold and align MHW years to match growth_seasons (1980-2018)
        duration_mask = mhw_det['duration'] >= 30 #DataArray (bool) -- shape: (39, 181, 231, 1442)
        det_mask = mhw_det[mhw_var] == 1 #DataArray (bool) -- shape: (39, 181, 231, 1442)

        # Ensure both are boolean, then combine
        mhw_mask = duration_mask & det_mask #DataArray (bool) -- shape: (39, 181, 231, 1442)

        # Growth during MHWs
        mhw_mask_clean = mhw_mask.drop_vars('days')
        mhw_mask_clean = mhw_mask_clean.rename({'days_of_yr': 'days'})
        mhw_mask_clean = mhw_mask_clean.assign_coords(years=mhw_mask_clean.years + 1980)
        growth_masked = growth_seasons['growth'].where(mhw_mask_clean) #DataArray (float) -- shape: (39, 181, 231, 1442)
        print("All values are NaN:", np.isnan(growth_masked).all().item())

        # Store DataArray to dictionnary
        growth_mhw_dict[growth_name] = growth_masked

        print(f'done with {growth_name}')

    # To Dataset
    growth_mhw_combined = xr.Dataset(
        data_vars=growth_mhw_dict,
        coords=growth_seasons.coords,
        attrs={
            "description": "Growth of krill during MHWs for the growth season (Nov 1 – Apr 30).",
            "depth": "5m.",
            "growth_ideg": (
                "Growth under different MHW intensity "
                "(MHWs defined as exceeding both the extended absolute threshold and the 90th percentile as well as lasting more than 30days)."
            )
        }
    )

    # Save output
    growth_mhw_combined.to_netcdf(growth_mhw_file, mode='w')
    
else:
    # Load data
    growth_mhw_combined = xr.open_dataset(growth_mhw_file)

# --------- NO MHWs ---------
# Write or load
growth_no_mhw_file = os.path.join(path_growth, "growth_Atkison2006_nonMHW.nc")

if not os.path.exists(growth_no_mhw_file):

    non_mhw_mask = mhw_det['duration'] == 0
    
    # Growth during non MHWs
    non_mhw_mask_clean = non_mhw_mask.drop_vars('days')
    non_mhw_mask_clean = non_mhw_mask_clean.rename({'days_of_yr': 'days'})
    non_mhw_mask_clean = non_mhw_mask_clean.assign_coords(years=non_mhw_mask_clean.years + 1980)
    growth_no_mhw = growth_seasons['growth'].where(non_mhw_mask_clean) #DataArray (float) -- shape: (39, 181, 231, 1442)
    print("All values are NaN:", np.isnan(growth_no_mhw).all().item())

    # Count how many valid time steps we have per grid cell
    valid_counts = growth_no_mhw.count(dim=('years', 'days'))
    max_val = valid_counts.max().item()

    # To Dataset
    growth_no_mhw_ds = xr.Dataset(
        data_vars={"growth_noMHW": growth_no_mhw},
        coords=growth_seasons.coords,
        attrs={
            "description": "Krill growth during **non-MHW** periods across the growth season (Nov 1 – Apr 30).",
            "depth": "5m",
            "growth_noMHW": (
                "Growth when no MHWs are detected, i.e. when mhw duration=0."
            )
        }
    )

    # Save output
    growth_no_mhw_ds.to_netcdf(growth_no_mhw_file, mode='w')

else:
    growth_no_mhw_ds= xr.open_dataset(growth_no_mhw_file)


# %% =========================== Computing the mean values ===========================
# Mean value or 1980-2018
growth_no_mhw_mean = growth_no_mhw_ds.mean(dim=('years','days'))
growth_mhw_mean = growth_mhw_combined.mean(dim=('years','days'))

# %% =========================== Plot Growth under MHWs ===========================
# --------- Plot setup ---------
plot='slides' #slides report

if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(4, 3, wspace=0.3, hspace=0.1)

    axs = []
    axs.append(fig.add_subplot(gs[0:2, 0], projection=ccrs.SouthPolarStereo())) #1°C
    axs.append(fig.add_subplot(gs[0:2, 1], projection=ccrs.SouthPolarStereo())) #2°C
    axs.append(fig.add_subplot(gs[2:4, 0], projection=ccrs.SouthPolarStereo())) #3°C
    axs.append(fig.add_subplot(gs[2:4, 1], projection=ccrs.SouthPolarStereo())) #4°C
    axs.append(fig.add_subplot(gs[1:3, 2], projection=ccrs.SouthPolarStereo())) # Non MHWs - Centered vertically

elif plot == 'slides':
    fig_width = 6.3228348611
    fig_height = fig_width

    fig = plt.figure(figsize=(fig_width * 5, fig_height))  # 5 columns wide
    gs = gridspec.GridSpec(1, 5, wspace=0.1, hspace=0.2)

    axs = []
    for j in range(5):
        ax = fig.add_subplot(gs[0, j], projection=ccrs.SouthPolarStereo())
        axs.append(ax)

# --------- Data ---------
plot_data = [(growth_mhw_mean.isel(xi_rho=slice(0, -1)).growth_1deg, r"MHWs $>$ 1$^\circ$C"),
             (growth_mhw_mean.isel(xi_rho=slice(0, -1)).growth_2deg, r"MHWs $>$ 2$^\circ$C"),
             (growth_mhw_mean.isel(xi_rho=slice(0, -1)).growth_3deg, r"MHWs $>$ 3$^\circ$C"),
             (growth_mhw_mean.isel(xi_rho=slice(0, -1)).growth_4deg, r"MHWs $>$ 4$^\circ$C"),
             (growth_no_mhw_mean.isel(xi_rho=slice(0, -1)).growth_noMHW, r"No MHWs")]

title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 20, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}

from matplotlib import colors
vmin, vmax = -0.2, 0.2
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

# --------- Plot ---------
for i, (data, title) in enumerate(plot_data):
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
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot data
    im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                       transform=ccrs.PlateCarree(), cmap='PuOr_r', norm=norm,
                       shading='auto', zorder=1, rasterized=True)

    ax.set_title(title, **title_kwargs)

# --------- Colorbar ---------
tick_positions = [-0.2, -0.1, 0.0, 0.1, 0.2]
if plot == 'report':
    orientation = 'horizontal'
    cbar_kwargs = {
        'fraction': 0.02,  # thinner
        'pad': 0.06,       # slightly closer to the plot
        'aspect': 50       # makes it longer (default is often 20)
    }
    # cbar_kwargs = {'fraction': 0.025, 'pad': 0.02}
else:
    orientation = 'horizontal'
    cbar_kwargs = {
        'fraction': 0.05,  # thinner
        'pad': 0.07,       # slightly closer to the plot
        'aspect': 40       # makes it longer (default is often 20)
    }

cbar = fig.colorbar(
    im, ax=axs, orientation=orientation,
    ticks=tick_positions, extend='both',
    **cbar_kwargs
)

cbar.set_label("Growth [mm/d]", **label_kwargs)
cbar.ax.tick_params(**tick_kwargs)

# --------- Axis labels and title ---------
if plot == 'report':
    suptitle_y = 0.88
    fig.suptitle(f'Average krill growth under different MHW intensities', y=suptitle_y, **suptitle_kwargs)
    fig.text(0.5, suptitle_y - 0.05, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', **title_kwargs, style='italic')
else:
    suptitle_y = 1.05
    fig.suptitle(f'Average krill growth under different MHW intensities', y=suptitle_y, **suptitle_kwargs)
    fig.text(0.5, suptitle_y - 0.08, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', fontsize=17, style='italic')


# --------- Output handling ---------
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/mhws_VS_non_mhws/')
    outfile = f"growth_mhw_VS_nomhw_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/mhws_VS_non_mhws/growth_mhw_VS_nomhw_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()
    
# %% =========================== Quantifying the Avg Chla under MHWs ===========================
# Load data 
chla_filtered_seasons_file = os.path.join(path_growth_inputs, "chla_surf_allyears_detrended_seasonal.nc")
chla_filtered_seasons=xr.open_dataset(chla_filtered_seasons_file) # shape (39, 181, 231, 1442)
mhw_det = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))) #shape: (39, 181, 231, 1442)

# -- Write or load data
chla_mhws_file = os.path.join(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_mhws.nc'))

if not os.path.exists(chla_mhws_file):
    # =========== Chla under MHWs ===========

    variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
    chla_var_names = ['chla_1deg', 'chla_2deg', 'chla_3deg', 'chla_4deg']
    chla_mhw_dict = {}

    for mhw_var, chla_name in zip(variables, chla_var_names):
        print(f'------------{mhw_var}------------')
        # mhw_var='det_1deg'

        # Masks
        duration_mask = mhw_det['duration'] >= 30 # DataArray bool
        det_mask = mhw_det[mhw_var] == 1 # DataArray bool
        mhw_mask = duration_mask & det_mask # shape (39, 181, 231, 1442)

        # Clean and align mask
        mhw_mask_clean = mhw_mask.drop_vars('days').rename({'days_of_yr': 'days'})
        mhw_mask_clean = mhw_mask_clean.assign_coords(years=mhw_mask_clean.years + 1980)

        # Mask CHLA with MHWs
        chla_mhw = chla_filtered_seasons.where(mhw_mask_clean) 

        # Mean across time
        chla_mhw_mean = chla_mhw.mean(dim=('years', 'days'), skipna=True)
        chla_mhw_dict[chla_name] = chla_mhw_mean

    # =========== Chla under NON MHWs ===========
    # Masks
    non_mhw_mask = mhw_det['duration'] == 0
        
    # Clean and align mask
    non_mhw_mask_clean = non_mhw_mask.drop_vars('days').rename({'days_of_yr': 'days'})
    non_mhw_mask_clean = non_mhw_mask_clean.assign_coords(years=non_mhw_mask_clean.years + 1980)

    # Mask CHLA with MHWs
    chla_non_mhw = chla_filtered_seasons.where(non_mhw_mask_clean) 

    # Mean across time
    chla_non_mhw_mean  = chla_non_mhw.mean(dim=('years', 'days'), skipna=True)


    # =========== Combine into DataSet
    chla_mhws_ds = xr.Dataset({
        'chla_mhw_1deg': chla_mhw_dict['chla_1deg'].chla,
        'chla_mhw_2deg': chla_mhw_dict['chla_2deg'].chla,
        'chla_mhw_3deg': chla_mhw_dict['chla_3deg'].chla,
        'chla_mhw_4deg': chla_mhw_dict['chla_4deg'].chla,
        'chla_non_mhw': chla_non_mhw_mean.chla,
        })
    chla_mhws_ds.attrs['title'] = 'Mean surface CHLA (detrented) during MHWs and non-MHWs'
    chla_mhws_ds.attrs['description'] = 'Computed over growth season (1 Nov–30 Apr), 1980–2018'
    chla_mhws_ds.attrs['units'] = 'mg/m3'
    chla_mhws_ds.to_netcdf(chla_mhws_file)

else: 
    # Load data
    chla_mhws_ds = xr.open_dataset(chla_mhws_file)


# %% =========================== Plot ===========================
plot = 'report'  # slides report

# --------- Figure layout ---------
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(4, 3, wspace=0.3, hspace=0.1)

    axs = []
    axs.append(fig.add_subplot(gs[0:2, 0], projection=ccrs.SouthPolarStereo()))  # 1°C
    axs.append(fig.add_subplot(gs[0:2, 1], projection=ccrs.SouthPolarStereo()))  # 2°C
    axs.append(fig.add_subplot(gs[2:4, 0], projection=ccrs.SouthPolarStereo()))  # 3°C
    axs.append(fig.add_subplot(gs[2:4, 1], projection=ccrs.SouthPolarStereo()))  # 4°C
    axs.append(fig.add_subplot(gs[1:3, 2], projection=ccrs.SouthPolarStereo()))  # Non MHWs

elif plot == 'slides':
    fig_width = 6.3228348611
    fig_height = fig_width
    fig = plt.figure(figsize=(fig_width * 5, fig_height))  # 5 columns wide
    gs = gridspec.GridSpec(1, 5, wspace=0.1, hspace=0.2)
    axs = [fig.add_subplot(gs[0, j], projection=ccrs.SouthPolarStereo()) for j in range(5)]

title_kwargs = {'fontsize': 16} if plot == 'slides' else {}
label_kwargs = {'fontsize': 15} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
# legend_kwargs = {'fontsize': 12} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 20, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}

chla_plot_data = [(chla_mhws_ds['chla_mhw_1deg'], r"MHWs $>$ 1$^\circ$C"),
                  (chla_mhws_ds['chla_mhw_2deg'], r"MHWs $>$ 2$^\circ$C"),
                  (chla_mhws_ds['chla_mhw_3deg'], r"MHWs $>$ 3$^\circ$C"),
                  (chla_mhws_ds['chla_mhw_4deg'], r"MHWs $>$ 4$^\circ$C"),
                  (chla_mhws_ds['chla_non_mhw'], r"No MHWs"),]

from matplotlib import colors
norm = colors.Normalize(vmin=0, vmax=1)
custom_colors = ["#0E1B11", "#4A8956", "#73A942", "#E7D20D", "#FBB02D"]

# --------- Plot ---------
for i, (data, title) in enumerate(chla_plot_data):
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
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # CHLA data
    im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                       transform=ccrs.PlateCarree(), cmap=colors.LinearSegmentedColormap.from_list('custom_cmap', custom_colors), norm=norm, 
                       shading='auto', zorder=1, rasterized=True)
    
    ax.set_title(title, **title_kwargs)

# --------- Colorbar ---------
tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
if plot == 'report':
    cbar_kwargs = {'fraction': 0.02, 'pad': 0.06, 'aspect': 50}
else:
    cbar_kwargs = {'fraction': 0.05, 'pad': 0.07, 'aspect': 40}

cbar = fig.colorbar(im, ax=axs, orientation='horizontal', ticks=tick_positions, extend='max', **cbar_kwargs)
cbar.set_label("Chla [mg m$^{-3}$]", **label_kwargs)
cbar.ax.tick_params(**tick_kwargs)

# --------- Title and subtitle ---------
if plot == 'report':
    suptitle_y = 0.88
    fig.suptitle(f'Average surface Chla under different MHW intensities', y=suptitle_y, **suptitle_kwargs)
    fig.text(0.5, suptitle_y - 0.05, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', **title_kwargs, style='italic')
else:
    suptitle_y = 1.05
    fig.suptitle(f'Average surface Chla under different MHW intensities', y=suptitle_y, **suptitle_kwargs)
    fig.text(0.5, suptitle_y - 0.08, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', fontsize=17, style='italic')


# --------- Output handling ---------
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/mhws_VS_non_mhws/')
    outfile = f"chla_mhw_VS_nomhw_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/mhws_VS_non_mhws/chla_mhw_VS_nomhw_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %%

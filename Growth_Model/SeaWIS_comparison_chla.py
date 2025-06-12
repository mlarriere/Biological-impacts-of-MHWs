#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 21 May 16:55:24 2025

Chla ROMS comparison with satellite data 

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
import cmocean
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from matplotlib.cm import ScalarMappable

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
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'
path_det_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer'
path_combined_thesh= '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/combined_thresholds'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
path_growth = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model'
path_growth_inputs_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs/austral_summer'
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
doy_list = list(range(305, 365)) + list(range(0, 121))
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

# %% ===== Observation =====
# Obs from CMEMS (copernicus - included different obs including Sea WIS)
cha_obs= xr.open_dataset(os.path.join(path_up_obs, 'cmems_chl_0.25deg_monthly_1998_to_2022.nc')).isel(depth=0).chl #time dim corresponds to month from 1998 to 2022

# === Select extent - south of 60°S
south_mask = cha_obs['latitude'] <= -60
chla_obs_60S_south = cha_obs.where(south_mask, drop=True)  #shape (time: 300, latitude: 89, longitude: 1440) 

# === Reshaping
n_time = chla_obs_60S_south.sizes['time']
n_years = n_time // 12  # Should be 25
n_months = 12
chl_reshaped = (chla_obs_60S_south.data.reshape((n_years, n_months, chla_obs_60S_south.sizes['latitude'], chla_obs_60S_south.sizes['longitude'])))
chla_obs_reshaped = xr.DataArray(chl_reshaped,
                              dims=('year', 'month', 'latitude', 'longitude'),
                              coords={
                                  'year': np.arange(1998, 1998 + n_years),
                                  'month': np.arange(1, 13),
                                  'latitude': chla_obs_60S_south['latitude'],
                                  'longitude': chla_obs_60S_south['longitude'],
                                  },
                              name='chl') #shape (25, 12, 89, 1440)

# === Time (1998 to 2019)
cha_obs_1998_2019 = chla_obs_reshaped.sel(year=slice(1998, 2019))

# %% ===== ROMS =====
# Weighted averaged chla of the first 100m - 60S - from 1980 to 2019 - daily
chla_surf= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears.nc')).raw_chla #shape: (years: 40, days: 365, eta_rho: 231, xi_rho: 1442)
chla_surf = chla_surf.rename({'year': 'years'})

# === To monthly dataset
month_day_bounds = [
    (0, 31),    # Jan: days 0-30
    (31, 59),   # Feb: days 31-58
    (59, 90),   # Mar
    (90, 120),  # Apr
    (120, 151), # May
    (151, 181), # Jun
    (181, 212), # Jul
    (212, 243), # Aug
    (243, 273), # Sep
    (273, 304), # Oct
    (304, 334), # Nov
    (334, 365)  # Dec: ends at 364
]

monthly_means = []
for start, end in month_day_bounds:
    month_mean = chla_surf.isel(days=slice(start, end)).mean(dim='days')
    monthly_means.append(month_mean)

cha_ROMS_monthly = xr.concat(monthly_means, dim='month')
cha_ROMS_monthly = cha_ROMS_monthly.assign_coords(month=('month', np.arange(1, 13)))

# === Time (1998 to 2019)
cha_ROMS_1998_2019 = cha_ROMS_monthly.isel(years=slice(18, 40)) 


# %% Select time and average
# Select only from november to april (included)
months_nov_apr = [10, 11, 0, 1, 2, 3]  # Corresponding to Nov-Apr

# Select years and avg
cha_ROMS_1998_2009_avg = cha_ROMS_1998_2019.isel(years=slice(0, 12), month=months_nov_apr).mean(dim=('month', 'years'))
cha_ROMS_2010_2019_avg = cha_ROMS_1998_2019.isel(years=slice(12, 23), month=months_nov_apr).mean(dim=('month', 'years'))

cha_obs_1998_2009_avg = cha_obs_1998_2019.isel(year=slice(0, 12), month=months_nov_apr).mean(dim=('month', 'year'))
cha_obs_2010_2019_avg = cha_obs_1998_2019.isel(year=slice(12, 23), month=months_nov_apr).mean(dim=('month', 'year'))

# %% Regridding 
import xesmf as xe

# Create the grid dictionaries
obs_grid = {
    "lon": cha_obs_1998_2009_avg.longitude,
    "lat": cha_obs_1998_2009_avg.latitude,
}
roms_grid = {
    "lon": cha_ROMS_1998_2009_avg.lon_rho,
    "lat": cha_ROMS_1998_2009_avg.lat_rho,
}

# Set up regridder
regridder = xe.Regridder(cha_obs_1998_2009_avg, cha_ROMS_1998_2009_avg, method="bilinear", periodic=True)

# Apply regridding to both observation averages
cha_obs_1998_2009_avg_rg = regridder(cha_obs_1998_2009_avg)
cha_obs_2010_2019_avg_rg = regridder(cha_obs_2010_2019_avg)

diff_1998_2009 = cha_ROMS_1998_2009_avg - cha_obs_1998_2009_avg_rg
diff_2010_2019 = cha_ROMS_2010_2019_avg - cha_obs_2010_2019_avg_rg

# %% === Comparison map year per year 

# === Create figure ===
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width * 2 / 3  # adjust for 2 rows
fig = plt.figure(figsize = (fig_width, fig_height)) #(20, 12)

# 2 rows, 3 columns: 
gs = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.25)

axes = [fig.add_subplot(gs[i, j], projection=ccrs.SouthPolarStereo())
        for i in range(2) for j in range(3)]

# === Circular boundary for all axes ===
theta = np.linspace(0, 2 * np.pi, 100)
circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)

# === Shared plot settings ===
def format_ax(ax):
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
    ax.coastlines(color='black', linewidth=1)
    ax.set_facecolor('#F6F6F3')
    # for lon in [-85, 150, 20]:
    #     ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#495057',
    #             linestyle='--', linewidth=1)
    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

# === Plot data ===
vmin, vmax = 0, 2.5
cmap = cmocean.cm.algae

# Row 1: 1998–2008
# Column 1: CMEMS 1998–2008
im0 = cha_obs_1998_2009_avg.plot(ax=axes[0], transform=ccrs.PlateCarree(),
                                 cmap=cmap, vmin=vmin, vmax=vmax, 
                                 add_colorbar=False,
                                 rasterized=True)
format_ax(axes[0])

# Column 2: ROMS 1998–2008
im1 = cha_ROMS_1998_2009_avg.plot.pcolormesh(ax=axes[1], transform=ccrs.PlateCarree(),
                                             x="lon_rho", y="lat_rho",
                                             cmap=cmap, vmin=vmin, vmax=vmax,
                                             add_colorbar=False,
                                             rasterized=True)
format_ax(axes[1])

# Column 3: Difference 1998–2008 (CMEMS - ROMS)
diff_vmin, diff_vmax = -1, 1
diff_cmap = "RdBu_r"
im2 = diff_1998_2009.plot.pcolormesh(ax=axes[2], transform=ccrs.PlateCarree(),
                                    x="lon_rho", y="lat_rho",
                                    cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax,
                                    add_colorbar=False, rasterized=True)
format_ax(axes[2])

# Row 2: 2009–2019
# Column 1: CMEMS 2009–2019
im3 = cha_obs_2010_2019_avg.plot(ax=axes[3], transform=ccrs.PlateCarree(),
                                 cmap=cmap, vmin=vmin, vmax=vmax, 
                                 add_colorbar=False,
                                 rasterized=True)
format_ax(axes[3])

# Column 2: ROMS 2009–2019
im4 = cha_ROMS_2010_2019_avg.plot.pcolormesh(ax=axes[4], transform=ccrs.PlateCarree(),
                                             x="lon_rho", y="lat_rho",
                                             cmap=cmap, vmin=vmin, vmax=vmax,
                                             add_colorbar=False,
                                             rasterized=True)
format_ax(axes[4])

# Column 3: Difference 2009–2019 (CMEMS - ROMS)
im5 = diff_2010_2019.plot.pcolormesh(ax=axes[5], transform=ccrs.PlateCarree(),
                                    x="lon_rho", y="lat_rho",
                                    cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax,
                                    add_colorbar=False, rasterized=True)
format_ax(axes[5])

# === Horizontal colorbar for chla ===
cax_avg = fig.add_axes([0.15, 0.08, 0.45, 0.02])  # [left, bottom, width, height]
norm_avg = mcolors.Normalize(vmin=vmin, vmax=vmax)
sm_avg = ScalarMappable(cmap=cmap, norm=norm_avg)
sm_avg.set_array([])
cbar_avg = fig.colorbar(sm_avg, cax=cax_avg, orientation='horizontal', extend='max')
# cbar_avg.ax.tick_params(labelsize=9)
cbar_avg.set_label('Chl-a (mg/m³)')

# === Vertical colorbar for differences ===
cax_diff = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
norm_diff = mcolors.TwoSlopeNorm(vmin=diff_vmin, vcenter=0, vmax=diff_vmax)
sm_diff = ScalarMappable(cmap=diff_cmap, norm=norm_diff)
sm_diff.set_array([])
cbar_diff = fig.colorbar(sm_diff, cax=cax_diff, extend='both')
# cbar_diff.ax.tick_params(labelsize=9)
cbar_diff.set_label('$\Delta$ Chl-a (mg/m³)')


# === Add column titles ===
fig.text(0.25, 0.9, 'CMEMS', ha='center')
fig.text(0.5, 0.9, 'ROMS', ha='center')
fig.text(0.8, 0.9, 'Difference$_{(CMEMS - ROMS)}$', ha='center')

# === Add row titles ===
fig.text(0.09, 0.75, '1998–2008 avg', va='center', rotation='vertical')
fig.text(0.09, 0.3, '2009–2019 avg', va='center', rotation='vertical')

fig.suptitle('Comparison CMEMS and ROMS Chlorophyll-a Concentrations', y=0.99, x=0.57)

# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/inputs/chla_CMEMS_ROMS_comparison.pdf'), dpi =150, format='pdf', bbox_inches='tight')


# %% --------------- Difference between decades ---------------
# Compute differences
cmems_decadal_diff = cha_obs_2010_2019_avg_rg - cha_obs_1998_2009_avg_rg
roms_decadal_diff = cha_ROMS_2010_2019_avg - cha_ROMS_1998_2009_avg

# === Plot ===
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width * 2 / 3  # adjust for 2 rows
fig = plt.figure(figsize = (fig_width, fig_height)) #(20, 12)

# 1 rows, 2 columns: 
gs = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.25)
axes = [fig.add_subplot(gs[i, j], projection=ccrs.SouthPolarStereo())
        for i in range(1) for j in range(2)]

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)

# Colormap and normalization
vmin, vmax = -1, 1
cmap = 'RdBu_r'
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# -- CMEMS subplot
format_ax(axes[0])
cmems_decadal_diff.plot(
    ax=axes[0], x='lon_rho', y='lat_rho',
    transform=ccrs.PlateCarree(),
    cmap=cmap, norm=norm, add_colorbar=False,
    rasterized=True
)
axes[0].set_title('CMEMS decadal change\n 2010–2019 minus 1998–2009')

# -- ROMS subplot
format_ax(axes[1])
roms_decadal_diff.plot(
    ax=axes[1], x='lon_rho', y='lat_rho',
    transform=ccrs.PlateCarree(),
    cmap=cmap, norm=norm, add_colorbar=False,
    rasterized=True
)
axes[1].set_title('ROMS decadal change\n 2010–2019 minus 1998–2009')

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='both')
cbar.set_label(r'$\Delta$Chl-a (mg/m³)')

plt.tight_layout(rect=[0, 0.12, 1, 1])  # leave space for colorbar
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/inputs/chla_CMEMS_ROMS_decadal_comparison.pdf'), dpi=150, format='pdf', bbox_inches='tight')

# %%

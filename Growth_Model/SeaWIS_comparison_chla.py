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
# cha_obs= xr.open_dataset(os.path.join(path_up_obs, 'cmems_chl_0.25deg_monthly_1998_to_2022.nc')).isel(depth=0).chl #time dim corresponds to month from 1998 to 2022

# # === Select extent - south of 60°S
# south_mask = cha_obs['latitude'] <= -60
# chla_obs_60S_south = cha_obs.where(south_mask, drop=True)  #shape (time: 300, latitude: 89, longitude: 1440) 

# # === Reshaping
# n_time = chla_obs_60S_south.sizes['time']
# n_years = n_time // 12  # Should be 25
# n_months = 12
# chl_reshaped = (chla_obs_60S_south.data.reshape((n_years, n_months, chla_obs_60S_south.sizes['latitude'], chla_obs_60S_south.sizes['longitude'])))
# chla_obs_reshaped = xr.DataArray(chl_reshaped,
#                               dims=('year', 'month', 'latitude', 'longitude'),
#                               coords={
#                                   'year': np.arange(1998, 1998 + n_years),
#                                   'month': np.arange(1, 13),
#                                   'latitude': chla_obs_60S_south['latitude'],
#                                   'longitude': chla_obs_60S_south['longitude'],
#                                   },
#                               name='chl') #shape (25, 12, 89, 1440)

# # === Time (1998 to 2019)
# cha_obs_1998_2019 = chla_obs_reshaped.sel(year=slice(1998, 2019))


# da = cha_obs_1998_2019.sel(year=2010, month=12)
# fig = plt.figure(figsize=(8, 6))
# ax = plt.axes(projection=ccrs.SouthPolarStereo())
# ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND, facecolor='lightgrey')
# ax.set_facecolor('lightblue')
# p = ax.pcolormesh(
#     cha_obs_1998_2019.longitude, cha_obs_1998_2019.latitude, da,
#     transform=ccrs.PlateCarree(), shading='auto', cmap='viridis'
# )
# cbar = plt.colorbar(p, orientation='horizontal', pad=0.05, aspect=30)
# cbar.set_label('Chlorophyll-a (mg/m³)')
# ax.set_title('Surface Chlorophyll-a\n Month 12 – Year 2010', fontsize=12)
# plt.tight_layout()
# plt.show()

# Obs from CMEMS (copernicus - included different obs including Sea WIS)
cha_obs = xr.open_dataset(os.path.join(path_up_obs, 'cmems_chl_SO_d025_monthly_1998_to_2022.nc')).isel(depth=0).chl # Load monthly chlorophyll dataset

# === Reshaping
# Reshape manually into (year, month)
assert cha_obs.sizes["time"] == 300
cha_obs_reshaped = cha_obs.data.reshape(25, 12, cha_obs.shape[1], cha_obs.shape[2])

# Into DataArray
cha_obs_reshaped = xr.DataArray(
    cha_obs_reshaped,
    coords=dict(
        year=np.arange(1998, 2023),
        month=np.arange(1, 13),
        lon_rho=(('eta_rho', 'xi_rho'), ds_roms['lon_rho'].values),
        lat_rho=(('eta_rho', 'xi_rho'), ds_roms['lat_rho'].values),
    ),
    dims=["year", "month", "eta_rho", "xi_rho"],
    name="chl"
)

# === Select extent - south of 60°S
south_mask = cha_obs_reshaped['lat_rho'] <= -60
chla_obs_60S_south = cha_obs_reshaped.where(south_mask, drop=True)  #shape (year: 25, month:12, eta_rho: 231, xi_rho: 1442) 


# === Time (1998 to 2019)
cha_obs_1998_2019 = chla_obs_60S_south.sel(year=slice(1998, 2019))

da = cha_obs_1998_2019.sel(year=2010, month=12)
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.set_facecolor('lightblue')
p = ax.pcolormesh(
    cha_obs_1998_2019.lon_rho, cha_obs_1998_2019.lat_rho, da,
    transform=ccrs.PlateCarree(), shading='auto', cmap='viridis'
)
cbar = plt.colorbar(p, orientation='horizontal', pad=0.05, aspect=30)
cbar.set_label('Chlorophyll-a (mg/m³)')
ax.set_title('Surface Chlorophyll-a (CMEMS) \n Month 12 – Year 2010', fontsize=12)
plt.tight_layout()
plt.show()


# %% ===== ROMS =====
# Chla at 5m - south of 60S - from 1980 to 2019 - daily
chla_surf= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended.nc')).raw_chla #shape: (years: 40, days: 365, eta_rho: 231, xi_rho: 1442)
# cha_ROMS_monthly= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_monthly_detrended.nc')).raw_chla #shape: (years: 40, days: 365, eta_rho: 231, xi_rho: 1442)

# Check
da = chla_surf.sel(year=2010, day=350)
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.set_facecolor('lightblue')
p = ax.pcolormesh(
    chla_surf.lon_rho, chla_surf.lat_rho, da,
    transform=ccrs.PlateCarree(), shading='auto', cmap='viridis'
)
cbar = plt.colorbar(p, orientation='horizontal', pad=0.05, aspect=30)
cbar.set_label('Chlorophyll-a (mg/m³)')
ax.set_title('Surface Chlorophyll-a\nDay 350 – Year 2010', fontsize=12)
plt.tight_layout()
plt.show()

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
    # month_mean = chla_surf.isel(days=slice(start, end)).mean(dim='days')
    month_mean = chla_surf.isel(day=slice(start, end)).mean(dim='day', skipna=False)
    monthly_means.append(month_mean)

cha_ROMS_monthly = xr.concat(monthly_means, dim='month')
cha_ROMS_monthly = cha_ROMS_monthly.assign_coords(month=('month', np.arange(1, 13)))

# === Time (1998 to 2019)
cha_ROMS_1998_2019 = cha_ROMS_monthly.isel(year=slice(18, 40)) 

# Check
m=12
da = cha_ROMS_1998_2019.sel(year=2010, month=m)
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.set_facecolor('lightblue')
p = ax.pcolormesh(
    cha_ROMS_monthly.lon_rho, cha_ROMS_monthly.lat_rho, da,
    transform=ccrs.PlateCarree(), shading='auto', cmap='viridis'
)
ax.set_title(f'Surface Chlorophyll-a month {m} – Year 2010', fontsize=12)
plt.tight_layout()
plt.show()


# %% Select time and average
# Select only from november to april (included)
months_nov_apr = [11, 12, 1, 2, 3, 4]  # Corresponding to Nov-Apr

# Select years and avg 
# ROMS: lon_rho range from 24.125 to 383.875
# Obs: longitude range from -179.875 to 179.875
cha_ROMS_1998_2009_avg = cha_ROMS_1998_2019.isel(year=slice(0, 12)).sel(month=months_nov_apr).mean(dim=('month', 'year'), skipna=True)
cha_ROMS_2010_2019_avg = cha_ROMS_1998_2019.isel(year=slice(12, 23)).sel(month=months_nov_apr).mean(dim=('month', 'year'), skipna=True)

cha_obs_1998_2009_avg = cha_obs_1998_2019.isel(year=slice(0, 12)).sel(month=months_nov_apr).mean(dim=('month', 'year'), skipna=True)
cha_obs_2010_2019_avg = cha_obs_1998_2019.isel(year=slice(12, 23)).sel(month=months_nov_apr).mean(dim=('month', 'year'), skipna=True)


# Shifting longitude to 0-360
cha_ROMS_1998_2009_avg = cha_ROMS_1998_2009_avg.assign_coords(lon_rho=(cha_ROMS_1998_2009_avg.lon_rho % 360))
cha_ROMS_2010_2019_avg = cha_ROMS_2010_2019_avg.assign_coords(lon_rho=(cha_ROMS_2010_2019_avg.lon_rho % 360))
cha_obs_1998_2009_avg = cha_obs_1998_2009_avg.assign_coords(lon_rho=(cha_obs_1998_2009_avg.lon_rho % 360))
cha_obs_2010_2019_avg = cha_obs_2010_2019_avg.assign_coords(lon_rho=(cha_obs_2010_2019_avg.lon_rho % 360))

# # Ensure latitude and longitude are sorted
# cha_obs_1998_2009_avg = cha_obs_1998_2009_avg.sortby(['latitude', 'longitude'])
# cha_obs_2010_2019_avg = cha_obs_2010_2019_avg.sortby(['latitude', 'longitude'])

# Compute differences (ROMS - OBS)
diff_1998_2009 = cha_ROMS_1998_2009_avg - cha_obs_1998_2009_avg
diff_2010_2019 = cha_ROMS_2010_2019_avg - cha_obs_2010_2019_avg

# # %% Regridding 
# import xesmf as xe
# from scipy.spatial import cKDTree

# def regrid_roms_to_obs_kdtree(roms_da, obs_da, radius=0.5):
#     """
#     Regrid ROMS data onto obs grid using KDTree-based local averaging.

#     Parameters:
#         roms_da (xr.DataArray): ROMS data with 2D lat_rho, lon_rho coordinates.
#         obs_da (xr.DataArray): Observational data with 1D latitude, longitude coords.
#         radius (float): Radius in degrees for neighborhood averaging.

#     Returns:
#         xr.DataArray: ROMS values regridded to obs grid.
#     """

#     # Flatten ROMS lat/lon and values
#     roms_lons = roms_da.lon_rho.values.flatten()
#     roms_lats = roms_da.lat_rho.values.flatten()
#     roms_vals = roms_da.values.flatten()

#     # Build KDTree for ROMS points
#     roms_tree = cKDTree(np.column_stack((roms_lons, roms_lats)))

#     # Get obs 1D coords
#     lon_obs_1d = obs_da.longitude.values
#     lat_obs_1d = obs_da.latitude.values

#     # Create 2D meshgrid and flatten
#     lon_obs_2d, lat_obs_2d = np.meshgrid(lon_obs_1d, lat_obs_1d)
#     obs_points = np.column_stack((lon_obs_2d.ravel(), lat_obs_2d.ravel()))

#     # Query neighbors in ROMS grid
#     indices_list = roms_tree.query_ball_point(obs_points, r=radius)

#     # Compute mean for each obs point
#     mean_roms_vals = np.full(obs_points.shape[0], np.nan, dtype=np.float32)
#     for i, inds in enumerate(indices_list):
#         if inds:
#             mean_roms_vals[i] = np.nanmean(roms_vals[inds])

#     # Reshape back to obs grid shape
#     mean_roms_2d = mean_roms_vals.reshape(len(lat_obs_1d), len(lon_obs_1d))

#     # Wrap in DataArray
#     return xr.DataArray(
#         mean_roms_2d,
#         coords={'latitude': lat_obs_1d, 'longitude': lon_obs_1d},
#         dims=['latitude', 'longitude'],
#         name=f"regridded_roms_{roms_da.name}"
#     )

# # Regrid ROMS to obs grid (via KDTree)
# roms_on_obs_1998_2009 = regrid_roms_to_obs_kdtree(cha_ROMS_1998_2009_avg, cha_obs_1998_2009_avg)
# roms_on_obs_2010_2019 = regrid_roms_to_obs_kdtree(cha_ROMS_2010_2019_avg, cha_obs_2010_2019_avg)

# # Compute differences (ROMS - OBS)
# diff_1998_2009 = roms_on_obs_1998_2009 - cha_obs_1998_2009_avg
# diff_2010_2019 = roms_on_obs_2010_2019 - cha_obs_2010_2019_avg

# %% === Comparison map year per year 
# === Create figure ===
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width * 2 / 3  # adjust for 2 rows
fig = plt.figure(figsize = (fig_width*2, fig_width)) #(20, 12)

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
    for lon in [-90, 120, 0]:
        ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#495057',
                linestyle='--', linewidth=1)
    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

# === Plot data ===
vmin, vmax = 0, 1
from matplotlib.colors import LinearSegmentedColormap
colors = ["#216869", "#73A942", "#FBB02D"]  # Blue, Green, Yellow
cmap = LinearSegmentedColormap.from_list("blue_green_yellow", colors, N=256)
# cmap = cmocean.cm.algae

# Row 1: 1998–2008
# Column 1: CMEMS 1998–2008
im0 = cha_obs_1998_2009_avg.plot.pcolormesh(ax=axes[0], transform=ccrs.PlateCarree(),
                                 x="lon_rho", y="lat_rho",
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
im3 = cha_obs_2010_2019_avg.plot.pcolormesh(ax=axes[3], transform=ccrs.PlateCarree(),
                                 x="lon_rho", y="lat_rho",
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
cax_avg = fig.add_axes([0.15, 0.03, 0.45, 0.02])  # [left, bottom, width, height]
norm_avg = mcolors.Normalize(vmin=vmin, vmax=vmax)
sm_avg = ScalarMappable(cmap=cmap, norm=norm_avg)
sm_avg.set_array([])
cbar_avg = fig.colorbar(sm_avg, cax=cax_avg, orientation='horizontal', extend='max')
# cbar_avg.ax.tick_params(labelsize=9)
cbar_avg.set_label('Chl-a (mg/m³)', fontsize=12)

# === Vertical colorbar for differences ===
cax_diff = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
norm_diff = mcolors.TwoSlopeNorm(vmin=diff_vmin, vcenter=0, vmax=diff_vmax)
sm_diff = ScalarMappable(cmap=diff_cmap, norm=norm_diff)
sm_diff.set_array([])
cbar_diff = fig.colorbar(sm_diff, cax=cax_diff, extend='both')
# cbar_diff.ax.tick_params(labelsize=9)
cbar_diff.set_label('$\Delta$ Chl-a (mg/m³)', fontsize=12)


# === Add column titles ===
fig.text(0.25, 0.9, 'CMEMS', ha='center', fontsize=13)
fig.text(0.5, 0.9, 'ROMS', ha='center', fontsize=13)
fig.text(0.8, 0.9, 'Difference$_{(CMEMS - ROMS)}$', ha='center', fontsize=13)

# === Add row titles ===
fig.text(0.09, 0.75, '1998–2008 avg', va='center', rotation='vertical', fontsize=13)
fig.text(0.09, 0.3, '2009–2019 avg', va='center', rotation='vertical', fontsize=13)

fig.suptitle('Comparison CMEMS and ROMS Chlorophyll-a Concentrations', y=0.99, x=0.57, fontsize=18)

plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/inputs/chla_CMEMS_ROMS_comparison.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %%
import cmocean
import matplotlib.colors as mcolors
def plot_comparison(ds, cmap_var=None, ticks=None, cbar_label=''):
    # Prepare figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    
    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    # Time periods
    data_1998_2009 = ds.sel(year=slice(1998, 2009)).mean(dim=('year', 'month'))
    data_2010_2019 = ds.sel(year=slice(2010, 2019)).mean(dim=('year', 'month'))
    data_diff = data_2010_2019 - data_1998_2009

    # Set normalization
    norm_main = mcolors.Normalize(vmin=0, vmax=1)
    extend_var ='max'
    
    # Difference normalization (centered at zero for difference)
    # abs_diff_max = np.max(np.abs(data_diff))
    norm_diff = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap_diff = plt.cm.RdBu_r

    # Titles and datasets
    plot_data = [
        (data_1998_2009, f"Avg Chla (1998–2009)", norm_main, cmap_var),
        (data_2010_2019, f"Avg Chla (2010–2019)", norm_main, cmap_var),
        (data_diff, "Difference${_{({warming}-{climatology})}}$", norm_diff, cmap_diff),
    ]

    # Plotting data
    scs = []  # List to hold the scatter plot objects for each subplot
    for ax, (data, title, norm, cmap_used) in zip(axs, plot_data):
        sc = data.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            x="lon_rho", y="lat_rho",
            add_colorbar=False, cmap=cmap_used, norm=norm, zorder=1, rasterized=True)
        scs.append(sc)  # Store the plot object
        
        ax.set_title(title, fontsize=16)
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Draw the land feature after the pcolormesh
        ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
        ax.coastlines(color='black', linewidth=1)
        ax.set_facecolor('#F6F6F3')
        
        # Sector boundaries
        for lon in [-90, 120, 0]:
            ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#495057',
                    linestyle='--', linewidth=1)

        # Gridlines
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.3)

    # Colorbar for the first and second subplots
    ticks = ticks
    pos1 = axs[1].get_position()
    cbar_ax1 = fig.add_axes([pos1.x1 + 0.01, pos1.y0, 0.01, pos1.height])
    cbar1 = fig.colorbar(scs[0], cax=cbar_ax1, cmap=cmap_var, ticks=ticks, extend=extend_var)
    cbar1.set_label(cbar_label, fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    # Second colorbar (for the difference)
    pos2 = axs[2].get_position()
    cbar_ax2 = fig.add_axes([pos2.x1 + 0.045, pos2.y0, 0.01, pos2.height])
    cbar2 = fig.colorbar(scs[2], cax=cbar_ax2, cmap=cmap_diff, extend='both')
    cbar2.set_label("Difference", fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    plt.show()

ds = cha_obs_1998_2019.isel(xi_rho=slice(0, -1))    
vmin, vmax = 0, 1
colors = ["#0E1B11", "#4A8956", "#73A942", "#E7D20D", "#FBB02D"]
color_positions = np.linspace(vmin, vmax, len(colors))
normalized_positions = (color_positions - vmin) / (vmax - vmin)  # Normalize to [0, 1]
cmap_var = LinearSegmentedColormap.from_list("blue_green_yellow_buffered", list(zip(normalized_positions, colors)), N=256)
# cmap_var = cmocean.cm.algae 
label = 'Chla [mg/m³]'
ticks =  [0, 0.5, 1] # 2.5, 3, 3.5, 4, 4.5 ,5

plot_comparison(ds, cmap_var=cmap_var, ticks=ticks, cbar_label=label)


# %% --------------- Difference between decades ---------------
# Compute differences
# cmems_decadal_diff = cha_obs_2010_2019_avg_rg - cha_obs_1998_2009_avg_rg
# roms_decadal_diff = cha_ROMS_2010_2019_avg - cha_ROMS_1998_2009_avg

# # === Plot ===
# fig_width = 6.3228348611  # inches = \textwidth
# fig_height = fig_width * 2 / 3  # adjust for 2 rows
# fig = plt.figure(figsize = (fig_width, fig_height)) #(20, 12)

# # 1 rows, 2 columns: 
# gs = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.25)
# axes = [fig.add_subplot(gs[i, j], projection=ccrs.SouthPolarStereo())
#         for i in range(1) for j in range(2)]

# # Circular boundary
# theta = np.linspace(0, 2 * np.pi, 100)
# circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)

# # Colormap and normalization
# vmin, vmax = -1, 1
# cmap = 'RdBu_r'
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# # -- CMEMS subplot
# format_ax(axes[0])
# cmems_decadal_diff.plot(
#     ax=axes[0], x='lon_rho', y='lat_rho',
#     transform=ccrs.PlateCarree(),
#     cmap=cmap, norm=norm, add_colorbar=False,
#     rasterized=True
# )
# axes[0].set_title('CMEMS decadal change\n 2010–2019 minus 1998–2009')

# # -- ROMS subplot
# format_ax(axes[1])
# roms_decadal_diff.plot(
#     ax=axes[1], x='lon_rho', y='lat_rho',
#     transform=ccrs.PlateCarree(),
#     cmap=cmap, norm=norm, add_colorbar=False,
#     rasterized=True
# )
# axes[1].set_title('ROMS decadal change\n 2010–2019 minus 1998–2009')

# # Shared colorbar
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='both')
# cbar.set_label(r'$\Delta$Chl-a (mg/m³)')

# plt.tight_layout(rect=[0, 0.12, 1, 1])  # leave space for colorbar
# # plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/inputs/chla_CMEMS_ROMS_decadal_comparison.pdf'), dpi=150, format='pdf', bbox_inches='tight')

# # %%

"""
Created on Mon 19 Nov 16:22:30 2025

Look at the relationship between temperature and chla concentration

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob
import collections

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


from datetime import datetime, timedelta
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

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
roms_bathymetry = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc').h
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


# %% ====================== Load data ======================
# --- Drivers
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (39, 181, 231, 1442)
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 

# --- MHW events
mhw_duration_seasonal = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))) #shape (39, 181, 231, 1442)


# %% ====================== Select year with high number of MHWs ======================
years = mhw_duration_seasonal['years'].values

# Count events per year
event_count = []
for y in range(len(years)):
    yr = mhw_duration_seasonal.isel(years=y)
    duration_mask = yr['duration'] > 30
    intensity_mask = yr['det_4deg'].astype(bool)
    combined_mask = duration_mask & intensity_mask
    event_count.append(combined_mask.sum().item())

event_count = np.array(event_count)
best_year_idx = event_count.argmax()
best_year = years[best_year_idx]
print(f'Year with highest number of mhws of 4°C: {1980+best_year}')

year_idx = 9#best_year
mhw_yr_max = mhw_duration_seasonal.isel(years=year_idx) #shape (181, 231, 1442)

# === Identify grid cells under MHW for ≥ 30 consecutive days
# -- Duration threshold: event have to last at least 1 month
long_event_days = mhw_yr_max['duration'] > 30

# -- Intensity threshold
intense_days = mhw_yr_max['det_4deg'] == 1

# -- Filter duration & threshold
# combined_mask = duration_mask & intensity_mask
cells_with_long_4deg_mhw = (long_event_days & intense_days).any(dim="days")

# Find location where these long events are happening
eta_idx, xi_idx = np.where(cells_with_long_4deg_mhw)

# === 
mhw_daily = (mhw_yr_max['det_4deg'] == 1) & (mhw_yr_max['duration'] > 30)

# %% ====================== Temperature and chla at these locations ======================
k=1
i = eta_idx[k]   # row index
j = xi_idx[k]   # col index

mhw_duration_cell  = mhw_yr_max.duration.isel(eta_rho=i, xi_rho=j) #contain also smaller MHWs
temp_mhws_cell = temp_avg_100m_SO_allyrs.avg_temp.isel(years=year_idx, eta_rho=i, xi_rho=j)
chla_mhws_cell = chla_surf_SO_allyrs.chla.isel(years=year_idx, eta_rho=i, xi_rho=j)



# %% ====================== PLOT timeseries ======================
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15) #ticks every 15days
tick_labels = [date_dict.get(day, '') for day in tick_positions]


# === Identify contiguous MHW segments (for shading)
mhw_mask = mhw_duration_cell > 30
mhw_bool = mhw_mask.values
segments = []
start = None
for d in range(len(mhw_bool)):
    if mhw_bool[d] and start is None:
        start = d
    if (not mhw_bool[d] or d == len(mhw_bool)-1) and start is not None:
        end = d if mhw_bool[d] and d == len(mhw_bool)-1 else d-1
        segments.append((start, end))
        start = None

# === Plot data
fig = plt.figure(figsize=(20,5))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

# Temperature
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(days_xaxis, temp_mhws_cell, lw=2, label='Temp 100m-avg')
ax1.set_ylabel('Temperature [°C]', fontsize=13)

# Settings x-axis 
ax1.set_xlabel('Days')
ax1.set_xlim(0,181)
ax1.tick_params(axis='x', labelbottom=True, length=1, width=0.5)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

# CHLA
ax2 = ax1.twinx()
ax2.plot(days_xaxis, chla_mhws_cell, lw=2, alpha=0.7, label='Chla Surface', linestyle='--')
ax2.set_ylabel('Chla [mg m$^{-3}$]', fontsize=13)

# Shade MHW periods
for (s, e) in segments:
    ax1.axvspan(days_xaxis[s], days[e], color='#F5562E', alpha=0.5)

# Legend 
mh_patch = Patch(facecolor='#F5562E', alpha=0.5, label=u'\u226530d 4°C MHW')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + [mh_patch], labels1 + labels2 + ['$4^{\circ}C\\ MHWs\\ (\\ge 30\\ days)$'], fontsize=12)

# Labels
ax1.set_title(f"Location: ({float(mhw_duration_cell.lat_rho.values):.2f}, "
              f"{float(mhw_duration_cell.lon_rho.values):.2f}), year {year_idx+1980}",
              fontsize=14)

# ------ Map location ------
ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2*np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)  # center + scale
ax_map.set_boundary(circle, transform=ax_map.transAxes)

# Add features
ax_map.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

# Plot cell
lat_cell = float(mhw_duration_cell.lat_rho.values)
lon_cell = float(mhw_duration_cell.lon_rho.values)
ax_map.plot(lon_cell, lat_cell, 'ro', markersize=8,
            transform=ccrs.PlateCarree(), label='MHW cell')

# ax_map.set_extent([lon_cell-10, lon_cell+10, lat_cell-5, lat_cell+5], crs=ccrs.PlateCarree()) # Optional zoom
ax_map.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# ax_map.legend(fontsize=12)
ax_map.set_title("Grid cell location", fontsize=12)

plt.tight_layout()
plt.show()

# %% ====================== Correlation ======================
# == Climatological Drivers -> Mean Chla and T°C  (days, eta, xi) period: 1980-2009
temp_clim = temp_avg_100m_SO_allyrs.isel(years=slice(0,30)) #shape: (30, 181, 231, 360)
temp_clim_mean = temp_clim.mean(dim=['years']) #shape: (181, 231, 360)
chla_clim = chla_surf_SO_allyrs.isel(years=slice(0,30))
chla_clim_mean = chla_clim.mean(dim=['years'])

# == Baseline Correlation
corr_baseline = xr.corr(temp_clim_mean.avg_temp, chla_clim_mean.chla, dim="days")
print('Baseline mean correlation:', np.mean(~np.isnan(corr_baseline)).values)

# == Year Correlation
corr_yr = xr.corr(temp_avg_100m_SO_allyrs.avg_temp.isel(years=year_idx), chla_surf_SO_allyrs.chla.isel(years=year_idx), dim="days")
print(f'{year_idx+1980} mean correlation:', np.mean(~np.isnan(corr_yr)).values)

# == Difference
corr_diff = corr_yr - corr_baseline

# %% ====================== PLOT correlation ======================
fig_width = 6
fig_height = 6

# Create circular boundary for polar plot
theta = np.linspace(0, 2 * np.pi, 200)
center, radius = [0.5, 0.5]
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Figure
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=1, ncols=1)

# South polar projection
ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())

# Set circular boundary
ax.set_boundary(circle, transform=ax.transAxes)

# Plot correlation field
im = ax.pcolormesh(corr_yr.lon_rho, corr_yr.lat_rho, corr_yr,
                   transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=-1, vmax=1, shading="auto", rasterized=True)

# Map features
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.05, label="Correlation coeff")

# Title
ax.set_title("Correlation Map – 1989 \nTemp(100m) vs Surface CHLA", fontsize=14)

plt.tight_layout()
plt.show()



# %%

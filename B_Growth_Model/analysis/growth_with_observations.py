#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thurs 12 June 08:50:32 2025

COmputing growth with satellite observation and comparison with ROMS

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

import copernicusmarine
import cdsapi
from pprint import pprint

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
path_obs = '/nfs/sea/work/mlarriere/mhw_krill_SO/sat_obs'
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
doy_list = list(range(304, 364)) + list(range(0, 121)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

# %% ============= TEMPERATURE =============
# From Copernicus : https://cds.climate.copernicus.eu/datasets/satellite-sea-surface-temperature?tab=overview
import cdsapi

c = cdsapi.Client()

dataset = "satellite-sea-surface-temperature"
request = {
    "variable": "all",
    "processinglevel": "level_4",
    "sensor_on_satellite": "combined_product",
    "version": "2_1",
    "year": ["2017"],
    "month": ["01"],
    "day": [f"{i:02d}" for i in range(1, 32)],
    "format": "netcdf"
}

file_path = os.path.join(path_obs, "sst_satellite_201701.nc")

# Remove existing file if needed
if os.path.exists(file_path):
    os.remove(file_path)

# Download the data
c.retrieve(dataset, request).download(file_path)
print(f"Downloaded to: {file_path}")

# Unzip files imported
import zipfile
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(path_obs)
print("Extracted files:")
print(os.listdir(path_obs))

# Get list of files - daily
nc_files = sorted([
    os.path.join(path_obs, f)
    for f in os.listdir(path_obs)
    if f.endswith(".nc") and "C3S-L4" in f
])

# Load and subset all datasets
datasets = []
for file in nc_files:
    ds = xr.open_dataset(file)
    ds_south = ds.sel(lat=slice(-90, -60))  # Southern Ocean
    ds_south_deg = ds_south['analysed_sst']-273.15 # Convert kelvin to °C 
    datasets.append(ds_south_deg)  # Extract variable of interest

# Combine into a single dataset along time dimension
combined_ds = xr.concat(datasets, dim='time')

# Save southern ocean SST dataset - for the full month
subset_path = os.path.join(path_obs, "sst_satellite_201701_south60.nc")
combined_ds.to_netcdf(subset_path)
print(f"Saved combined subset dataset to: {subset_path}")

# Remove other netcdf files except subset
for f in os.listdir(path_obs):
    if f.endswith(".nc") and f != os.path.basename(subset_path):
        os.remove(os.path.join(path_obs, f))
print("Removed original daily .nc files except subset.")

temp_obs_2017_jan= xr.open_dataset(subset_path).analysed_sst 

# %% ============= CHLOROPHYLL =============
# Obs from CMEMS (copernicus - included different obs including Sea WIS)
# Get list of files - weekly
path_2017 = os.path.join(path_up_obs, '2017')
files_chl_jan = sorted([
    os.path.join(path_2017, f)
    for f in os.listdir(path_2017)
    if f.endswith(".nc") and "201701" in f
])

# Combine file together
datasets_chla = []
for file in files_chl_jan:
    ds = xr.open_dataset(file)
    ds_south = ds.sel(latitude=slice(-90, -60))  # Southern Ocean
    ds_south_surf = ds_south.isel(depth=0).chl # Select surface data
    datasets_chla.append(ds_south_surf)  # Extract variable of interest

ds_chl_2017_jan_weekly = xr.concat(datasets_chla, dim="time")

# %% Align observations dataset
# --- Spatially
import xesmf as xe

# Use chlorophyll lat/lon coords as target grid since 0.25° resolution (same as ROMS)
target_lat = ds_chl_2017_jan_weekly.latitude.values
target_lon = ds_chl_2017_jan_weekly.longitude.values

# Target grid 
target_grid = {
    'lat': target_lat,
    'lon': target_lon
}

# Regridding
regridder = xe.Regridder(temp_obs_2017_jan, target_grid, 'bilinear')
temp_obs_2017_jan_025 = regridder(temp_obs_2017_jan)

# --- Temporally
# Take monthly mean
temp_obs_2017_jan_monthly = temp_obs_2017_jan_025.resample(time='1MS').mean() #shape: (1, 89, 1440)
ds_chl_2017_jan_monthly = ds_chl_2017_jan_weekly.resample(time='1MS').mean() #shape: (1, 89, 1440)
ds_chl_2017_jan_monthly = ds_chl_2017_jan_monthly.rename({'latitude': 'lat', 'longitude': 'lon'})

# %% Investigate chla 
# Extract data and flatten, excluding NaNs
chl_data = ds_chl_2017_jan_monthly.values.flatten()
chl_data = chl_data[~np.isnan(chl_data)]

# Get min and max
chl_min = np.min(chl_data)
chl_max = np.max(chl_data)

print(f"Temp min: {chl_min:.4f} mg/m³")
print(f"Temp max: {chl_max:.4f} mg/m³")

# Plot histogram
plt.figure(figsize=(6, 4))
plt.hist(chl_data, bins=50, color='seagreen', edgecolor='black')
plt.title('Histogram of Chlorophyll concentration \n(January 2017)', fontsize=14)
plt.xlabel('Chla [mg/m3]', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% Plot temperature and chlorophyll
# Prepare data
temp = temp_obs_2017_jan_monthly.isel(time=0)
chl = ds_chl_2017_jan_monthly.isel(time=0)

norm_chla=mcolors.Normalize(vmin=0, vmax=2)

# ---- Figure and Axes ----
fig_width, fig_height = 6, 5
fig = plt.figure(figsize=(fig_width *2, fig_height))
gs = gridspec.GridSpec(1, 2, wspace=0.08, hspace=0.2)

axs = []
for j in range(2): 
    ax = fig.add_subplot(gs[0, j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    axs.append(ax)

# ---- Plot Definitions ----
plot_data = [
    (temp, "Sea Surface Temperature [°C]", 'inferno', None),
    (chl, "Surface Chlorophyll-a [mg/m³]", 'YlGn', norm_chla)
]

ims = []
for i, (data, title, cmap, norm) in enumerate(plot_data):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    import matplotlib.path as mpath
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Coastlines and land
    ax.coastlines(color='black', linewidth=1)
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

    # Plot data if available
    if data is not None:
        lon_name = [name for name in data.coords if 'lon' in name][0]
        lat_name = [name for name in data.coords if 'lat' in name][0]

        im = ax.pcolormesh(
            data[lon_name], data[lat_name], data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading='auto',
            rasterized=True
        )
        ims.append(im)
    else:
        ims.append(None)

    ax.set_title(title, fontsize=14)

# ---- Colorbars ----
# Temperature colorbar
cbar_ax1 = fig.add_axes([0.13, 0.01, 0.32, 0.03])  # [left, bottom, width, height]
cbar1 = fig.colorbar(ims[0], cax=cbar_ax1, orientation='horizontal', extend='both')
cbar1.set_label("°C", fontsize=12)

# Chlorophyll colorbar
cbar_ax2 = fig.add_axes([0.56, 0.01, 0.32, 0.03])
cbar2 = fig.colorbar(ims[1], cax=cbar_ax2, orientation='horizontal', extend='both')
cbar2.set_label("mg/m³", fontsize=12)


plt.suptitle("Satellite observations \n January 2017", fontsize=18, y=1.05)
plt.show()






# %% ============= Run model with Observations =============
import sys
sys.path.append(working_dir+'Growth_Model') 
from B_Growth_Model.Atkinson2006_model import growth_Atkinson2006  # import growth function

output_file = os.path.join(path_growth, "growth_Atkison2006_obs_201701.nc")
if not os.path.exists(output_file):
    growth_obs_2017_jan = growth_Atkinson2006(ds_chl_2017_jan_monthly, temp_obs_2017_jan_monthly) #shape: (1, 89, 1440)
    print(growth_obs_2017_jan.shape)

    # To dataset
    growth_ds = growth_obs_2017_jan.to_dataset(name='growth')

    # Write to file
    growth_ds.to_netcdf(output_file)

else:
    growth_obs_2017_jan = xr.open_dataset(output_file) #shape (89, 1440, 89, 1440)

# Convert longitude from -180..180 to 0..360
growth_obs_2017_jan = growth_obs_2017_jan.assign_coords(
    lon = (growth_obs_2017_jan.lon + 360) % 360  #24 is the start from ROMS (ranging from 24.125 to 383.875)
)

#Sort longitudes so they are increasing from 0 to 360
growth_obs_2017_jan = growth_obs_2017_jan.sortby('lon')

dlat = np.mean(np.abs(np.diff(growth_obs_2017_jan['lat'].values)))
dlon = np.mean(np.abs(np.diff(growth_obs_2017_jan['lon'].values)))
print(f"OBS -- Approx. grid resolution: {dlat:.3f}° lat × {dlon:.3f}° lon")
 
# %% === Plot
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width
fig = plt.figure(figsize=(fig_width, fig_height))  # Only 1 plot now
gs = gridspec.GridSpec(1, 1)

axs = []
ax = fig.add_subplot(gs[0, 0], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
axs.append(ax)

# Only one plot
plot_data = [
    (growth_obs_2017_jan.growth.isel(time=0), "Observations")
]

# Color normalization
vmin, vmax = -0.2, 0.2
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
cmap = 'PuOr_r'

# -----------------------------
# Plotting
# -----------------------------
ims = []
for data, title in plot_data:
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

    lon_name = [name for name in data.coords if 'lon' in name][0]
    lat_name = [name for name in data.coords if 'lat' in name][0]
    im = ax.pcolormesh(
        data[lon_name],
        data[lat_name],
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading='auto',
        zorder=1,
        rasterized=True
    )
    ims.append(im)

    # ax.set_title(title, fontsize=15)

# -----------------------------
# Colorbar
# -----------------------------
cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.03])  # [left, bottom, width, height]
tick_positions = [-0.2, -0.1, 0.0, 0.1, 0.2]
cbar = fig.colorbar(
    ims[0], cax=cbar_ax, orientation='horizontal',
    ticks=tick_positions, extend='both'
)
cbar.set_label("Growth [mm]", fontsize=14)
cbar.ax.tick_params(labelsize=13)

plt.suptitle("Growth with observations – January 2017", fontsize=20, y=1.05)
plt.show()

# %% ============= ROMS =============
# Resulting growth with ROMS inputs
growth_ROMS = xr.open_dataset(os.path.join(path_growth, 'growth_Atkison2006_fullyr.nc'))

# Select January 2017
growth_ROMS_2017 = growth_ROMS.isel(years=2017-1980)

# Mean growth for this period
growth_ROMS_2017_jan_monthly = growth_ROMS_2017.isel(days=slice(0,30)).mean(dim='days') #shape: (eta_rho, xi_rho) : (231, 1442)

# Resolution
dlat = np.mean(np.abs(np.diff(growth_ROMS_2017_jan_monthly['lat_rho'][:, 0].values)))
dlon = np.mean(np.abs(np.diff(growth_ROMS_2017_jan_monthly['lon_rho'][0, :].values)))
print(f"ROMS -- Approx. grid resolution: {dlat:.3f}° lat × {dlon:.3f}° lon")

from geopy.distance import geodesic  # more accurate than haversine
lat = growth_ROMS_2017_jan_monthly['lat_rho'].values
lon = growth_ROMS_2017_jan_monthly['lon_rho'].values

# Compute meridional (north-south) distance between adjacent eta_rho points
dy = np.zeros(lat.shape[0] - 1)
for i in range(len(dy)):
    p1 = (lat[i, 0], lon[i, 0])
    p2 = (lat[i+1, 0], lon[i+1, 0])
    dy[i] = geodesic(p1, p2).km

# Compute zonal (east-west) distance between adjacent xi_rho points
dx = np.zeros(lon.shape[1] - 1)
for j in range(len(dx)):
    p1 = (lat[0, j], lon[0, j])
    p2 = (lat[0, j+1], lon[0, j+1])
    dx[j] = geodesic(p1, p2).km

avg_dy_km = np.mean(dy)
avg_dx_km = np.mean(dx)

print(f"ROMS grid resolution (approx): {avg_dy_km:.2f} km (north-south) × {avg_dx_km:.2f} km (east-west)")


# %% Align model and observations
# To avoid interpolating and thus creating new data - comparison will be made using the nearest neighbors 

# Flatten model grid to lat/lon/growth arrays
model_lat = growth_ROMS_2017_jan_monthly['lat_rho'].values.ravel()
model_lon = growth_ROMS_2017_jan_monthly['lon_rho'].values.ravel()
model_growth = growth_ROMS_2017_jan_monthly['growth'].values.ravel()

# Remove points where any of lat/lon/growth are NaN
valid = np.isfinite(model_lat) & np.isfinite(model_lon) & np.isfinite(model_growth)
model_lat_valid = np.asarray(model_lat)[valid]
model_lon_valid = np.asarray(model_lon)[valid]
model_growth_valid = np.asarray(model_growth)[valid]

# Get observation grid points
obs_lat = growth_obs_2017_jan['lat'].values
obs_lon = growth_obs_2017_jan['lon'].values

print(obs_lon.min(), obs_lon.max())
print(model_lon_valid.min(), model_lon_valid.max())

# Make 2D mesh of obs grid
obs_lon2d, obs_lat2d = np.meshgrid(obs_lon, obs_lat)

# Flatten for matching
obs_points = np.column_stack((obs_lat2d.ravel(), obs_lon2d.ravel()))
model_points = np.column_stack((model_lat_valid, model_lon_valid))

# Build KD-tree on model points
from scipy.spatial import cKDTree
tree = cKDTree(model_points)

# Query model points nearest to each obs grid point
_, indices = tree.query(obs_points, k=20, distance_upper_bound=0.2) #indices: shape:(128160, 20)

# Initialize output array
model_growth_KNN = np.full(obs_points.shape[0], np.nan)

# For each obs point, average the nearby model growth values
for i, idx_array in enumerate(indices):
    # idx_array can be a list of up to k indices (or invalid entries >= len(model_growth))
    valid_idx = idx_array[idx_array < len(model_growth_valid)]
    if valid_idx.size > 0:
        model_growth_KNN[i] = np.nanmean(model_growth_valid[valid_idx])

model_growth_KNN_reshaped = model_growth_KNN.reshape(obs_lat2d.shape)
model_growth_KNN_da = xr.DataArray(
    model_growth_KNN_reshaped,
    coords={'lat': obs_lat, 'lon': obs_lon},
    dims=['lat', 'lon'],
    name='model_growth_KNN'
)
# %% Differences Obs-ROMS

diff = growth_obs_2017_jan.growth - model_growth_KNN_da #shape: (1, 89, 1440)



# %% ======== Plot
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width
fig = plt.figure(figsize=(fig_width * 3, fig_height))  # Adjust if you add more plots
gs = gridspec.GridSpec(1, 3, wspace=0.08, hspace=0.2)

axs = []
for j in range(3): 
    ax = fig.add_subplot(gs[0, j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    axs.append(ax)

plot_data = [
    (growth_obs_2017_jan.growth.isel(time=0), "Observations"), #shape: (89, 1440)
    (growth_ROMS_2017_jan_monthly.growth, "ROMS"),  #shape: (231, 1442)
    (diff.isel(time=0),  r"Difference$_{Obs-ROMS}$") #shape: (1, 89, 1440)
]

norm_growth=mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
cmpa_growth = 'PuOr_r'

norm_diff = mcolors.TwoSlopeNorm(vmin=diff.min(), vcenter=0, vmax=diff.max())
cmpa_diff = 'coolwarm'

from matplotlib import colors
vmin, vmax = -0.2, 0.2
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

# -----------------------------
# Loop over plots
# -----------------------------
ims = []
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
    if "Difference" in title:
        cmap = cmpa_diff
        color_norm = norm_diff
    else:
        cmap = cmpa_growth
        color_norm = norm_growth

    lon_name = [name for name in data.coords if 'lon' in name][0]
    lat_name = [name for name in data.coords if 'lat' in name][0]
    im = ax.pcolormesh(
        data[lon_name],
        data[lat_name],
        data,
        transform=ccrs.PlateCarree(),
        cmap = cmap,
        norm=color_norm,
        shading='auto',
        zorder=1,
        rasterized=True
    )
    ims.append(im)

    ax.set_title(title, fontsize=15)

# -----------------------------
# Common colorbar
# -----------------------------
# tick_positions = [-0.2, -0.1, 0.0, 0.1, 0.2]
# cbar = fig.colorbar(
#     im, ax=axs, orientation='horizontal',
#     fraction=0.09, pad=0.1, 
#     ticks=tick_positions, 
#     extend='both'
# )
# cbar.set_label("Growth [mm]", fontsize=14)
# cbar.ax.tick_params(labelsize=13) 

# Common colorbar for first two plots (Observations and ROMS)
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.03])  # [left, bottom, width, height]
tick_positions = [-0.2, -0.1, 0.0, 0.1, 0.2]
cbar = fig.colorbar(
    ims[0], cax=cbar_ax, orientation='horizontal',
    ticks=tick_positions, extend='both'
)
cbar.set_label("Growth [mm]", fontsize=14)
cbar.ax.tick_params(labelsize=13)

# Separate colorbar for the difference plot (3rd plot)
cbar_diff_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # [left, bottom, width, height]
cbar_diff = fig.colorbar(
    ims[2], cax=cbar_diff_ax, orientation='vertical',
    extend='both'
)
cbar_diff.set_label("Growth difference [mm]", fontsize=13)
cbar_diff.ax.tick_params(labelsize=12)


plt.suptitle("Difference in Growth between Observational and Modelled data\n January 2017", fontsize=20, y=1.1)
plt.show()
# %%

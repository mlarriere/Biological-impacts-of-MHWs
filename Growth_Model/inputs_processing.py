#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Friday 04 April 16:41:11 2025

Processing chlorophyll data from ROMS

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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

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
    "axes.titlesize": 10,
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
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

file_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_surface.nc' # drift and bias corrected temperature files
path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_chla_corrected = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/' #files: TOT_CHL_BC_*nc 
# /nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/TOT_CHL_surface_daily_corrected.nc
# /nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/TOT_CHL_surface_monthly_corrected.nc

path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
path_growth_inputs_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs/austral_summer'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434
nxi = 1442

# %% Comparison chla before and after trend correection 
# Load datasets
after_chla = xr.open_dataset(os.path.join(path_chla_corrected, 'TOT_CHL_surface_daily_corrected.nc'))
before_chla = xr.open_dataset(os.path.join(path_chla_corrected, 'TOT_CHL_BC_2017.nc'))

# Extract data
after_day360 = after_chla['TOT_CHL'].isel(year=37, day=360)
before_day360 = before_chla['TOT_CHL'].isel(time=360)

# Set figure size
fig_width = 5  # adjust as needed
fig_height = fig_width
fig = plt.figure(figsize=(fig_width*2, fig_height))  # 2 subplots
gs = gridspec.GridSpec(1, 2, wspace=0.05, hspace=0.2)
axs = []
for j in range(2):
    ax = fig.add_subplot(gs[0, j], projection=ccrs.SouthPolarStereo())
    axs.append(ax)

data_list = [after_day360, before_day360]
titles = ["After Correction (2017 - Day 360)", "Before Correction (2017 - Day 360)"]

for i, data in enumerate(data_list):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map features
    ax.coastlines(color='black', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xlabel_style = gl.ylabel_style = {'size': 8}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot
    im = ax.pcolormesh(after_chla.lon_rho, after_chla.lat_rho, data,
                       transform=ccrs.PlateCarree(),
                       cmap='viridis', shading='auto', vmin=0, vmax=5, rasterized=True)
    ax.set_title(titles[i], fontsize=12)

cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.06, pad=0.07)
cbar.set_label('Chl (mg/m³)', fontsize=10)
plt.tight_layout()
plt.show()


#%% ============================== ROMS chlorophyll in [mg Chla/m3] ==============================
# -------------------------------- Weighted averaged Chla --------------------------------
def mean_chla(yr):
    # yr=1
    start_time = time.time()

    # Read data
    # ds_chla = xr.open_dataset(os.path.join(path_chla, f"z_SO_d025_avg_daily_{1980+yr}.nc"))
    # ds_chla_100m = ds_chla.isel(time= slice(0,365), depth=slice(0, 14)) #depth from 0 to 100m depth
    ds_chla_surf_correct = xr.open_dataset(os.path.join(path_chla_corrected, f"TOT_CHL_BC_{1980+yr}.nc")) #corrected and surface chla-- shape:time: 365eta_rho: 434xi_rho: 1442
    # ds_chla_surf_correct = ds_chla.isel(time= slice(0,365)) #corrected using SeaWIFS at surf

    # Reformating
    ds_chla_mean_yr = ds_chla_surf_correct.rename({'TOT_CHL':'raw_chla'}) # Rename variable
    ds_chla_mean_yr = ds_chla_mean_yr.rename({'time': 'days'})# Rename dimension
    ds_chla_mean_yr = ds_chla_mean_yr.assign_coords(lon_rho = ds_roms.lon_rho , lat_rho=ds_roms.lat_rho, days=np.arange(0,365)) # Replace cftime with integer day-of-year: 1 to 365, (lat, lon) coordinates from temperature dataset
    ds_chla_mean_yr = xr.Dataset({'raw_chla': ds_chla_mean_yr.raw_chla}).expand_dims(year=[1980 + yr]) # To dataset and adding year dimension - shape (year:1, eta_rho:434, xi_rho:1442, days:365), max: 2091.5225mg/m3, min: 0mg/m3

    # === Select extent - south of 60°S
    south_mask = ds_chla_mean_yr['lat_rho'] <= -60
    chla_60S_south = ds_chla_mean_yr.where(south_mask, drop=True) #shape (1, 365, 14, 231, 1442)

    # === Cap the data - max 5mg/m3 ===
    chla_filtered = chla_60S_south.where(chla_60S_south.raw_chla <= 5) #shape (years:1, days:365, eta_rho:231, xi_rho:1442)

    # === PLOT ===
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    # Histogram (left)
    ax0 = fig.add_subplot(gs[0])
    ax0.hist(chla_filtered.raw_chla.values.flatten(), bins=50, color='teal', alpha=0.75)
    ax0.set_title(f"Histogram of Surface Chlorophyll\n(South of 60°S, {yr+1980})")
    ax0.set_xlabel("Chlorophyll-a [mg/m³]")
    ax0.set_ylabel("Frequency")
    ax0.grid(True, alpha=0.3)
    # Map (right)
    ax1 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
    ax1.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax1.set_facecolor('lightblue')
    p = ax1.pcolormesh(
        chla_filtered['lon_rho'], chla_filtered['lat_rho'],
        chla_filtered['raw_chla'].isel(year=0, days=330),
        transform=ccrs.PlateCarree(), shading='auto', cmap='viridis')
    cbar = plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.04, aspect=30, ax=ax1)
    cbar.set_label('Chlorophyll-a (mg/m³)')
    ax1.set_title(f'Chlorophyll-a\nDay {330} -- {yr+1980}', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Write dataset to file
    output_file = os.path.join(path_growth_inputs, f"chla_surf_daily_{1980+yr}.nc") #chla_avg100m_daily_{1980+yr}.nc
    if not os.path.exists(output_file):
        chla_filtered.to_netcdf(output_file, mode='w')  

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for year {1980+yr}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")

process_map(mean_chla, range(0, nyears), max_workers=30, desc="Processing year")  #computing time ~3min per year

# ==== Combining years
files_chla_yearly = sorted(glob.glob(os.path.join(path_growth_inputs, "chla_surf_daily_*.nc"))) #chla_avg100m_daily_*.nc
datasets = [xr.open_dataset(f) for f in files_chla_yearly]
chla_mean_all = xr.concat(datasets, dim='year')
# Write dataset to file
output_file = os.path.join(path_growth_inputs, f"chla_surf_allyears.nc") #chla_avg100m_allyears.nc
if not os.path.exists(output_file):
    chla_mean_all.to_netcdf(output_file, mode='w')  

# %% -------------------------------- Chla trend corrected --------------------------------
chla_trend_corrected = xr.open_dataset(os.path.join(path_chla_corrected, 'TOT_CHL_surface_daily_corrected.nc')) #shape (40, 365, 434, 1442)

# === Select extent - south of 60°S
south_mask = chla_trend_corrected['lat_rho'] <= -60
chla_trend_corrected_60S_south = chla_trend_corrected.where(south_mask, drop=True) #shape (40, 365, 231, 1442)
chla_trend_corrected_60S_south = chla_trend_corrected_60S_south.rename({'TOT_CHL':'raw_chla'}) # Rename variable

# Investigating data
max_chla = chla_trend_corrected_60S_south.raw_chla.max() # 189.182336 mg/m3
min_chla = chla_trend_corrected_60S_south.raw_chla.min() # 4.88155679e-08 mg/m3

# === Cap the data - max 5mg/m3 and min 0mg/m3 ===
chla_trend_corrected_filtered = chla_trend_corrected_60S_south.where(chla_trend_corrected_60S_south.raw_chla <= 5) #shape (years:1, days:365, eta_rho:231, xi_rho:1442)

# Investigating data
max_chla_after = chla_trend_corrected_filtered.raw_chla.max() # 4.9999994 mg/m3
min_chla_after = chla_trend_corrected_filtered.raw_chla.min() # 4.88155679e-08 mg/m3

# === PLOT ===
data = chla_trend_corrected_filtered.isel(year=37)
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
# Histogram (left)
ax0 = fig.add_subplot(gs[0])
ax0.hist(data.raw_chla.values.flatten(), bins=50, color='teal', alpha=0.75)
ax0.set_title(f"Histogram of Surface Chlorophyll\n(South of 60°S, {37+1980})")
ax0.set_xlabel("Chlorophyll-a [mg/m³]")
ax0.set_ylabel("Frequency")
ax0.grid(True, alpha=0.3)
# Map (right)
ax1 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
ax1.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND, facecolor='lightgrey')
ax1.set_facecolor('lightblue')
p = ax1.pcolormesh(
    data['lon_rho'], data['lat_rho'],
    data['raw_chla'].isel(day=360),
    transform=ccrs.PlateCarree(), shading='auto', cmap='viridis')
cbar = plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.04, aspect=30, ax=ax1)
cbar.set_label('Chlorophyll-a (mg/m³)')
ax1.set_title(f'Chlorophyll-a\nDay {330} -- {37+1980}', fontsize=12)
plt.tight_layout()
plt.show()

# Write dataset to file
output_file = os.path.join(path_growth_inputs, f"chla_surf_allyears_detrended.nc")
if not os.path.exists(output_file):
    chla_trend_corrected_filtered.to_netcdf(output_file, mode='w')  

#%% Monthly
chla_trend_corrected_monthly = xr.open_dataset(os.path.join(path_chla_corrected, 'TOT_CHL_surface_monthly_corrected.nc')) #shape (40, 12, 434, 1442)

# Reformating
ds_chla_mean_monthly = chla_trend_corrected_monthly.rename({'TOT_CHL':'raw_chla'}) # min: 6.16463534e-07, max:143.92303515 mg/m3

# === Select extent - south of 60°S
south_mask = ds_chla_mean_monthly['lat_rho'] <= -60
chla_60S_south_monthly = ds_chla_mean_monthly.where(south_mask, drop=True) #shape (40, 12, 231, 1442)

# === Cap the data - max 5mg/m3 ===
chla_trend_corrected_monthly = chla_60S_south_monthly.where(chla_60S_south_monthly.raw_chla <= 5)

# Investigating data
max_chla_after = chla_trend_corrected_monthly.raw_chla.max() # 4.9998512 mg/m3
min_chla_after = chla_trend_corrected_monthly.raw_chla.min() # 8.64039176e-05 mg/m3

# === PLOT ===
data = chla_trend_corrected_monthly.isel(year=37, month=11)
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
# Histogram (left)
ax0 = fig.add_subplot(gs[0])
ax0.hist(data.raw_chla.values.flatten(), bins=50, color='teal', alpha=0.75)
ax0.set_title(f"Histogram of Surface Chlorophyll\n(South of 60°S, {37+1980})")
ax0.set_xlabel("Chlorophyll-a [mg/m³]")
ax0.set_ylabel("Frequency")
ax0.grid(True, alpha=0.3)
# Map (right)
ax1 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
ax1.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND, facecolor='lightgrey')
ax1.set_facecolor('lightblue')
p = ax1.pcolormesh(
    data['lon_rho'], data['lat_rho'],
    data['raw_chla'],
    transform=ccrs.PlateCarree(), shading='auto', cmap='viridis')
cbar = plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.04, aspect=30, ax=ax1)
cbar.set_label('Chlorophyll-a (mg/m³)')
ax1.set_title(f'Chlorophyll-a\nMonth {12} -- {37+1980}', fontsize=12)
plt.tight_layout()
plt.show()

# Write dataset to file
output_file = os.path.join(path_growth_inputs, f"chla_surf_allyears_monthly_detrended.nc")
if not os.path.exists(output_file):
    chla_trend_corrected_monthly.to_netcdf(output_file, mode='w')  

# %% Temperature from ROMS [°C]
# -------------------------------- Weighted averaged Temperature --------------------------------
ds = xr.open_dataset(path_mhw + file_var + 'eta200.nc') #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
all_depths= ds['z_rho'].values

def mean_temp(ieta, yr):
    # ieta=200
    # yr=40
    start_time = time.time()

    # Read data
    fn = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_' + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_temp_100m = xr.open_dataset(fn)['temp'].isel(year=yr, z_rho=slice(0,14)) #Extracts daily data : 40yr, consider 365 days per year and until 100m depth

    # Reformating
    ds_temp_mean_yr = ds_temp_100m.rename({'day': 'days'})
    ds_temp_mean_yr = ds_temp_mean_yr.rename({'z_rho': 'depth'})
    
    # Check if there is at least 14 layers to compute the mean
    valid_depth = ~ds_temp_mean_yr.isnull().any(dim='depth') # True where all 14 layers are valid, i.e. Non Nan values
    ds_temp_mean_valid = ds_temp_mean_yr.where(valid_depth)

    # Compute the mean - if not enough vertical layers, mean = Nan
    depth_values = np.abs(ds_temp_mean_valid.depth.values)
    depth_thickness = np.diff(depth_values, prepend=0) # Compute the thickness of each depth layer
    depth_thickness_da = xr.DataArray(depth_thickness,  # shape (14,)
                                      dims=["depth"],
                                      coords={"depth": ds_temp_mean_valid.depth}
                                      )
    da_temp_weighted_mean = (ds_temp_mean_valid * depth_thickness_da).sum(dim='depth') / depth_thickness.sum() # Need to consider that the cell don't have the same height -- WEIGHTING
    # da_temp_weighted_mean.isel(days=100).plot()
    
    elapsed_time = time.time() - start_time
    print(f"Processing time for eta {ieta}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")

    return da_temp_weighted_mean

from functools import partial
for yr in range(1, 41):
    # yr=40
    print(f'------------ YEAR {1979+yr} ------------')
    extract_year_eta_for_yr = partial(mean_temp, yr=yr)
    da_temp_list = process_map(extract_year_eta_for_yr, range(0, neta), max_workers=30, desc="Processing ieta for 1yr")  #computing time ~5min per yr
    
    # Combine eta
    da_temp_combined = xr.concat(da_temp_list, dim='ieta')
    da_temp_combined = da_temp_combined.rename({'ieta': 'eta_rho'})
    da_temp_combined_transposed = da_temp_combined.transpose('days', 'eta_rho', 'xi_rho')

    # === Correct coords -- (lat, lon) must be 2 dims (eta_rho, xi_rho)
    da_temp_combined_transposed = da_temp_combined_transposed.assign_coords({'lon_rho': (('eta_rho', 'xi_rho'), ds_roms['lon_rho'].values),
                                                                             'lat_rho': (('eta_rho', 'xi_rho'), ds_roms['lat_rho'].values)})
    
    # === Select extent - south of 60°S
    print('Select extent')
    south_mask = da_temp_combined_transposed['lat_rho'] <= -60
    temp_60S_south = da_temp_combined_transposed.where(south_mask, drop=True) #shape (181, 231, 1442)

    # Write dataset to file
    output_file = os.path.join(path_growth_inputs, f"temp_avg100m_daily_{1979+yr}.nc")
    if not os.path.exists(output_file):
        temp_60S_south.to_netcdf(output_file, mode='w')  

# ==== Combining years
files_temp_yearly = sorted(glob.glob(os.path.join(path_growth_inputs, "temp_avg100m_daily_*.nc")))
datasets = [xr.open_dataset(f) for f in files_temp_yearly]
temp_mean_all = xr.concat(datasets, dim='year')
temp_mean_all = temp_mean_all.rename({'__xarray_dataarray_variable__':'avg_temp'})
# Write dataset to file
output_file = os.path.join(path_growth_inputs, f"temp_avg100m_allyears.nc")
if not os.path.exists(output_file):
    temp_mean_all.to_netcdf(output_file, mode='w')  

#%% Visualization
# === Parameters ===
# Chosen to be min or max krill growth according to 1st_attempt.py
yr = 2010  #2018
eta = 230 #6 
xi = 1060 #670 
day = 35 #11

# === Temperature Profile ===
fn_temp = f'/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_eta{eta}.nc'
test_temp = xr.open_dataset(fn_temp)['temp'].isel(year=slice(1, 41), z_rho=slice(0, 14))
profile_temp = test_temp.isel(xi_rho=xi, day=day, year=yr - 1980)
temp_profile = profile_temp.values
depth_levels = profile_temp.z_rho.values
mean_temp = np.nanmean(temp_profile)

# === Chlorophyll-a Profile ===
ds_chla = xr.open_dataset(os.path.join(path_chla, f"z_SO_d025_avg_daily_{yr}.nc"))
ds_chla_100m = ds_chla.isel(time=slice(0, 365), depth=slice(0, 14))
profile_chla = ds_chla_100m.TOT_CHL.isel(eta_rho=eta, xi_rho=xi, time=day)
chla_profile = profile_chla.values
depth_chla = ds_chla_100m.depth.values
mean_chla = np.nanmean(chla_profile)

# === Map Data ===
ds_temp_100m_mean = xr.open_dataset(os.path.join(path_growth_inputs, "temp_avg100m_yearly_60S.nc"))['avg_temp']
map_data = ds_temp_100m_mean.isel(years=yr - 1980, days=day)
point_data = ds_temp_100m_mean.isel(eta_rho=eta, xi_rho=xi, years=yr - 1980, days=day)

# === Create Figure with 3 Subplots ===
fig = plt.figure(figsize=(18, 6))

# --- Map subplot ---
ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax1.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)

map_plot = map_data.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False,
    cmap='viridis'
)
ax1.scatter(point_data.lon_rho.item(), point_data.lat_rho.item(), c='red', transform=ccrs.PlateCarree(), s=70, edgecolor='black', linewidth=0.5, zorder=10, marker='*')
ax1.coastlines(color='black', linewidth=1.5, zorder=1)
ax1.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
ax1.set_title(f"Avg Temp (0-100m)\nDay {day}, {yr}", fontsize=12)

# --- Temperature profile subplot ---
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(temp_profile, depth_levels, marker='o', color='tomato')
ax2.set_xlabel('Temperature [°C]')
ax2.set_ylabel('Depth [m]')
ax2.set_title('Temperature Profile')
ax2.set_ylim(-110, 0)
ax2.grid(True)
# ax2.text(0.78, 0.95, f'Mean: {mean_temp:.5f} °C', ha='left', va='top', transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white'))

# --- Chlorophyll-a profile subplot ---
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(chla_profile, depth_chla, marker='o', color='green')
ax3.set_xlabel('Chl-a [mg m$^{-3}$]')
ax3.set_title('Chl-a Profile')
ax3.set_ylim(-210, 0)
ax3.grid(True)
# ax3.text(0.72, 0.95, f'Mean: {mean_chla:.2f} mg/m³', ha='left', va='top', transform=ax3.transAxes, fontsize=10, bbox=dict(facecolor='white'))

# === Final Layout ===
fig.suptitle(f'Vertical Profile of Chla and Temperature', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




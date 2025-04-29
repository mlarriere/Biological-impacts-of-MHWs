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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import time
from tqdm.contrib.concurrent import process_map

from joblib import Parallel, delayed

# %% -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

file_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_surface.nc' # drift and bias corrected temperature files
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
os.makedirs(os.path.join(path_growth_inputs, "austral_summer"), exist_ok=True)
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

# det_combined_ds = xr.open_dataset(os.path.join(path_det, f"det_rel_abs_combined.nc")) #detected event (SST>abs and rel threshold) - boolean

#%% ROMS chlorophyll in [mg Chla/m3]
def mean_chla(yr):

    start_time = time.time()

    # Read data
    yr=0
    ds_chla = xr.open_dataset(os.path.join(path_chla, f"z_SO_d025_avg_daily_{1980+yr}.nc"))
    ds_chla_100m = ds_chla.isel(time= slice(0,365), depth=slice(0, 14)) #depth from 0 to 100m depth

    # Check if there is at least 14 layers to compute the mean
    valid_depth = ~ds_chla_100m.isnull().any(dim='depth') # True where all 14 layers are valid, i.e. Non Nan values
    ds_chla_100m_valid = ds_chla_100m.where(valid_depth)

    # Compute the mean - if not enough vertical layers, mean = Nan
    da_chla_mean = ds_chla_100m_valid.TOT_CHL.mean(dim='depth') #Negative values -- Can't be!!
    da_chla_mean = xr.where(da_chla_mean < 0, 0, da_chla_mean) # Set negative values to 0

    # Reformating
    da_chla_mean = da_chla_mean.rename('raw_chla') # Rename variable
    da_chla_mean = da_chla_mean.rename({'time': 'days'})# Rename dimension
    da_chla_mean = da_chla_mean.assign_coords(lon_rho = det_combined_ds.lon_rho , lat_rho=det_combined_ds.lat_rho, days=np.arange(1, 366)) # Replace cftime with integer day-of-year: 1 to 365, (lat, lon) coordinates from temperature dataset
    ds_chla_mean_yr = xr.Dataset({'raw_chla': da_chla_mean}).expand_dims(year=[1980 + yr]) # To dataset and adding year dimension


    # Write dataset to file
    ds_chla_mean_yr.to_netcdf(path=os.path.join(path_growth_inputs, f"chla_avg100m_daily_{1980+yr}.nc"), mode='w')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for {1980+yr}: {elapsed_time:.2f} seconds")

    return ds_chla_mean_yr 
    
# Calling function
chla_mean_yearly = Parallel(n_jobs=30)(delayed(mean_chla)(yr) for yr in range(0, nyears)) 

# Combine year
chla_mean_all = xr.concat(chla_mean_yearly, dim='year')

# === Select extent - south of 60°S
south_mask = chla_mean_all['lat_rho'] <= -60
chla_100m_mean_south = chla_mean_all.where(south_mask, drop=True) #shape (40, 365, 231, 1442)

# Write to file
chla_100m_mean_south.to_netcdf(path=os.path.join(path_growth_inputs, "chla_avg100m_yearly_60S.nc"), mode='w')

# === Select only austral summer and early spring
file = path_growth_inputs + '/chla_avg100m_yearly_60S.nc'
# Read data
ds_chla = xr.open_dataset(file) #days ranging from idx0 to idx364 -for chla contain already coord days

# Select only austral summer and early spring
jan_april = ds_chla.sel(days=slice(1, 120)) # 1 Jan to 30 April (Day 1-120) 
nov_dec = ds_chla.sel(days=slice(305, 365)) # 1 Nov to 31 Dec (Day 305–365)
ds_austral = xr.concat([nov_dec, jan_april], dim="days") #181days

# Save to file
output_file = os.path.join(path_growth_inputs_summer, f"chla_austral_avg100m.nc")
if not os.path.exists(output_file):
    ds_austral.to_netcdf(output_file, engine="netcdf4")
    
# %% Temperature from ROMS [°C]
def mean_temp(ieta):
    # Read data
    # ieta=200
    fn = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_' + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_temp_100m = xr.open_dataset(fn)['temp'].isel(year=slice(1,41), z_rho=slice(0,14)) #Extracts daily data : 40yr, consider 365 days per year and until 100m depth
    
    # Check if there is at least 14 layers to compute the mean
    valid_depth = ~ds_temp_100m.isnull().any(dim='z_rho') # True where all 14 layers are valid, i.e. Non Nan values
    ds_temp_100m_valid = ds_temp_100m.where(valid_depth)

    # Compute the mean
    ds_temp_100m_mean = ds_temp_100m_valid.mean(dim='z_rho', skipna=True)
    # ds_temp_100m_mean.isnull().sum()

    print(f'Processing temperature for eta {ieta}')
    return ds_temp_100m_mean 

results = Parallel(n_jobs=30)(delayed(mean_temp)(ieta) for ieta in range(0, neta)) 

# Combine eta
temp_100m_mean = np.zeros((nyears, ndays, neta, nxi))  # Shape: (40, 365, neta, nxi)

for ieta in range(0, neta):
    temp_100m_mean[:, :, ieta, :] = results[ieta]

# To dataset
ds_temp_100m_mean = xr.Dataset(
    {"avg_temp": (["years", "days", "eta_rho", "xi_rho"], temp_100m_mean)},
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),
        years=np.arange(1980,2020),
        days = np.arange(1,366)
    ),
    attrs={"description": "Averaged temperature of the first 100m [°C]"}
)

# === Select extent - south of 60°S
south_mask = ds_temp_100m_mean['lat_rho'] <= -60
temp_100m_mean_south = ds_temp_100m_mean.where(south_mask, drop=True) #shape (40, 365, 231, 1442)

# Write to file
temp_100m_mean_south.to_netcdf(path=os.path.join(path_growth_inputs, "temp_avg100m_yearly_60S.nc"), mode='w')

# === Select only austral summer and early spring
file = path_growth_inputs + '/temp_avg100m_yearly_60S.nc'
# Read data
ds_temp = xr.open_dataset(file) #days ranging from idx0 to idx364 -for chla contain already coord days

# Select only austral summer and early spring
jan_april = ds_temp.sel(days=slice(1, 120)) # 1 Jan to 30 April (Day 1-120) 
nov_dec = ds_temp.sel(days=slice(305, 365)) # 1 Nov to 31 Dec (Day 305–365)
ds_austral = xr.concat([nov_dec, jan_april], dim="days") #181days

# Save to file
output_file = os.path.join(path_growth_inputs_summer, f"temp_austral_avg100m.nc")
if not os.path.exists(output_file):
    ds_austral.to_netcdf(output_file, engine="netcdf4")


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


# %%

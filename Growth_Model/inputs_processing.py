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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

#%% ============================== ROMS chlorophyll in [mg Chla/m3] ==============================
# -------------------------------- Weighted averaged Chla --------------------------------
def mean_chla(yr):
    # yr=0
    start_time = time.time()

    # Read data
    ds_chla = xr.open_dataset(os.path.join(path_chla, f"z_SO_d025_avg_daily_{1980+yr}.nc"))
    ds_chla_100m = ds_chla.isel(time= slice(0,365), depth=slice(0, 14)) #depth from 0 to 100m depth

    # Reformating
    ds_chla_mean_yr = ds_chla_100m.rename({'TOT_CHL':'raw_chla'}) # Rename variable
    ds_chla_mean_yr = ds_chla_mean_yr.rename({'time': 'days'})# Rename dimension
    ds_chla_mean_yr = ds_chla_mean_yr.assign_coords(lon_rho = ds_roms.lon_rho , lat_rho=ds_roms.lat_rho, days=np.arange(0,365)) # Replace cftime with integer day-of-year: 1 to 365, (lat, lon) coordinates from temperature dataset
    ds_chla_mean_yr = xr.Dataset({'raw_chla': ds_chla_mean_yr.raw_chla}).expand_dims(year=[1980 + yr]) # To dataset and adding year dimension - shape (year:1, depth:14, eta_rho:434, xi_rho:1442, days:365)

    # === Select extent - south of 60°S
    south_mask = ds_chla_mean_yr['lat_rho'] <= -60
    chla_60S_south = ds_chla_mean_yr.where(south_mask, drop=True) #shape (1, 365, 14, 231, 1442)

    # === Select only austral summer and early spring
    print('--- Austral Summer ---')
    jan_april = chla_60S_south.sel(days=slice(0, 119)) # 1 Jan to 30 April (Day 0-119) 
    nov_dec = chla_60S_south.sel(days=slice(304, 364)) # 1 Nov to 31 Dec (Day 304–364)
    chla_austral_60S_south = xr.concat([nov_dec, jan_april], dim="days") #181days

    # Check if there is at least 14 layers to compute the mean
    valid_depth = ~chla_austral_60S_south.isnull().any(dim='depth') # True where all 14 layers are valid, i.e. Non Nan values
    ds_chla_mean_valid = chla_austral_60S_south.where(valid_depth)

    # Compute the mean - if not enough vertical layers, mean = Nan
    print('--- Computing mean ---')
    depth_values = np.abs(ds_chla_mean_valid.depth.values)
    depth_thickness = np.diff(depth_values, prepend=0) # Compute the thickness of each depth layer
    depth_thickness = depth_thickness[:, np.newaxis, np.newaxis]  # Shape (14, 1, 1)
    da_chla_weighted_mean = (ds_chla_mean_valid.raw_chla * depth_thickness).sum(dim='depth') / depth_thickness.sum() # Need to consider that the cell don't have the same height -- WEIGHTING
    da_chla_weighted_mean = xr.where(da_chla_weighted_mean < 0, 0, da_chla_weighted_mean) # Set negative values to 0
    # da_chla_weighted_mean.isel(days=100).plot()

    # Write dataset to file
    output_file = os.path.join(path_growth_inputs, f"chla_avg100m_daily_{1980+yr}.nc")
    if not os.path.exists(output_file):
        da_chla_weighted_mean.to_netcdf(output_file, mode='w')  

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for year {1980+yr}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")

process_map(mean_chla, range(0, nyears), max_workers=30, desc="Processing year")  #computing time ~3min per year

# %% -------------------------------- Chla at each depth --------------------------------
def chla_process(yr):

    start_time = time.time()

    # Read data
    # yr=0
    ds_chla = xr.open_dataset(os.path.join(path_chla, f"z_SO_d025_avg_daily_{1980+yr}.nc"))
    ds_chla_100m = ds_chla.isel(time= slice(0,365), depth=slice(0, 14)) #depth from 0 to 100m depth

    # Reformating
    ds_chla_100m = ds_chla_100m.rename({'TOT_CHL':'raw_chla'}) # Rename variable
    ds_chla_100m = ds_chla_100m.rename({'time': 'days'})# Rename dimension
    ds_chla_100m = ds_chla_100m.assign_coords(lon_rho = ds_roms.lon_rho , lat_rho=ds_roms.lat_rho, days=np.arange(0, 365)) # Replace cftime with integer day-of-year: 1 to 365, (lat, lon) coordinates from temperature dataset
    ds_chla_mean_yr = xr.Dataset({'raw_chla': ds_chla_100m.raw_chla}).expand_dims(year=[1980 + yr]) # To dataset and adding year dimension - shape (year:1, depth:14, eta_rho:434, xi_rho:1442, days:365)

    # Write dataset to file
    ds_chla_mean_yr.to_netcdf(path=os.path.join(path_growth_inputs, f"chla_daily_{1980+yr}.nc"), mode='w')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for {1980+yr}: {elapsed_time:.2f} seconds")

    return ds_chla_mean_yr 
    
# Calling function
process_map(chla_process, range(0, nyears), max_workers=30, desc="Processing file")  #computing time ~1min per file

# === Select only austral summer and early spring
file = path_growth_inputs + '/chla_avg100m_yearly_60S.nc'

# Read data
ds_chla = xr.open_dataset(file) #days ranging from idx0 to idx364 -for chla contain already coord days

# Combine year per depth
import glob
chla_files = sorted(glob.glob(os.path.join(path_growth_inputs, "chla_daily_*.nc")))

for i in range(14):
    print(f'Depth {i}')
    depth_data = [] # list to store data of different years - same depth
    for file in chla_files:
        # Read data
        ds = xr.open_dataset(file)
        da = ds.isel(depth=13)

        # === Select extent - south of 60°S
        south_mask = da['lat_rho'] <= -60
        chla_60S_south = da.where(south_mask, drop=True) #shape (40, 365, 231, 1442)

        # === Select only austral summer and early spring
        jan_april = chla_60S_south.sel(days=slice(0, 119)) # 1 Jan to 30 April (Day 0-119) 
        nov_dec = chla_60S_south.sel(days=slice(304, 364)) # 1 Nov to 31 Dec (Day 304–364)
        chla_austral_60S_south = xr.concat([nov_dec, jan_april], dim="days") #181days

        # Store
        depth_data.append(chla_austral_60S_south)
        ds.close()

    # Concatenate all years for this depth
    ds_depth = xr.concat(depth_data, dim='year')

    # Save to file
    depth_str = str(int(abs(float(ds_depth.depth.values))))
    fname = f"chla_daily_depth_{depth_str}m.nc"
    ds_depth.to_netcdf(os.path.join(path_growth_inputs, fname))

    
# %% Temperature from ROMS [°C]
# -------------------------------- Weighted averaged Temperature --------------------------------
ds = xr.open_dataset(path_mhw + file_var + 'eta200.nc') #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
all_depths= ds['z_rho'].values

def mean_temp(ieta, yr):
    # ieta=200
    # yr=1
    start_time = time.time()

    # Read data
    fn = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_' + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_temp_100m = xr.open_dataset(fn)['temp'].isel(year=yr, z_rho=slice(0,14)) #Extracts daily data : 40yr, consider 365 days per year and until 100m depth

    # Reformating
    ds_temp_mean_yr = ds_temp_100m.rename({'day': 'days'})
    ds_temp_mean_yr = ds_temp_mean_yr.rename({'z_rho': 'depth'})
    
    # === Select only austral summer and early spring
    jan_april = ds_temp_mean_yr.sel(days=slice(0, 119)) # 1 Jan to 30 April (Day 0-119) 
    nov_dec = ds_temp_mean_yr.sel(days=slice(304, 364)) # 1 Nov to 31 Dec (Day 304–364)
    temp_austral_mean_yr = xr.concat([nov_dec, jan_april], dim="days") #181days

    # Check if there is at least 14 layers to compute the mean
    valid_depth = ~temp_austral_mean_yr.isnull().any(dim='depth') # True where all 14 layers are valid, i.e. Non Nan values
    ds_temp_mean_valid = temp_austral_mean_yr.where(valid_depth)

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
for yr in range(1, 40):
    # yr=0
    print(f'------------ YEAR {1979+yr} ------------')
    extract_year_eta_for_yr = partial(mean_temp, yr=yr)
    da_temp_list = process_map(extract_year_eta_for_yr, range(0, neta), max_workers=30, desc="Processing ieta for 1yr")  #computing time ~5min per yr
    
    # Combine eta
    da_temp_combined = xr.concat(da_temp_list, dim='ieta')
    da_temp_combined = da_temp_combined.rename({'ieta': 'eta_rho'})
    da_temp_combined_transposed = da_temp_combined.transpose('days', 'eta_rho', 'xi_rho')

    # === Select extent - south of 60°S
    print('Select extent')
    south_mask = da_temp_combined_transposed['lat_rho'] <= -60
    temp_60S_south = da_temp_combined_transposed.where(south_mask, drop=True) #shape (181, 231, 1442)

    # Write dataset to file
    output_file = os.path.join(path_growth_inputs, f"temp_avg100m_daily_{1980+yr}.nc")
    if not os.path.exists(output_file):
        temp_60S_south.to_netcdf(output_file, mode='w')  



# %% -------------------------------- Temperature at each depth --------------------------------
ds = xr.open_dataset(path_mhw + file_var + 'eta200.nc') #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
all_depths= ds['z_rho'].values

def extract_year_eta(ieta, depth):
    # depth=1
    # ieta=200

    start_time = time.time()

    # Read data
    fn = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_' + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_temp_100m = xr.open_dataset(fn)['temp'].isel(year=slice(1,41), z_rho=depth) #Extracts daily data : 40yr, consider 365 days per year and until 100m depth


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for eta {ieta}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")
    return ieta, ds_temp_100m.values

from functools import partial

for idepth in range(14):
    # idepth=0
    print(f"=== Processing depth {idepth} ===")

    # Initialization -- Move neta to axis 0 for faster computing
    ds_eta_combined = np.empty((neta, nyears, ndays, nxi), dtype=np.float32) 

    # Extract all eta for the given year and depth and combine them together
    extract_year_eta_for_yr = partial(extract_year_eta, depth=idepth)
    for ieta, ds_eta_all in process_map(extract_year_eta_for_yr, range(neta), max_workers=30): # ~3min computing
        # print(f"Shape of ds_eta_all for eta {ieta}: {ds_eta_all.shape}")  # (40, 365, 1442)
        ds_eta_combined[ieta] = ds_eta_all
    
    # Transpose dimension to have eta on 4th position
    ds_eta_combined_transposed = ds_eta_combined.transpose(1, 2, 0, 3)

    # Check 
    print(np.allclose(ds_eta_combined[200], ds_eta_combined_transposed[:, :, 200, :], equal_nan=True))# True

    # To dataset
    ds_temp_depth = xr.Dataset(
        {"temp": (["years", "days", "eta_rho", "xi_rho"], ds_eta_combined_transposed)},
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),
            years=(['years'], np.arange(1980,2020)),
            original_days=(["days"],np.arange(0,365))
            ),
        attrs={"description": f"Temperature of the first 100m [°C] for depth {-all_depths[idepth]}m, i.e. layer {idepth}"}
    )

    # === Select only austral summer and early spring  --- HERE
    # Select only austral summer and early spring
    jan_april = ds_temp_depth.sel(days=slice(0, 120)) # 1 Jan to 30 April (Day 1-120) 
    nov_dec = ds_temp_depth.sel(days=slice(304, 365)) # 1 Nov to 31 Dec (Day 305–365)
    ds_austral = xr.concat([nov_dec, jan_april], dim="days") #181days

    # === Select extent - south of 60°S
    south_mask = ds_austral['lat_rho'] <= -60
    temp_60S_south = ds_austral.where(south_mask, drop=True) #shape (40, 181, 231, 1442)

    # Write to file
    fname = f"temp_daily_depth_{-all_depths[idepth]}m.nc"
    ds_depth.to_netcdf(os.path.join(path_growth_inputs, fname))


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



# %% Associating temperature occurring during detected events
det_temp_surf = surf_temp.where(det_combined_ds) #extremely long computing!! 
det_temp_surf.to_netcdf(path_det+'det_sst_combined.nc', mode='w')

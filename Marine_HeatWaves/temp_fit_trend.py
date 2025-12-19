"""
Created on Mond 15 Dec 08:19:45 2025

Finding trend from the temperature signal

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
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.gridspec as gridspec

import time
from datetime import datetime, timedelta
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
path_biomass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_mass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/krill_mass'
path_length = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/krill_length'


# %% ======================== Load data ========================
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 365, 231, 1442)

# Concatenate the years together 
temp_stacked = temp_avg_100m_SO_allyrs.stack(time=("year", "days")) #shape (14600, 231, 1442)
temp_stacked = temp_stacked.transpose("time", "eta_rho", "xi_rho")


# %% test
eta_choice = 200
xi_choice = 1000

temp_1cell = temp_avg_100m_SO_allyrs.isel(eta_rho = eta_choice, xi_rho=xi_choice, days=0)

t = np.arange(temp_1cell.avg_temp.size)
plt.figure(figsize=(10,5))
plt.plot(t, temp_1cell.avg_temp)
plt.xlabel("Time index")
plt.ylabel("Temperature (°C)")
plt.show()


# %% ======================== Compute trend ========================

def compute_trend_cell_vectorized(args):
    """
    Compute 1 linear trend for one eta row (all xi).
    Skips cells that are land (all zeros or NaNs).
    Returns slopes, intercepts, and r2 arrays: shape (time, xi)
    """

    # Extract arguments
    ieta = args

    # test
    # ieta=120

    temp_data = temp_stacked
    ntime = temp_data.sizes['time']
    nxi = temp_data.sizes['xi_rho']
    
    # Initialisation
    slopes = np.full(nxi, np.nan)
    intercepts = np.full(nxi, np.nan)
    r2 = np.full(nxi, np.nan)

    # --- Linear regression
    # Create design matrix for linear regression: Y = X @ coeffs
    t = np.arange(ntime) / 365.0   # years
    X = np.column_stack((t, np.ones_like(t))) #shape (time, 2) 

    # Extract Y
    Y = temp_data.isel(eta_rho=ieta).avg_temp.values  # (time, xi)
    
    # Mask land cells, i.e. when all are zeros or NaN -> continue
    # valid_mask = ~(np.all(np.isnan(Y) | (Y == 0), axis=0))
    # if not np.any(valid_mask):
    #     return ieta, slopes, intercepts, r2

    # # Keep valid cells (ocean)
    # Y_valid = Y[:, valid_mask]  # shape (years, n_valid_xi)
    
    # Solve linear regression
    coeffs, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # shape (2, n_valid_xi)

    # Store coeffs of interest
    slopes= coeffs[0] #shape: (nxi) - [°C/day]
    intercepts = coeffs[1] #shape: (nxi)
        
    # --- Evaluate the regression
    # Compute R2 for each xi
    y_mean = np.mean(Y, axis=0)                 # shape: (n_valid_xi,)
    ss_tot = np.sum((Y - y_mean)**2, axis=0)   # shape: (n_valid_xi,)

    # Compute residuals manually
    Y_pred = X @ coeffs                               # shape: (ntime, n_valid_xi)
    ss_res = np.sum((Y - Y_pred)**2, axis=0)   # shape: (n_valid_xi,)

    r2 = np.where(ss_tot != 0, 1 - ss_res / ss_tot, np.nan)

    return ieta, slopes, intercepts, r2

from tqdm.contrib.concurrent import process_map

output_file_100mavg = os.path.join(path_biomass, 'fake_worlds/trends/linear_trend_100mavg.nc')

if not os.path.exists(output_file_100mavg):
    neta = temp_avg_100m_SO_allyrs.sizes['eta_rho']
    nxi  = temp_avg_100m_SO_allyrs.sizes['xi_rho']

    # --- Call function in parallel
    args_list = list(range(neta))
    # args_list = [(ieta, temp_stacked.avg_temp) for ieta in range(neta)]
    results = process_map(compute_trend_cell_vectorized, args_list, max_workers=8, desc="Processing eta") 

    # --- Unwarp results
    # Initialisation
    slopes_all = np.full((neta, nxi), np.nan)
    intercepts_all = np.full((neta, nxi), np.nan)
    r2_all = np.full((neta, nxi), np.nan)

    for res in results:
        ieta, slopes, intercepts, r2 = res
        slopes_all[ieta, :] = slopes #shape (231, 1442)
        intercepts_all[ieta, :] = intercepts #shape (231, 1442)
        r2_all[ieta, :] = r2 #shape (231, 1442)

    # --- To Dataset
    trends_100mavg_ds = xr.Dataset(
        {"slope": (("eta_rho", "xi_rho"), slopes_all),
        "intercept": (("eta_rho", "xi_rho"), intercepts_all),
        "r2": (("eta_rho", "xi_rho"), r2_all),
        },
        coords={
            "lat_rho": temp_avg_100m_SO_allyrs.lat_rho,
            "lon_rho": temp_avg_100m_SO_allyrs.lon_rho,
        },
    )

    trends_100mavg_ds["slope"].attrs = {
        "description": "Linear temperature trend",
        "units": "°C/yr",
    }

    trends_100mavg_ds["intercept"].attrs = {
        "description": "Temperature at t=0, i.e. 1st January 1980",
        "units": "°C",
    }
    # Save
    trends_100mavg_ds.to_netcdf(output_file_100mavg)

else:
    # Load data
    trends_100mavg_ds = xr.open_dataset(output_file_100mavg)

    # Convert to decades
    slopes_dec = trends_100mavg_ds.slope * 10

# %% ======================== Visualisation ========================
# ---------------------------------------
#               Slope Map
# ---------------------------------------
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

# Base map features
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.coastlines(color='black', linewidth=0.7, zorder=3)

# Plot slope
im = ax.pcolormesh(trends_100mavg_ds.lon_rho, trends_100mavg_ds.lat_rho, 
                   trends_100mavg_ds.slope*40, cmap='coolwarm', 
                   vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), zorder=1)
# im = ax.pcolormesh(trends_100mavg_ds.lon_rho, trends_100mavg_ds.lat_rho, trends_ds.r2, cmap='Blues', vmin=0, vmax=0.5, transform=ccrs.PlateCarree(), zorder=1)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
gl.xlabel_style = {'size': 9, 'rotation': 0} 
gl.ylabel_style = {'size': 9, 'rotation': 0} 

# Colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.05, extend='both')
# cbar.set_label("Slope [°C/decade]")
cbar.set_label("Warming [°C]")
# cbar.set_label("R2")

# plt.title('ROMS 100m-avg temperature\n Slope of Linear trend: mx+b', fontsize=15)
plt.title('ROMS 100m-avg temperature\n Warming after 40years', fontsize=15)
# plt.title('ROMS 100m-avg temperature\n Evaluation of Linear trend', fontsize=15)
plt.show()

# %% ======================== Worklow 2 ========================
path_eta = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
ds_test = xr.open_dataset(path_eta + file_var + 'eta' + str(220) + '.nc')['temp'][1:,0:365,:,:]

def linear_trend_depth(ieta):
    # test
    # ieta =200

    # Read data
    fn = path_eta + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)['temp'][1:,0:365,:,:] #Extracts daily data : 40yr + consider 365 days per year. shape:(40, 365, 35, 1442)
    ds_original_100m = ds_original.isel(z_rho=slice(0,14)) #first 100m depth - 14 levels

    ndepth = ds_original_100m.sizes['z_rho']
    nxi = ds_original_100m.sizes['xi_rho']
    ntime = ds_original_100m.sizes['year'] * ds_original_100m.sizes['day']

    # Initialization
    slopes_all = np.full((ndepth, nxi), np.nan)
    intercepts_all = np.full((ndepth, nxi), np.nan)
    r2_all = np.full((ndepth, nxi), np.nan)
    
    Y_all = ds_original_100m.stack(time=('year', 'day')).transpose('time', 'z_rho', 'xi_rho').data #shape (14600, 14, 1442)
    ntime, ndepth, nxi = Y_all.shape

    # Design matrix
    t = np.arange(ntime) - ntime / 2
    X = np.column_stack((t, np.ones_like(t)))  # shape (ntime, 2)

    # Flatten depth & xi for vectorized lstsq
    Y_flat = Y_all.reshape(ntime, ndepth * nxi)

    # Solve regression for all depth*xi at once
    coeffs, _, _, _ = np.linalg.lstsq(X, Y_flat, rcond=None)  # shape: (2, depth*xi)

    # Reshape back to depth x xi
    slopes_all = coeffs[0].reshape(ndepth, nxi)
    intercepts_all = coeffs[1].reshape(ndepth, nxi)

    # Predicted values for R2
    Y_pred = (X @ coeffs).reshape(ntime, ndepth, nxi)
    y_mean = np.mean(Y_all, axis=0)
    ss_tot = np.sum((Y_all - y_mean)**2, axis=0)
    ss_res = np.sum((Y_all - Y_pred)**2, axis=0)
    r2_all = np.where(ss_tot != 0, 1 - ss_res/ss_tot, np.nan)

    # Loop on depth
    # for idepth in range(14):
    #     print('depth: ', idepth)
    #     # idepth=0

    #     # Extract data
    #     temp_depth = ds_original_100m.isel(z_rho=idepth)
    #     # Y = temp_depth.values.reshape(ntime, nxi)
    #     Y = temp_depth.stack(time=('year', 'day')).transpose('time', 'xi_rho').data

    #     # Linear regression
    #     t = np.arange(ntime) - ntime / 2
    #     X = np.column_stack((t, np.ones_like(t)))  # design matrix
    #     coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        
    #     # Store slopes and intercepts
    #     slopes_all[idepth, :] = coeffs[0]
    #     intercepts_all[idepth, :] = coeffs[1]

        
    #     # R2 calculation
    #     Y_pred = X @ coeffs
    #     y_mean = np.mean(Y, axis=0)
    #     ss_tot = np.sum((Y - y_mean)**2, axis=0)
    #     ss_res = np.sum((Y - Y_pred)**2, axis=0)
    #     r2_all[idepth, :] = np.where(ss_tot != 0, 1 - ss_res / ss_tot, np.nan)

    return ieta, slopes_all, intercepts_all, r2_all

from tqdm.contrib.concurrent import process_map

neta = temp_avg_100m_SO_allyrs.sizes['eta_rho']
ndepth =14

output_file = os.path.join(path_biomass, 'fake_worlds/trends/linear_trend_depth.nc')

if not os.path.exists(output_file):
    # --- Call function in parallel
    results = process_map(linear_trend_depth, range(neta), max_workers=8, desc="Processing eta") #231 tuples of shape (14, 1442)

    # --- Unwarp results
    # Initialisation
    slopes_all = np.full((neta, ndepth, nxi), np.nan)
    intercepts_all = np.full((neta, ndepth, nxi), np.nan)
    r2_all = np.full((neta, ndepth, nxi), np.nan)

    for res in results:
        ieta, slopes, intercepts, r2 = res
        slopes_all[ieta, :, :] = slopes #shape (231, 14, 1442)
        intercepts_all[ieta, :, :] = intercepts #shape (231, 14, 1442)
        r2_all[ieta, :, :] = r2 #shape (231, 14, 1442)

    # Transpose
    slopes_transp = slopes_all.transpose(1,0,2)
    intercepts_transp = intercepts_all.transpose(1,0,2)
    r2_transp = r2_all.transpose(1,0,2)

    # To Dataset
    trends_ds = xr.Dataset(
        {"slope": (("depth", "eta_rho", "xi_rho"), slopes_transp),
        "intercept": (("depth", "eta_rho", "xi_rho"), intercepts_transp),
        "r2": (("depth", "eta_rho", "xi_rho"), r2_transp),
        },
        coords={
            "lat_rho": temp_avg_100m_SO_allyrs.lat_rho,
            "lon_rho": temp_avg_100m_SO_allyrs.lon_rho,
            "depth": ds_test.z_rho,
        }
    )
    # Save to file
    trends_ds.to_netcdf(output_file)

else: 
    trends_ds = xr.open_dataset(output_file)


# %% ======================== Visualisation ========================
# ---------------------------------------
#               Slope Map
# ---------------------------------------
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

# Base map features
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.coastlines(color='black', linewidth=0.7, zorder=3)

# Plot slope
im = ax.pcolormesh(trends_ds.isel(depth=13).lon_rho, trends_ds.isel(depth=13).lat_rho, trends_ds.isel(depth=13).slope, 
                   vmin=-1e-4, vmax=1e-4, cmap='plasma',  transform=ccrs.PlateCarree(), zorder=1)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
gl.xlabel_style = {'size': 9, 'rotation': 0} 
gl.ylabel_style = {'size': 9, 'rotation': 0} 

# Colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.05, extend='both')
cbar.set_label("Slope [°C/year]")
# cbar.set_label("R2")

plt.title('ROMS temperature (depth=13)\n Slope of Linear trend: mx+b', fontsize=15)
# plt.title('ROMS 100m-avg temperature\n Evaluation of Linear trend', fontsize=15)
plt.show()
# %% ======================== Removing trend ========================
# -- Reconstruct the linear fit using coeffs from linear regression 
# Preparing (linear trend fitted on daily data over 40 years - need to reconstruct time to have trend)
n_years = temp_avg_100m_SO_allyrs.sizes["year"]
n_days_per_year = temp_avg_100m_SO_allyrs.sizes["days"]
total_days = n_years * n_days_per_year
t_cont = np.arange(total_days) / 365.0  # fractional years
t_da = xr.DataArray(t_cont, dims=["time"], coords={"time": np.arange(total_days)})

intercept = trends_100mavg_ds.intercept
slope = trends_100mavg_ds.slope
slope_exp = slope.expand_dims({"time": total_days}).transpose("time", "eta_rho", "xi_rho") 
intercept_exp = intercept.expand_dims({"time": total_days}).transpose("time", "eta_rho", "xi_rho")

# Linear trend (computation time ~10min)
y_fit_daily = slope_exp * t_da + intercept_exp #shape (14600, 231, 1442)
y_fit_da = xr.DataArray(y_fit_daily, dims=temp_stacked.avg_temp.dims, coords=temp_stacked.avg_temp.coords)

# -- Detrend signal: roms - trend (computation time ~10min)
roms_detrended_yearly = temp_stacked.avg_temp - y_fit_da #shape (14600, 231, 1442)

# -- Unstack time (computation time ~6min)
detrended_unstacked = roms_detrended_yearly.unstack("time").transpose("year", "days", "eta_rho", "xi_rho") #shape(40, 365, 231, 1442)


# -- Select only growth season
jan_april_detrend = detrended_unstacked.sel(days=slice(0, 119)) # 1 Jan to 30 April (Day 0-119: 120days) - last idx excluded
jan_april_detrend.coords['days'] = jan_april_detrend.coords['days'] #keep info on day
nov_dec_detrend = detrended_unstacked.sel(days=slice(304, 364)) # 1 Nov to 31 Dec (Day 304–364: 61days) - last idx excluded
nov_dec_detrend.coords['days'] = np.arange(304, 365) #keep info on day
detrended_season = xr.concat([nov_dec_detrend, jan_april_detrend], dim="days") #shape: (181, 231, 1442)

# -- To dataset
detrended_season_ds = xr.Dataset({"temp_detrend": detrended_season})
detrended_season_ds["temp_detrend"].attrs["description"] = (
    "Daily ROMS temperature with linear trend removed at each grid cell. "
    "Trend estimated using OLS (over daily signal of 40years). "
)

# -- Save to file
detrend_file = os.path.join(path_biomass, f'fake_worlds/trends/temp_detrended.nc')
if not os.path.exists(detrend_file):
    detrended_season_ds.to_netcdf(detrend_file)


# # Loop over years and save 
# for yr in range(39):
#     print(f'Processing {yr+1980}')
#     # test
#     # yr=38
#     # Extract day
#     detrended_yr = detrended_unstacked.isel(year=yr)

#     # -- Select only growth season
#     jan_april_detrend = detrended_yr.sel(days=slice(0, 119)) # 1 Jan to 30 April (Day 0-119: 120days) - last idx excluded
#     jan_april_detrend.coords['days'] = jan_april_detrend.coords['days'] #keep info on day
#     nov_dec_detrend = detrended_yr.sel(days=slice(304, 364)) # 1 Nov to 31 Dec (Day 304–364: 61days) - last idx excluded
#     nov_dec_detrend.coords['days'] = np.arange(304, 365) #keep info on day
#     detrended_yr_season = xr.concat([nov_dec_detrend, jan_april_detrend], dim="days") #shape: (181, 231, 1442)

#     # -- To dataset
#     detrended_yr_season_ds = xr.Dataset({"temp_detrend": detrended_yr_season})
#     detrended_yr_season_ds["temp_detrend"].attrs["description"] = (
#         "Daily ROMS temperature with linear trend removed at each grid cell. "
#         "Trend estimated using OLS (over daily signal of 40years). "
#     )

#     # -- Save to file
#     detrend_file = os.path.join(path_biomass, f'fake_worlds/trends/temp_detrend_{yr+1980}.nc')
#     if not os.path.exists(detrend_file):
#         detrended_yr_season_ds.to_netcdf(detrend_file)




# %% ======================== Visualisation ========================
eta_choice = 200
xi_choice = 1100
lat = temp_avg_100m_SO_allyrs.isel(eta_rho=eta_choice, xi_rho=xi_choice).lat_rho.values
lon = temp_avg_100m_SO_allyrs.isel(eta_rho=eta_choice, xi_rho=xi_choice).lon_rho.values

# ------------------ Time Series daily ------------------
# === ROMS time series ===
temp_roms = temp_stacked.isel(eta_rho=eta_choice, xi_rho=xi_choice)
y_obs_daily = temp_roms.avg_temp.values #shape (14600,)
time = np.arange(len(y_obs_daily))

# === Trend coefficients ===
slope = trends_100mavg_ds.slope.isel(eta_rho=eta_choice, xi_rho=xi_choice).values
intercept = trends_100mavg_ds.intercept.isel(eta_rho=eta_choice, xi_rho=xi_choice).values

# === Reconstruct fitted trend ===
ntime = len(y_obs_daily)
t_years = np.arange(ntime) / 365.0
y_fit = slope * t_years + intercept

# === Detrended signal ===
y_detrended = y_obs_daily - y_fit + y_obs_daily[0]

# Linear fit to detrended signal
t_years = np.arange(len(y_detrended)) / 365.0
coeffs = np.polyfit(t_years, y_detrended, 1)
slope_detrended = coeffs[0]
coeffs_obs = np.polyfit(t_years, temp_roms.avg_temp, 1)
slope_obs = coeffs_obs[0]
print("ROMS slope (°C/dec):", slope_obs*10)
print("Detrended slope (°C/dec):", slope_detrended * 10)

# === Plot ===
plt.figure(figsize=(12,5))
plt.plot(time, y_obs_daily, "-", label="ROMS", alpha=0.7, color="#0A9396")
plt.plot(time, y_fit, "r--", label="Linear trend", linewidth=2)
plt.plot(time, y_detrended, "#FF8800", label="Detrended", linewidth=2)
plt.xlabel("Days", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)
plt.title(f"Detrending ROMS Temperature signal\nLoc: ({-lat:.2f}°S, {lon:.2f}°E)", fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()


# ------------------ Time Series yearly ------------------
# === Convert ROMS to yearly averages ===
years = np.arange(ntime) // 365  # integer year index
y_obs_yearly = np.array([temp_roms.avg_temp.values[years == yr].mean() for yr in np.unique(years)])
t_years = np.unique(years)

# === Recompute coeffs: yearly slope and intercept
coeffs_yearly = np.polyfit(t_years, y_obs_yearly, 1)
slope_yearly = coeffs_yearly[0]
intercept_yearly = coeffs_yearly[1]

# === Reconstruct trend
y_fit_yearly = slope_yearly * t_years + intercept_yearly

# === Detrend
y_detrended = y_obs_yearly - y_fit_yearly + y_obs_yearly[0]

slope_detrended = np.polyfit(t_years, y_detrended, 1)[0]
print("Detrended slope (°C/dec):", slope_detrended*10)

# === Plot ===
calendar_years = t_years + 1980 
plt.figure(figsize=(12,6))
plt.plot(calendar_years, y_obs_yearly, "-", label="ROMS", alpha=0.7, color="#0A9396")
plt.plot(calendar_years, y_fit_yearly, "r--", label="Linear trend", linewidth=2)
plt.plot(calendar_years, y_detrended, "#FF8800", label="Detrended", linewidth=2)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)
plt.title(f"Detrending ROMS Temperature signal\nLoc: ({-lat:.2f}°S, {lon:.2f}°E)", fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(1980-1, 2019+1)
plt.tight_layout()
plt.show()



# %%tes

# -----------------------------
# Time axis
# -----------------------------
nyears = 40
ndays = nyears * 365
t_days = np.arange(ndays)
t_years = t_days / 365.0

# -----------------------------
# Components
# -----------------------------

# Mean temperature
T0 = 1.0  # °C

# Linear trend
trend = -0.1  # °C / year  (~0.1 °C / decade)
T_trend = trend * t_years

# Seasonal cycle
season_amp = 1.5  # °C
season = season_amp * np.sin(2 * np.pi * t_days / 365.0)

# Interannual / weather noise
np.random.seed(42)
noise = 0.3 * np.random.randn(ndays)

# Total
T = T0 + T_trend + season + noise

# detrend
coeffs = np.polyfit(t_years, T, 1)
print("Recovered trend (°C/yr):", coeffs[0])
print("Expected trend (°C/yr):", trend)
T_fit = coeffs[0] * t_years + coeffs[1]
T_detrended = T - T_fit + T_fit[0]

plt.figure(figsize=(12, 4))
plt.plot(t_years, T, color="0.6", linewidth=0.5, label="Daily temperature")
plt.plot(t_years, T0 + T_trend, color="red", linewidth=2, label="Linear trend")
plt.plot(t_years, T_detrended, color="green", linewidth=0.5, label="Linear trend")

plt.xlabel("Time (years)")
plt.ylabel("Temperature (°C)")
plt.title("Synthetic temperature time series with seasonality and warming")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %%

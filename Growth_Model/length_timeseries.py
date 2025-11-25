"""
Created on Mon 17 Nov 11:03:30 2025

Length timeseries for each cell in the Southern Ocean

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

# --- Biomass
biomass_data = xr.open_dataset('/net/meso/work/aschickele/CEPHALOPOD/output/Marguerite_krill_SO_2025-11-21 16:44:03.790757/236217/euphausia_biomass.nc') #shape (bootstrap=10, northing=64800, time=12)

# %% ====================== Reformatting biomass ======================
nlat, nlon = 180, 360
lat = np.linspace(-89.5, 89.5, nlat)
lon = np.linspace(-179.5, 179.5, nlon)

# Raw data
arr = biomass_data.euphausia_biomass.values  # shape: (bootstrap=10, northing=64800, time=12)

# Transpose to (months, bootstrap, northing)
arr = arr.transpose(2, 0, 1)  # (12, 10, 64800)

# Reshape northing into (lat, lon)
arr = arr.reshape(12, 10, 180, 360)  # (months, bootstrap, lat, lon)

# Flip latitude (before it was going from -90 to 90, i.e. from south pole to north pole)
arr = arr[:, :, ::-1, :]  # flip along lat axis
lat_flipped = lat[::-1]

# Create new Dataset
biomass_cephalopod = xr.Dataset(
    data_vars=dict(euphausia_biomass=(["months", "bootstrap", "lat", "lon"], arr)), 
    coords=dict(months=np.arange(1, 13),
                bootstrap=np.arange(1, 11),
                lat=lat_flipped, lon=lon),
    attrs=biomass_data.attrs)

# Add info in attributed
biomass_cephalopod.attrs.update({
    "model_name": "Cephalopod",
    "model_resolution": "1 degree",
    "model_extent": "global",
    "model_inputs": "WOA",
    "ensemble_member_selection": "RF and SVM",
    "note": "Under assumption that krill spend most of their time in the 0-100m, all observations are integrated (median concentration on depth)."
})

# %% Plot test
ds_1d = biomass_cephalopod.euphausia_biomass.isel(months=0, bootstrap=0)

plt.figure(figsize=(12,6))
plt.pcolormesh(lon, lat, ds_1d, cmap='coolwarm', shading='auto')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Euphausia superba - Month 1, Bootstrap 1')
plt.colorbar(label='Krill density / abundance')
plt.show()
# %% ====================== Calculating length ======================
from Growth_Model.growth_model import length_Atkison2006  
# == Climatological Drivers -> Mean Chla and T°C  (days, eta, xi) period: 1980-2009
temp_clim = temp_avg_100m_SO_allyrs.isel(years=slice(0,30)) #shape: (30, 181, 231, 360)
temp_clim_mean = temp_clim.mean(dim=['years']) #shape: (181, 231, 360)
chla_clim = chla_surf_SO_allyrs.isel(years=slice(0,30))
chla_clim_mean = chla_clim.mean(dim=['years'])

# == Climatological Length 
climatological_length = length_Atkison2006(chla=chla_clim_mean.chla, temp=temp_clim_mean.avg_temp, initial_length=35, intermoult_period=10) #shape:(181, 231, 1442)

# # %% Surface Temperature
# path_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
# temp_surf_eta = xr.open_dataset(os.path.join(path_temp, 'temp_DC_BC_eta220.nc')).isel(year=slice(1,40), z_rho=0) #shape: (39, 365, 1442)
# chla_surf_SO_allyrs_eta = chla_surf_SO_allyrs.isel(eta_rho=220) #shape: (39, 181, 1442)

# # Select only austral summer and early spring
# jan_april = temp_surf_eta.sel(day=slice(0, 119)) # 1 Jan to 30 April (Day 0-119) - last idx excluded
# jan_april.coords['day'] = jan_april.coords['day'] #keep info on day

# nov_dec = temp_surf_eta.sel(day=slice(304, 365)) # 1 Nov to 31 Dec (Day 304–364) - last idx excluded
# nov_dec.coords['day'] = np.arange(304, 365) #keep info on day

# temp_surf_eta_austral = xr.concat([nov_dec, jan_april], dim="day") #shape: (39, 181, 1442)


# # == Climatological Drivers -> Mean Chla and T°C  (days, eta, xi) period: 1980-2009
# temp_clim_eta = temp_surf_eta_austral.isel(year=slice(0,30)) #shape: (30, 181, 231, 360)
# temp_clim_eta_mean = temp_clim_eta.mean(dim=['year']) #shape: (181, 231, 360)
# chla_clim_eta = chla_surf_SO_allyrs_eta.isel(years=slice(0,30))
# chla_clim_eta_mean = chla_clim_eta.mean(dim=['years'])


# # === Select year of interest
# year_idx = 26
# temp_surf_eta_yr = temp_surf_eta_austral.isel(year=year_idx)
# chla_surf_eta_yr = chla_surf_SO_allyrs_eta.isel(years=year_idx)


# # === Detrend the signal by removing the climatological signal
# temp_surf_eta_yr_med = temp_surf_eta_yr.temp.median(dim='day')
# chla_surf_eta_yr_med = chla_surf_eta_yr.chla.median(dim='days')

# temp_detr = temp_surf_eta_yr.temp - temp_clim_eta_mean
# chla_detr = chla_surf_eta_yr.chla - chla_clim_eta_mean

# # === Correlation
# # --- Select 1 pixel along xi ---
# xi_sel = 1100

# temp_ts = temp_detr.temp.isel(xi_rho=xi_sel)   # shape: (181)
# chla_ts = chla_detr.chla.isel(xi_rho=xi_sel)   # shape: (181)
# temp_ts = temp_ts.rename({"day": "days"})

# corr_value = float(xr.corr(temp_ts, chla_ts, dim="days"))
# print("Correlation at xi=1000:", corr_value)


# # --- Plot ---
# days_xaxis = np.arange(181)
# base_date = datetime(2021, 11, 1)
# date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
# date_dict = dict(date_list)
# tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15) #ticks every 15days
# tick_labels = [date_dict.get(day, '') for day in tick_positions]

# plt.figure(figsize=(12,5))

# plt.plot(days_xaxis, temp_ts, label="Detrended Temp", linewidth=2)
# plt.plot(days_xaxis, chla_ts, label="Detrended Chla", linewidth=2)

# plt.axhline(0, color='gray', linestyle='--', linewidth=1)

# plt.xlabel("Day of austral season")
# plt.ylabel("Detrended anomalies")
# plt.title(f"Detrended Temperature and Chla Time Series at xi=1000\nCorrelation = {corr_value:.2f}")

# plt.legend()
# plt.tight_layout()
# plt.show()


# %% ====================== Correlation ======================
# === Select year of interest
year_idx = 26
temp_100m_yr = temp_avg_100m_SO_allyrs.isel(years=year_idx)
chla_suf_yr = chla_surf_SO_allyrs.isel(years=year_idx)

# === Detrend the signal by removing the climatological signal
temp_100m_yr_med = temp_100m_yr.avg_temp.median(dim='days')
chla_suf_yr_med = chla_suf_yr.chla.median(dim='days')

temp_100m_yr_detrended = temp_100m_yr - temp_clim_mean
chla_suf_yr_detrended = chla_suf_yr - chla_clim_mean

# == Lag response of Chla
lags = np.arange(-30, 31)  # days
# lagged_corrs = []
# for lag in lags:
#     lag=30
#     temp_shifted = temp_100m_yr_detrended.avg_temp.shift(days=lag)
#     corr_lag = xr.corr(temp_shifted, chla_suf_yr_detrended.chla, dim='days')
#     lagged_corrs.append(corr_lag)
#  === Person correlation of detrended seasonal signal
# corr = xr.corr(temp_100m_yr_detrended.avg_temp, chla_suf_yr_detrended.chla, dim='days') #shape (231, 1442)

def compute_corr_for_lag(lag):
    # shift inside function
    temp_shifted = temp_100m_yr_detrended.avg_temp.shift(days=lag)
    corr_lag = xr.corr(temp_shifted, chla_suf_yr_detrended.chla, dim='days')
    return corr_lag

lagged_corrs_list = process_map(compute_corr_for_lag, lags, max_workers=8, chunksize=1)

# Reformatting to DataArray
lagged_corrs = xr.concat(lagged_corrs_list, dim='lag')
lagged_corrs = lagged_corrs.assign_coords(lag=lags) #shape (lag:61, 231, 1442)

# Reformatting to Numpy Array
lagged_corrs_np = lagged_corrs.values  # shape: (lag, eta, xi)
lags_np = lagged_corrs.lag.values

# Mask fully-NaN pixels
all_nan_mask = np.all(np.isnan(lagged_corrs_np), axis=0)

# Fill NaNs with very negative value for safe argmax
lagged_corrs_fill = np.where(np.isnan(lagged_corrs_np), -9999, lagged_corrs_np)

# === Maximum correlation
max_corr_np = np.nanmax(lagged_corrs_np, axis=0)
max_corr_np[all_nan_mask] = np.nan
max_corr = xr.DataArray(
    max_corr_np,
    coords={'lat_rho': temp_100m_yr_med.lat_rho,
            'lon_rho': temp_100m_yr_med.lon_rho},
    dims=['eta_rho', 'xi_rho']
)

# Best lag value at each location
argmax_idx = np.argmax(lagged_corrs_fill, axis=0)
best_lag_np = lags_np[argmax_idx].astype(float)
best_lag_np[all_nan_mask] = np.nan
best_lag = xr.DataArray(
    best_lag_np,
    coords={'lat_rho': temp_100m_yr_med.lat_rho,
            'lon_rho': temp_100m_yr_med.lon_rho},
    dims=['eta_rho', 'xi_rho']
)
best_lag_masked = best_lag.where(max_corr > 0.2) # Correlation threshold to reduce noise

# %% =============== Plot Correlation and best lag ===============
fig_width = 20
fig_height = 6
title_kwargs = {'fontsize': 14}

fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=1, ncols=2)

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5]
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# --- Panel 1: Max correlation
ax1 = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
vmin, vmax = -1, 1
im1 = ax1.pcolormesh(max_corr.lon_rho, max_corr.lat_rho, max_corr,
                     transform=ccrs.PlateCarree(),
                     cmap='coolwarm', shading='auto', rasterized=True,
                     norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
cbar_ax1 = fig.add_axes([0.45, 0.15, 0.007, 0.7])
cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', extend='both')
cbar1.set_label("Max Corr (T – Chla)", fontsize=12)
ax1.set_title("Maximum correlation between Temperature and Chla", **title_kwargs)

# --- Panel 2: Lag at max correlation
ax2 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
vmin, vmax = -30, 30  # lag in days
im2 = ax2.pcolormesh(best_lag_masked.lon_rho, best_lag_masked.lat_rho, best_lag_masked,
                     transform=ccrs.PlateCarree(),
                     cmap='Spectral_r', shading='auto', rasterized=True,
                     vmin=vmin, vmax=vmax)
cbar_ax2 = fig.add_axes([0.87, 0.15, 0.007, 0.7])
cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='vertical', extend='both')
cbar2.set_label("Lag (days)", fontsize=12)
ax2.set_title("Lag at maximum correlation (Chla response to Temp)", **title_kwargs)

# --- Common map features
for ax in [ax1, ax2]:
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
    ax.coastlines(color='black', linewidth=1)
    ax.set_facecolor('#F6F6F3')
    # Sector boundaries
    for lon in [-90, 0, 120]:
        ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(),
                color='#080808', linestyle='--', linewidth=0.5)

plt.show()




# %% ====================== Length timeseries seasons ======================
output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/krill_length/'
output_file_length = os.path.join(output_path, "length_35mm_yearly.nc")
output_file_diff = os.path.join(output_path, "diff_toclim_length_35mm_yearly.nc")

# Save to file
if not os.path.exists(output_file_length) and os.path.exists(output_file_diff):
    try:
        # Initialization
        nyears, ndays, neta, nxi = chla_surf_SO_allyrs.chla.shape
        length_yearly = np.zeros((nyears, ndays, neta, nxi), dtype=np.float64)
        diff_length_yearly = np.zeros((nyears, ndays, neta, nxi), dtype=np.float64)

        for year_idx in range(39):
            print(f'Processing year: {year_idx+1980}')
            # == Calculating length
            length_yearly[year_idx] = length_Atkison2006(chla=chla_surf_SO_allyrs.chla.isel(years=year_idx),
                                    temp=temp_avg_100m_SO_allyrs.avg_temp.isel(years=year_idx),
                                    initial_length=35, intermoult_period=10) #shape: (181, 231, 1442)
            
            # == Difference in length (seasonal-clim)
            diff_length_yearly[year_idx]= length_yearly[year_idx] - climatological_length #shape (181, 231, 1442)

        # === To dataset
        length_yearly_ds = xr.Dataset({
            "length": (("years", "days", "eta_rho", "xi_rho"), length_yearly),
            }, coords={
            "lon_rho":(["eta_rho", "xi_rho"], chla_surf_SO_allyrs.lon_rho.values),  # (231, 1442)
            "lat_rho":(["eta_rho", "xi_rho"], chla_surf_SO_allyrs.lat_rho.values),  # (231, 1442)
            })
        
        diff_length_yearly_ds = xr.Dataset({
            "diff_length": (("years", "days", "eta_rho", "xi_rho"), diff_length_yearly),
            }, coords={
            "lon_rho":(["eta_rho", "xi_rho"], chla_surf_SO_allyrs.lon_rho.values),  # (231, 1442)
            "lat_rho":(["eta_rho", "xi_rho"], chla_surf_SO_allyrs.lat_rho.values),  # (231, 1442)
            })

        length_yearly_ds.to_netcdf(output_file_length, engine="netcdf4")
        diff_length_yearly_ds.to_netcdf(output_file_diff, engine="netcdf4")
        print(f"Files written")

    except Exception as e:

        print(f"Error writing: {e}")
else:
    print('Files already exist')



    
# %% ====================== Length 1 year ======================
year_idx=26
length_yr = length_Atkison2006(chla=chla_surf_SO_allyrs.chla.isel(years=year_idx),
                                    temp=temp_avg_100m_SO_allyrs.avg_temp.isel(years=year_idx),
                                    initial_length=35, intermoult_period=10) #shape: (181, 231, 1442)
        
diff_length_yr= length_yr - climatological_length #shape (181, 231, 1442)

# %% ====================== Partial Derivatives ======================
year_idx=26
# === 4 cases ===
# 1. Real world (actual temperature and chla concentrations)
L_actual = length_Atkison2006(chla=chla_surf_SO_allyrs.chla.isel(years=year_idx),
                                    temp=temp_avg_100m_SO_allyrs.avg_temp.isel(years=year_idx),
                                    initial_length=35, intermoult_period=10) #shape: (181, 231, 1442)

# 2. Climatological world (climatological temperature and chla concentrations)
L_baseline = length_Atkison2006(chla=chla_clim_mean.chla, temp=temp_clim_mean.avg_temp, initial_length=35, intermoult_period=10) #shape:(181, 231, 1442)

# 3. Temperature-only world (actual temperature, climatological chla concentrations)
L_temp = length_Atkison2006(chla=chla_clim_mean.chla,
                            temp=temp_avg_100m_SO_allyrs.avg_temp.isel(years=year_idx),
                            initial_length=35, intermoult_period=10) #shape: (181, 231, 1442)

# 4. Chla-only world (climatological temperature, actual chla concentrations)
L_chla = length_Atkison2006(chla=chla_surf_SO_allyrs.chla.isel(years=year_idx),
                            temp=temp_clim_mean.avg_temp,
                            initial_length=35, intermoult_period=10) #shape: (181, 231, 1442)

# === Total changes relative to climatology
DeltaL_total = L_actual - L_baseline 

# === Contribution
DeltaL_temp = L_temp - L_baseline # temperature-only effect
DeltaL_chla = L_chla - L_baseline #chla-only effect 

# === Interaction between Chla and Temp 
# DeltaL_interaction = L_actual - L_temp - L_chla + L_baseline
delta_L_residual = DeltaL_total - (DeltaL_temp + DeltaL_chla) # Residual (interaction term)

# === Contribution factors [%]
T_contrib = DeltaL_temp/DeltaL_total * 100 #shape (181, 231, 1442)
Chla_contrib = DeltaL_chla/DeltaL_total * 100 #shape (181, 231, 1442)
resitual_contrib = delta_L_residual/DeltaL_total * 100 

# %% ====================== Plot fraction maps ======================
# Dates to plot
plot_days = [30, 61, 92, 120, 150]
plot_labels = ["1 Dec", "1 Jan", "1 Feb", "1 Mar", "1 Apr"]

# Circular boundary for polar plot
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5]
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 18), subplot_kw={'projection': ccrs.SouthPolarStereo()})

# Color settings
vmin, vmax = -100, 100  
cmap = "RdBu_r"

for i, day_idx in enumerate(plot_days):

    # Extract data for that day
    temp_slice = T_contrib.isel(days=day_idx)
    chla_slice = Chla_contrib.isel(days=day_idx)
    res_slice = resitual_contrib.isel(days=day_idx)

    # --- Temperature Contribution Map (col 0) ---
    ax = axes[i, 0]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(color="black", linewidth=0.5)
    im1 = ax.pcolormesh(
        temp_slice.lon_rho, temp_slice.lat_rho, temp_slice,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    ax.set_title(f"{plot_labels[i]} – Temp Contribution (\%)")

    # --- Chla Contribution Map (col 1) ---
    ax = axes[i, 1]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(color="black", linewidth=0.5)
    im2 = ax.pcolormesh(
        chla_slice.lon_rho, chla_slice.lat_rho, chla_slice,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    ax.set_title(f"{plot_labels[i]} – Chla Contribution (\%)")

    # --- Residuals ---
    ax = axes[i, 2]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(color="black", linewidth=0.5)
    im3 = ax.pcolormesh(
        res_slice.lon_rho, res_slice.lat_rho, res_slice,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    ax.set_title(f"{plot_labels[i]} – Residuals Contribution (\%)")

# Add colorbar under the entire figure
cbar = fig.colorbar(im1, ax=axes, orientation="horizontal", pad=0.03, shrink=0.8)
cbar.set_label("Contribution to DeltaL (%)")

plt.tight_layout()
plt.show()


# %% ====================== Plot fraction map ======================
# === Plot settings ===
fig_width = 20
fig_height = 6
title_kwargs = {'fontsize': 14}

# Circular boundary for polar plot
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5]
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Setup figure with 3 panels
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=1, ncols=3)

fractions = [frac_temp, frac_chla, frac_residuals]
titles = ["Fraction explained by Temperature", "Fraction explained by CHLA", "Residual / Interaction"]
cmaps = ['Reds', 'Greens', 'Purples']
vmin, vmax = 0, 1  # fractions between 0 and 1

for i, (frac, title, cmap) in enumerate(zip(fractions, titles, cmaps)):
    ax = fig.add_subplot(gs[i], projection=ccrs.SouthPolarStereo())
    im = ax.pcolormesh(frac.lon_rho, frac.lat_rho, frac,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap,
                       norm=Normalize(vmin=vmin, vmax=vmax),
                       shading='auto', rasterized=True)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.32 + i*0.18, 0.15, 0.007, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', extend='both')
    cbar.set_label("Variance fraction", fontsize=12)
    
    # Map features
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
    ax.coastlines(color='black', linewidth=1)
    ax.set_facecolor('#F6F6F3')
    for lon in [-90, 0, 120]:
        ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(),
                color='#080808', linestyle='--', linewidth=0.5)
    
    ax.set_title(title, **title_kwargs)

plt.tight_layout()
plt.show()


# %% Analysis
eta_idx=200
xi_idx=100

# Select 1 cell
DeltaL_total_cell = DeltaL_total.isel(eta_rho = eta_idx, xi_rho = xi_idx)
DeltaL_temp_cell = DeltaL_temp.isel(eta_rho = eta_idx, xi_rho = xi_idx)
DeltaL_chla_cell = DeltaL_chla.isel(eta_rho = eta_idx, xi_rho = xi_idx)
DeltaL_inter_cell = DeltaL_interaction.isel(eta_rho = eta_idx, xi_rho = xi_idx)

# Relative contribution of the different term to the changes in length relative to climaotlofy
frac_interaction_mean = (np.abs(DeltaL_inter_cell) / np.abs(DeltaL_total_cell)).mean()
print(frac_interaction_mean.values)
fraction_temp = 

# Plot
# === Prepare time axis ===
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15) #ticks every 15days
tick_labels = [date_dict.get(day, '') for day in tick_positions]

plt.figure(figsize=(10,5))
plt.plot(days_xaxis, DeltaL_temp_cell, label='DeltaL Temperature', color='red')
plt.plot(days_xaxis, DeltaL_chla_cell, label='DeltaL CHLA', color='green')
plt.plot(days_xaxis, DeltaL_inter_cell, label='Interaction', color='purple')
plt.plot(days_xaxis, DeltaL_total_cell, '--', label='Total DeltaL', color='black')
plt.xlabel('Day of Year')
plt.ylabel('Krill length anomaly (mm)')
plt.title(f'Krill length anomaly at cell (eta={eta_idx}, xi={xi_idx})')
plt.legend()
plt.grid(True)
plt.show()

# %% ====================== Plot timeseries for 1 cell ======================
eta_idx=220
xi_idx=1000

# === Layout config ===
plot = 'slides'  # or 'slides'

if plot == 'report':
    fig_width = 6.3228348611  # narrower figure width
    fig_height = fig_width * 1.5  # keep proportional height
else:
    fig_width = 15
    fig_height = 3

# === Font and style settings ===
title_kwargs = {'fontsize': 14} if plot == 'slides' else {}
label_kwargs = {'fontsize': 13} if plot == 'slides' else {}
annotation_kwargs = {'fontsize': 10} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 11} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 15, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}
gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
lw = 0.7 if plot == 'slides' else 0.4

# === Prepare time axis ===
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15) #ticks every 15days
tick_labels = [date_dict.get(day, '') for day in tick_positions]

# === Setup figure ===
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[0.7, 0.3])

# === Krill Length ===
ax0 = fig.add_subplot(gs[0])

# Plot data
ax0.plot(days_xaxis, length_yr.isel(eta_rho=eta_idx, xi_rho=xi_idx), color='#A22C29', linewidth=lw, label='Krill Length')
ax0.plot(days_xaxis, climatological_length.isel(eta_rho=eta_idx, xi_rho=xi_idx), color='#542E71', linewidth=lw, linestyle='--', label='Climatology')

# Adding the difference
diff_val = diff_length_yr.isel(days=-1, eta_rho=eta_idx, xi_rho=xi_idx) #value of the difference on 30th April
y_start = float(length_yr.isel(days=-1, eta_rho=eta_idx, xi_rho=xi_idx))
y_end = float(climatological_length.isel(days=-1, eta_rho=eta_idx, xi_rho=xi_idx))
ax0.annotate('', 
             xy=(181, y_start), xytext=(181, y_end), #the 2 ends of the arrow
             arrowprops=dict(arrowstyle='<->', lw=1.2, color='black'))
ax0.text(181 + 2, # horizontal position of the text
         (y_start + y_end) / 2, # vertical position of the text
         f"{diff_val:.3f} mm", # value
         va='center', ha='left', color='black', **annotation_kwargs)

# Settings x-axis 
ax0.set_xlabel('Days', **label_kwargs)
ax0.set_xlim(0,200)
ax0.tick_params(axis='x', labelbottom=True, length=1, width=0.5)
ax0.set_xticks(tick_positions)
ax0.set_xticklabels(tick_labels, rotation=45, ha='right')

# Settings y-axis 
ax0.set_ylabel("Length [mm]", **label_kwargs)
ytick_start = int(np.ceil(length_yr.isel(eta_rho=eta_idx, xi_rho=xi_idx).max().item()))
ytick_end = int(np.floor(length_yr.isel(eta_rho=eta_idx, xi_rho=xi_idx).min().item())) - 1
ax0.set_yticks(np.arange(ytick_start, ytick_end, -1))
ax0.tick_params(axis='y', length=2, width=0.5)

ax0.tick_params(**{k: v for k, v in tick_kwargs.items() if k not in ['length', 'width']})
ax0.legend(loc='upper left', frameon=True, bbox_to_anchor=(0.83, 0.25), fontsize=9)

# === Plot location ===
ax1 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Color Settings
colors = ["#D1105A", "#F24B04", "#FFFFFF", "#884AB2", "#471CA8"]  
# colors = ["#FF9E1F", "#FF930A","#F24B04", "#D1105A", "#AD2D86", "#471CA8", "#371F6F", "#2A1E48"]
cmap = LinearSegmentedColormap.from_list("biomass", colors, N=256)
cmap='coolwarm'

# Plot data
im = ax1.pcolormesh(diff_length_yr.isel(days=-1).lon_rho, diff_length_yr.isel(days=-1).lat_rho, diff_length_yr.isel(days=-1).data,
                    transform=ccrs.PlateCarree(), cmap=cmap, shading='auto', rasterized=True, zorder=1)

ax1.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
ax1.set_boundary(circle, transform=ax1.transAxes)

# Map extent and features
ax1.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
ax1.coastlines(color='black', linewidth=1)
ax1.set_facecolor('#F6F6F3')
    
# Sector boundaries
for lon in [-90, 120, 0]:
        ax1.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=0.5)

# Gridlines
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter
gl = ax1.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
gl.xformatter = LongitudeFormatter(degree_symbol='°', number_format='.0f', dateline_direction_label=False)
gl.yformatter = LatitudeFormatter()
gl.xlabels_top = False
gl.xlabels_bottom = False
gl.ylabels_right = False
gl.xlabels_left = True
gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs

# Mark at location
marker_kwargs = {'markersize': 10} if plot == 'slides' else {'markersize': 5}
ax1.plot(diff_length_yr.isel(eta_rho=eta_idx, xi_rho=xi_idx).lon_rho, 
            diff_length_yr.isel(eta_rho=eta_idx, xi_rho=xi_idx).lat_rho, 
            marker='*', color='black', **marker_kwargs, transform=ccrs.PlateCarree(), zorder=6)

# === Titles ===
if plot == 'slides':
    ax0.set_title(f"Time Series of Krill Length \n Growth season {year_idx+1980}-{year_idx+1980+1}", **title_kwargs)
    ax1.set_title(f"Spatial Distribution of Length Difference \n 30th April {year_idx+1980+1}", **title_kwargs)


# === Output handling ===
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/paper/length')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"length_diff_1cell{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/temp_chla_diagrams/chla_ts_loc_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %% ====================== Plot difference map ======================
# === Plot location ===
ax_map = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
colors = ["#D1105A", "#F24B04", "#FFFFFF", "#884AB2", "#471CA8"]  
# colors = ["#FF9E1F", "#FF930A","#F24B04", "#D1105A", "#AD2D86", "#471CA8", "#371F6F", "#2A1E48"]
cmap = LinearSegmentedColormap.from_list("biomass", colors, N=256)
cmap='coolwarm'
# Plot data
im = ax_map.pcolormesh(diff_length.isel(days=-1).lon_rho, 
                       diff_length.isel(days=-1).lat_rho, 
                       diff_length.isel(days=-1).data,
                    transform=ccrs.PlateCarree(), cmap=cmap, 
                    shading='auto', rasterized=True, zorder=1)

ax_map.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
ax_map.set_boundary(circle, transform=ax_map.transAxes)

# Map extent and features
ax_map.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
ax_map.coastlines(color='black', linewidth=1)
ax_map.set_facecolor('#F6F6F3')
    
# Sector boundaries
for lon in [-90, 120, 0]:
        ax_map.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=0.5)

# Gridlines
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter
gl = ax_map.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
gl.xformatter = LongitudeFormatter(degree_symbol='°', number_format='.0f', dateline_direction_label=False)
gl.yformatter = LatitudeFormatter()
gl.xlabels_top = False
gl.xlabels_bottom = False
gl.ylabels_right = False
gl.xlabels_left = True
gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs

# Star at location
marker_kwargs = {'markersize': 10} if plot == 'slides' else {'markersize': 5}
ax_map.plot(diff_length.isel(eta_rho=eta_idx, xi_rho=xi_idx).lon_rho, 
            diff_length.isel(eta_rho=eta_idx, xi_rho=xi_idx).lat_rho, 
            marker='*', color='black', **marker_kwargs, transform=ccrs.PlateCarree(), zorder=6)

# == Title
if plot == 'slides':
    ax3.set_title(f"Time Series of Krill Length \n Growth season {year_idx+1980}-{year_idx+1980+1}", **title_kwargs)
    ax_map.set_title(f"Spatial Distribution of Length Difference \n 30th April {year_idx+1980+1}", **title_kwargs)


# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/paper/length')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"length_diff_1cell{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/temp_chla_diagrams/chla_ts_loc_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

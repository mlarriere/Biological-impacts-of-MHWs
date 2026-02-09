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

#%% -------------------------------- Server --------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% -------------------------------- Figure settings --------------------------------
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
# %% -------------------------------- Settings --------------------------------
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
path_surrogates = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/surrogates'
path_mass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/krill_mass'
path_length = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/krill_length'


# %% ======================== Load data ========================
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 365, 231, 1442)

# Concatenate the years together 
temp_stacked = temp_avg_100m_SO_allyrs.stack(time=("year", "days")) #shape (14600, 231, 1442)
temp_stacked = temp_stacked.transpose("time", "eta_rho", "xi_rho")

# %% ======================== Functions ========================
def compute_trend_cell_vectorized(ieta):
    """
    Compute linear trend for one eta (all xi).
    Returns slopes, intercepts, and r2 arrays: shape (time, xi)
    """
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
    t = np.arange(ntime) / 365.0   # from days to years
    X = np.column_stack((t, np.ones_like(t))) #shape (time, 2) 

    # Extract Y
    Y = temp_data.isel(eta_rho=ieta).avg_temp.values  # (time, xi)
    
    # Solve linear regression
    coeffs, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # shape (2, xi)

    # Store coeffs of interest
    slopes= coeffs[0] #shape: (nxi) - [°C/yr]
    intercepts = coeffs[1] #shape: (nxi)
        
    # --- Evaluate the regression
    # Compute R2 for each xi
    y_mean = np.mean(Y, axis=0)
    ss_tot = np.sum((Y - y_mean)**2, axis=0)

    # Compute residuals
    Y_pred = X @ coeffs
    ss_res = np.sum((Y - Y_pred)**2, axis=0)  
    r2 = np.where(ss_tot != 0, 1 - ss_res / ss_tot, np.nan)

    return ieta, slopes, intercepts, r2

# Select growth season for each year
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

# %% ======================== Compute the trend ========================
from tqdm.contrib.concurrent import process_map
output_file_100mavg = os.path.join(path_surrogates, 'detrended_signal/temp_linear_trend_100mavg.nc')

if not os.path.exists(output_file_100mavg):
    neta = temp_avg_100m_SO_allyrs.sizes['eta_rho']
    nxi  = temp_avg_100m_SO_allyrs.sizes['xi_rho']

    # --- Call function in parallel
    ieta_list = list(range(neta))
    results = process_map(compute_trend_cell_vectorized, ieta_list, max_workers=8, desc="Processing eta") 

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
# im = ax.pcolormesh(trends_100mavg_ds.lon_rho, trends_100mavg_ds.lat_rho, 
#                    trends_100mavg_ds.slope, cmap='coolwarm', transform=ccrs.PlateCarree(), zorder=1)
im = ax.pcolormesh(trends_100mavg_ds.lon_rho, trends_100mavg_ds.lat_rho, 
                   trends_100mavg_ds.slope*40, cmap='coolwarm', 
                   vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), zorder=1)
# im = ax.pcolormesh(trends_100mavg_ds.lon_rho, trends_100mavg_ds.lat_rho, trends_100mavg_ds.r2, cmap='Blues', vmin=0, vmax=0.25, transform=ccrs.PlateCarree(), zorder=1)

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

# %% ======================== Removing trend ========================
detrended_temp_seasons_file = os.path.join(path_surrogates, f'detrended_signal/temp_detrended_seasonal.nc')
if not os.path.exists(detrended_temp_seasons_file):
    # Preparing (linear trend fitted on daily continuous data over 40 years -> need to reconstruct time to have trend)
    n_years = temp_avg_100m_SO_allyrs.sizes["year"]
    n_days_per_year = temp_avg_100m_SO_allyrs.sizes["days"]
    total_days = n_years * n_days_per_year
    t_cont = np.arange(total_days) / 365.0  # fractional years
    t_da = xr.DataArray(t_cont, dims=["time"], coords={"time": np.arange(total_days)})

    # -- Reconstruct the linear fit using coeffs from linear regression 
    intercept = trends_100mavg_ds.intercept
    slope = trends_100mavg_ds.slope
    slope_exp = slope.expand_dims({"time": total_days}).transpose("time", "eta_rho", "xi_rho") 
    intercept_exp = intercept.expand_dims({"time": total_days}).transpose("time", "eta_rho", "xi_rho")

    # Linear trend
    y_fit_daily = slope_exp * t_da + intercept_exp #shape (14600, 231, 1442)
    y_fit_da = xr.DataArray(y_fit_daily, dims=temp_stacked.avg_temp.dims, coords=temp_stacked.avg_temp.coords)

    # -- Detrend signal = ROMS signal - linear trend
    roms_detrended_yearly = temp_stacked.avg_temp - y_fit_da #shape (14600, 231, 1442)

    # -- Unstack time (put back to (years, days) - not continuous anymore)
    detrended_unstacked = roms_detrended_yearly.unstack("time").transpose("year", "days", "eta_rho", "xi_rho") #shape(40, 365, 231, 1442)

    # -- Select only growth season
    detrended_unstacked = detrended_unstacked.rename({'year':'years'})
    temp_detrended_seasons = define_season_all_years_parallel(detrended_unstacked, max_workers=6)
    
    # -- To dataset
    temp_detrended_seasons = temp_detrended_seasons.rename({'season_year': 'season_year_temp'})
    temp_detrended_seasons = temp_detrended_seasons.drop_vars('years')
    temp_detrended_seasons = temp_detrended_seasons.rename({'season_year_temp': 'years'})
    temp_detrended_seasons.attrs['description'] = ("Daily ROMS temperature with linear trend removed at each grid cell. "
                                                    "Trend estimated using OLS (over daily signal of 40years).\n"
                                                    "Temporal extent: Growth season (Nov 1 – Apr 30)")
    temp_detrended_seasons_ds = temp_detrended_seasons.to_dataset(name="avg_temp")
    
    # -- Save to file
    temp_detrended_seasons_ds.to_netcdf(detrended_temp_seasons_file)
else:
    temp_detrended_seasons_ds = xr.open_dataset(detrended_temp_seasons_file) 


# %% ======================== Visualisation ========================
eta_choice = 200
xi_choice = 800 #1100
lat = temp_detrended_seasons_ds.isel(eta_rho=eta_choice, xi_rho=xi_choice).lat_rho.values
lon = temp_detrended_seasons_ds.isel(eta_rho=eta_choice, xi_rho=xi_choice).lon_rho.values

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

# %% ======================== Defining MPAs ========================
# == Load data
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)

# == Fix extent 
# South of 60°S
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

# == Settings plot
mpa_dict = {
    "Ross Sea": (mpas_ds.mask_rs, "#5F0F40"),
    "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#FFBA08"),
    "East Antarctic": (mpas_ds.mask_ea, "#E36414"),
    "Weddell Sea": (mpas_ds.mask_ws, "#4F772D"),
    "Antarctic Peninsula": (mpas_ds.mask_ap, "#0A9396")
}

# %% ======================== Areas and volume MPAs ========================
# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km2
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km3

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)

# --- Calculate area and volume of each MPA
mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}

area_mpa = {}
volume_mpa = {}

for abbrv, (name, mask) in mpa_masks.items():
    area_mpa[abbrv] = area_60S_SO.where(mask)
    volume_mpa[abbrv] = volume_60S_SO_100m.where(mask)

# %% ======================== Warming in MPAs ========================
# Mask MPA
trends_100mavg_mpa = trends_100mavg_ds

# Reformat 
trends_100mavg_ds_reformat = trends_100mavg_ds.isel(xi_rho = slice(0, mpas_south60S.xi_rho.size))

for abbrv, (name, mask_2d) in mpa_masks.items():
    output_file= os.path.join(path_surrogates, f'detrended_signal/mpas/temp_linear_trend_{abbrv}.nc')

    if not os.path.exists(output_file) :
        # Test
        # abbrv='RS'
        # name='Ross Sea'
        # mask_2d = mpas_south60S.mask_rs
        print(f"Masking of {name} ({abbrv})")
    
        mask_mpa = mask_2d.astype(bool)

        # Mask MPA 
        slope_warming_mpa = trends_100mavg_ds_reformat.slope.where(mask_mpa)    
        interc_warming_mpa = trends_100mavg_ds_reformat.intercept.where(mask_mpa)    
        r2_warming_mpa = trends_100mavg_ds_reformat.r2.where(mask_mpa)    
        
        # To Dataset
        warming_mpa_ds = xr.Dataset({"slope": slope_warming_mpa,
                                     "intercept": interc_warming_mpa,
                                     "r2": r2_warming_mpa,})
        
        warming_mpa_ds.attrs["MPA name"] =  f"{name} ({abbrv})"

        warming_mpa_ds["slope"].attrs["Description"] = "Linear temperature trend"
        warming_mpa_ds["slope"].attrs["Units"] = "°C yr⁻¹"

        warming_mpa_ds["intercept"].attrs["Description"] = "Temperature at t=0, i.e. 1st January 1980"
        warming_mpa_ds["intercept"].attrs["Units"] = "°C"

                
        
        # Save to file
        warming_mpa_ds.to_netcdf(output_file) #shape (231, 1442)
    else:
        print(f'MHWs in {name} already saved to file') 

# -- Load data
warming_mpa_ds = {}
for region in mpa_masks.keys():
    warming_mpa_ds[region] = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/interpolated/duration_AND_thresh_{region}.nc'))

# %%
"""
Created on Tues 12 May 11:38:30 2026

Run MHWs on observations 

@author: Marguerite Larriere (mlarriere)
"""

# %% ======================== PACKAGES ========================
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

#%% ======================== Server ======================== 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% ======================== Figure settings ========================
import matplotlib as mpl
print(mpl.get_cachedir())
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
# %% ======================== SETTINGS ========================
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_duration = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/mhw_durations'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'
path_det_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer'
path_combined_thesh= '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/combined_thresholds'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
path_growth = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model'
path_growth_inputs_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs/austral_summer'
path_biomass= '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_biomass_surrogates = os.path.join(path_surrogates, f'biomass_timeseries')



def extract_one_season_pair(args):
    ds_y, ds_y1, y = args
    try:
        days_nov_dec = ds_y.sel(days=slice(304, 365))
        days_jan_apr = ds_y1.sel(days=slice(0, 120))

        # Concatenate days and days_of_yr for new season dimension
        combined_days = np.concatenate([
            days_nov_dec['days'].values,
            days_jan_apr['days'].values
        ])
        combined_doy = np.concatenate([
            days_nov_dec['days_of_yr'].values,
            days_jan_apr['days_of_yr'].values
        ])

        season = xr.concat([days_nov_dec, days_jan_apr],
                           dim=xr.DataArray(combined_days, dims="days", name="days"))
        
        season = season.assign_coords(days_of_yr=("days", combined_doy))

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


# %% ======================== Load Copernicus data ========================
# Datasets: monthly means!
fnd_ostia_file = "/nfs/sea/work/datasets/gridded/ocean/surface/obs/sst/ukmo_ostia_fnd/METOFFICE-GLO-SST-L4-REP-OBS-SST__P1m_25km.zarr" # File from 1st Oct 1981 to 1st Dec 2023
chla_GlobColour_file = "/nfs/sea/work/datasets/gridded/ocean/3d/obs/chl/cmems_bgc_chl/cmems_chl_SO_d025_monthly_1998_to_2022.nc" # From 1st January 1998 to 1st Dec 2022
fn_cmems = '/nfs/meso/work/jwongmeng/ROMS/evaluation/data/cmems_chl2d_SO_d025_monthly_1998_to_2019.nc'

# Load data
ds_ostia = xr.open_zarr(fnd_ostia_file, consolidated=True)
sst_fnd = ds_ostia["analysed_sst"]
chl_cmems = xr.open_dataset(chla_GlobColour_file)["chl"].isel(depth=0) #shape (300, 434, 1442)

#-- test plot
chl_cmems.isel(time=0).plot()

# -- 1. Subset to the region of interest (Southern Ocean)
sst_so = sst_fnd.sel(latitude=slice(-90, -60))
mask=chl_cmems.lat_rho <= -60
chl_cmems_so = chl_cmems.where(mask, drop=True) #shape (300, 231, 1442)

# -- 2. Select temporal extent
sst_so_subset = sst_so.sel(time=slice("1981-11-01", "2019-04-30")) # Select from 1981 to 2018, i.e. from 1st Nov 1981 to 30th April 2019
chl_cmems_so_subset = chl_cmems_so.sel(time=slice("1998-11-01", "2019-04-30"))

# Growth season: Nov–Apr
season_mask_temp = sst_so_subset.time.dt.month.isin([11, 12, 1, 2, 3, 4])
sst_season = sst_so_subset.where(season_mask_temp, drop=True)
season_mask_chl = chl_cmems_so_subset.time.dt.month.isin([11, 12, 1, 2, 3, 4])
chl_season = chl_cmems_so_subset.where(season_mask_chl, drop=True) #shape (126, 231, 1442)


# Check resolution -- already in 0.25°
dlat = sst_season.latitude.diff("latitude")
dlon = sst_season.longitude.diff("longitude")
print("\nMean latitude resolution:", float(dlat.mean().values))
print("Mean longitude resolution:", float(dlon.mean().values))

# 4. Group by season and add coordinates
# Temperature
season_year_sst = xr.where(sst_season.time.dt.month >= 11, sst_season.time.dt.year, sst_season.time.dt.year - 1).data
sst_season = sst_season.assign_coords(season_year=("time", season_year_sst))
sst_by_season = sst_season.groupby("season_year")

# Chla
chl_cmems_so_subset = chl_cmems_so_subset.assign_coords(year=("time", chl_cmems_so_subset.time.dt.year.data), month=("time", chl_cmems_so_subset.time.dt.month.data))
season_year_chla = np.where(chl_season.time.dt.month.values >= 11, chl_season.time.dt.year.values, chl_season.time.dt.year.values - 1)
chl_season = chl_season.assign_coords(season_year=("time", season_year_chla), month=("time", chl_season.time.dt.month.values))

# 5. Put into a xarray dataset instead of dask
sst_season = sst_season.load()
sst_seasonal = (sst_season.assign_coords(month=("time", sst_season.time.dt.month.data))
                          .set_index(time=["season_year", "month"])
                          .unstack("time")
                          .sel(month=[11, 12, 1, 2, 3, 4]))
sst_seasonal_ds = sst_seasonal.to_dataset(name="sst")

chl_seasonal_ds = (
    chl_season
    .set_index(time=["season_year", "month"])
    .unstack("time")
    .sel(month=[11, 12, 1, 2, 3, 4])
    .to_dataset(name="chl")
)

# 6. From Kelvin to Celsius
sst_seasonal_ds["sst"] = sst_seasonal_ds["sst"] - 273.15
print("Min SST (°C):", float(sst_seasonal_ds["sst"].min().values)) #-1.91 °C
print("Max SST (°C):", float(sst_seasonal_ds["sst"].max().values)) #7.72°C



# %% ======================== Load ROMS ========================
# Load data
main_path='/nfs/sea/work/mlarriere/mhw_krill_SO'
temp_surf_roms = xr.open_dataset(os.path.join(main_path, 'temp_surf_seasonal.nc'))
chla_roms = xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc'))

# 1. Monthly mean of SST and Chla from ROMS
month_lengths = np.array([30, 31, 31, 28, 31, 30])  # Nov–Apr
months = np.array([11, 12, 1, 2, 3, 4])
month_index = np.concatenate([np.full(month_lengths[i], months[i]) for i in range(6)])
temp_roms = temp_surf_roms.assign_coords(month=("days", month_index))
chla_roms = chla_roms.assign_coords(month=("days", month_index))
temp_monthly = xr.concat([temp_roms["temp_surf"].isel(years=y).groupby("month").mean("days") for y in range(temp_roms.years.size)], dim=temp_roms.years)
chla_monthly = xr.concat([chla_roms["chla"].isel(years=y).groupby("month").mean("days") for y in range(chla_roms.years.size)], dim=chla_roms.years)

# Check
before_temp = temp_roms["temp_surf"].isel(eta_rho=200, xi_rho=1000, years=0, days=slice(0,30)).mean("days").values
after_temp  = temp_monthly.sel(month=11).isel(eta_rho=200, xi_rho=1000, years=0).values
print(f"before: {before_temp:.4f}°C, after: {after_temp:.4f}°C")
before_chla = chla_roms["chla"].isel(eta_rho=200, xi_rho=1000, years=0, days=slice(0,30)).mean("days").values
after_chla  = chla_monthly.sel(month=11).isel(eta_rho=200, xi_rho=1000, years=0).values
print(f"before: {before_chla:.4f} mg/m³, after: {after_chla:.4f} mg/m³")

# 2. MHWs from ROMS
ds_mhw = xr.open_dataset(os.path.join(path_combined_thesh, f"duration_AND_thresh_5mSEASON.nc"))
mhw_90th = ds_mhw["duration"] > 0

# Monthly MHW fraction
mhw_90th = mhw_90th.assign_coords(years=temp_surf_roms.years)  # assign 1980-2018
mhw_90th_coords = mhw_90th.assign_coords(month=("days", month_index))

mhw_monthly_frac = xr.concat([
    mhw_90th_coords.isel(years=y).groupby("month").mean("days")
    for y in range(mhw_90th_coords.years.size)
], dim=temp_surf_roms.years)

mhw_monthly_frac_reformat = mhw_monthly_frac.isel(years=slice(1998-1980, None)).rename({'years': 'season_year'})

# Check
mhw_before = mhw_90th_coords.isel(eta_rho=200, xi_rho=1100, years=36, days=slice(30, 61)).mean("days").values
mhw_after  = mhw_monthly_frac.sel(month=12).isel(eta_rho=200, xi_rho=1100, years=36).values
print(f"before: {mhw_before:.4f}, after: {mhw_after:.4f}")

# %% ======================== Regrid ROMS onto a regular grid (0.25°) to match OSTIA ========================
import xesmf as xe
regridded_path = os.path.join(main_path, 'sat_obs/roms_regridded')
regridded_path_obs = os.path.join(main_path, 'sat_obs/obs_regridded')
file_regrid_temp = os.path.join(regridded_path, "temp_monthly_regrid.nc")
file_regrid_chla = os.path.join(regridded_path, "chla_monthly_regrid.nc")
file_regrid_chla_obs = os.path.join(regridded_path_obs, "cmems_chla_monthly_regrid.nc")
file_regrid_mhw = os.path.join(regridded_path, "mhw_monthly_frac_regrid.nc")

if not os.path.exists(file_regrid_temp) and not os.path.exists(file_regrid_chla) and not os.path.exists(file_regrid_chla_obs) and not os.path.exists(file_regrid_mhw): 
    # Define source and target grids
    roms_grid = xr.Dataset({
        "lat": (["eta_rho", "xi_rho"], temp_surf_roms.lat_rho.values),
        "lon": (["eta_rho", "xi_rho"], temp_surf_roms.lon_rho.values),
    })
    ostia_grid = xr.Dataset({
        "lat": sst_seasonal_ds.latitude.values,
        "lon": sst_seasonal_ds.longitude.values,
    })

    # Perform the regridding
    regridder = xe.Regridder(roms_grid, ostia_grid, method="bilinear", periodic=True)
    temp_monthly_regrid = regridder(temp_monthly)
    chla_monthly_regrid = regridder(chla_monthly)
    chla_monthly_regrid_obs = regridder(chl_seasonal_ds["chl"])
    mhw_monthly_frac_regrid = regridder(mhw_monthly_frac)

    # Save to file
    temp_monthly_regrid.to_dataset(name="temp_surf").to_netcdf(file_regrid_temp)
    chla_monthly_regrid.to_dataset(name="chla").to_netcdf(file_regrid_chla)
    chla_monthly_regrid_obs.to_dataset(name="chla").to_netcdf(file_regrid_chla_obs)
    mhw_monthly_frac_regrid.to_dataset(name="mhw_monthly_frac").to_netcdf(file_regrid_mhw)

else:
    # Load the files and reformat 
    temp_monthly_regrid = xr.open_dataset(file_regrid_temp)["temp_surf"].rename({'years': 'season_year', 'lat': 'latitude', 'lon': 'longitude'})
    chla_monthly_regrid = xr.open_dataset(file_regrid_chla)["chla"].rename({'years': 'season_year', 'lat': 'latitude', 'lon': 'longitude'}) #shape (21, 6, 231, 1442)
    chla_monthly_regrid_obs = xr.open_dataset(file_regrid_chla_obs)["chla"].rename({'lat': 'latitude', 'lon': 'longitude'})
    mhw_monthly_frac_regrid = xr.open_dataset(file_regrid_mhw)["mhw_monthly_frac"].rename({'years': 'season_year', 'lat': 'latitude', 'lon': 'longitude'})


# %% ======================== Monthly Climatology ========================
# Compute the monthly temperature climatology (baseline: 1981-2010) for both ROMS and observations
temp_monthly_roms_clim = temp_monthly_regrid.sel(season_year=slice(1981, 2010)).mean(dim=['season_year'], skipna=True) #shape (6, 120, 1440)
temp_monthly_obs_clim = sst_seasonal_ds["sst"].sel(season_year=slice(1981, 2010)).mean(dim=['season_year'], skipna=True) #shape (6, 120, 1440)

# Compute the 90th percentile threshold
temp_monthly_roms_p90 = temp_monthly_regrid.sel(season_year=slice(1981, 2010)).quantile(0.9, dim="season_year", skipna=True)
temp_monthly_obs_p90 = sst_seasonal_ds["sst"].sel(season_year=slice(1981, 2010)).quantile(0.9, dim="season_year", skipna=True)

# Based on the monthly climatology and 90th percentile threshold, detect 'extreme' thermal events, i.e. exceeding 90th perc
thermal_ext_monthly_roms = temp_monthly_regrid.groupby("month") > temp_monthly_roms_p90
thermal_ext_monthly_obs = sst_seasonal_ds['sst'].groupby("month") > temp_monthly_obs_p90

# Check
ever_extreme_roms = thermal_ext_monthly_roms.any(dim=["season_year", "month"])
ever_extreme_obs  = thermal_ext_monthly_obs.any(dim=["season_year", "month"])
print("ROMS fraction ever extreme:", ever_extreme_roms.mean().values)
print("OBS fraction ever extreme:", ever_extreme_obs.mean().values)

# -- Chla during extreme thermal events
# Hindcast = 1998-2018, i.e. 21 years, to match the observation hindcast
# Align 
thermal_extreme_21yr_roms = thermal_ext_monthly_roms.sel(season_year=slice(chla_monthly_regrid_obs.season_year[0].values, chla_monthly_regrid_obs.season_year[-1].values))
thermal_extreme_21yr_obs = thermal_ext_monthly_obs.sel(season_year=slice(chla_monthly_regrid_obs.season_year[0].values, chla_monthly_regrid_obs.season_year[-1].values))

# Extract chla values during extreme thermal events
chla_ext_monthly_roms = chla_monthly_regrid.where(thermal_extreme_21yr_roms) #shape (21, 6, 120, 1440)
chla_ext_monthly_obs = chla_monthly_regrid_obs.where(thermal_extreme_21yr_obs) #shape (21, 6, 120, 1440)

chla_ext_monthly_roms_yearly = chla_ext_monthly_roms.mean(dim=['season_year'], skipna=True)
chla_ext_monthly_obs_yearly = chla_ext_monthly_obs.mean(dim=['season_year'], skipna=True)
bias_chla_ext_yearly = chla_ext_monthly_roms_yearly - chla_ext_monthly_obs_yearly


chla_ext_monthly_roms_test = chla_ext_monthly_roms.sel(season_year=2016, month=1)
chla_ext_monthly_obs_test = chla_ext_monthly_obs.sel(season_year=2016, month=1)
bias_chla_ext_test = chla_ext_monthly_roms_test - chla_ext_monthly_obs_test

# Compute (event-conditioned) mean 
# chla_ext_monthly_roms_mean = chla_ext_monthly_roms.mean(dim=["season_year", "month"], skipna=True) #shape (6, 120, 1440)
# chla_ext_monthly_obs_mean = chla_ext_monthly_obs.mean(dim=["season_year", "month"], skipna=True) #shape (6, 120, 1440)
# bias_chla_ext = chla_ext_monthly_roms_mean - chla_ext_monthly_obs_mean

# %% ======================== Visualise Extreme on timeseries ========================
import pandas as pd
# ----- Reorganise the time
# ROMS
time_index_roms = pd.to_datetime([f"{y}-{m:02d}-15"
                                  for y in temp_monthly_regrid.season_year.values
                                  for m in temp_monthly_regrid.month.values])

# OBS
time_index_obs = pd.to_datetime([f"{y}-{m:02d}-15"
                                 for y in sst_seasonal_ds.season_year.values
                                 for m in sst_seasonal_ds.month.values])
sort_idx_obs = np.argsort(time_index_obs)
time_index_obs = time_index_obs[sort_idx_obs]

# ----- Mask time
start = "2010-01-01"
end   = "2018-12-31"

# ROMS
time_mask_roms = (time_index_roms >= start) & (time_index_roms <= end)
time_roms_crop = time_index_roms[time_mask_roms]

# OBS
time_mask_obs = (time_index_obs >= start) & (time_index_obs <= end)
time_obs_crop = time_index_obs[time_mask_obs]

# ----- Select location
lat_choice, lon_choice = 100, 1100

# ROMS
sst_roms = temp_monthly_regrid.isel(latitude=lat_choice, longitude=lon_choice).values.reshape(-1)
sst_roms_crop = sst_roms[time_mask_roms]
roms_clim_month = temp_monthly_roms_clim.isel(latitude=lat_choice, longitude=lon_choice).sel(month=temp_monthly_regrid.month.values).values
roms_p90_month = temp_monthly_roms_p90.isel(latitude=lat_choice, longitude=lon_choice).sel(month=temp_monthly_regrid.month.values).values

# OBS
sst_obs = sst_seasonal_ds["sst"].isel(latitude=lat_choice, longitude=lon_choice).values.reshape(-1)
sst_obs_crop = sst_obs[time_mask_obs]
obs_clim_month = temp_monthly_obs_clim.isel(latitude=lat_choice, longitude=lon_choice).sel(month=sst_seasonal_ds.month.values).values
obs_p90_month = temp_monthly_obs_p90.isel(latitude=lat_choice, longitude=lon_choice).sel(month=sst_seasonal_ds.month.values).values

# ----- Expand the time dimensions to fit the full hindcast 
roms_clim_full = np.tile(roms_clim_month, len(temp_monthly_regrid.season_year))
clim_roms_crop = roms_clim_full[time_mask_roms]
roms_p90_full = np.tile(roms_p90_month, len(temp_monthly_regrid.season_year))
p90_roms_crop  = roms_p90_full[time_mask_roms]

obs_clim_full = np.tile(obs_clim_month, len(sst_seasonal_ds.season_year))
clim_obs_crop = obs_clim_full[time_mask_obs]
obs_p90_full = np.tile(obs_p90_month, len(sst_seasonal_ds.season_year))
p90_obs_crop = obs_p90_full[time_mask_obs]

# ----- Extreme thermal events
ext_roms_crop  = thermal_ext_monthly_roms.isel(latitude=lat_choice, longitude=lon_choice).values.reshape(-1)[time_mask_roms]
ext_obs_crop = thermal_ext_monthly_obs.isel(latitude=lat_choice, longitude=lon_choice).values.reshape(-1)[time_mask_obs]

# ----- Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# -- 1.ROMS
ax = axes[0]
ax.plot(time_roms_crop, sst_roms_crop, color="black", linewidth=1, label="SST")
ax.plot(time_roms_crop, clim_roms_crop, linestyle="--", label="Climatology")
ax.plot(time_roms_crop, p90_roms_crop, linestyle="-", label="90th percentile")
ax.fill_between(time_roms_crop, sst_roms_crop, p90_roms_crop, where=ext_roms_crop,
                color="red", alpha=0.4, label="Extreme warm months")
ax.set_title(f"ROMS ({start[0:4]}-{end[0:4]})")
ax.set_ylabel("Temperature [°C]")
ax.legend()

# -- 2. OBS
ax = axes[1]
ax.plot(time_obs_crop, sst_obs_crop, color="black", linewidth=1, label="SST")
ax.plot(time_obs_crop, clim_obs_crop, linestyle="--", label="Climatology")
ax.plot(time_obs_crop, p90_obs_crop, linestyle="-", label="90th percentile")
ax.fill_between(time_obs_crop, sst_obs_crop, p90_obs_crop, where=ext_obs_crop,
                color="red", alpha=0.4, label="Extreme warm months")
ax.set_title(f"OBS ({start[0:4]}-{end[0:4]})")
ax.set_ylabel("Temperature [°C]")
ax.set_xlabel("Year")
ax.legend()
plt.tight_layout()
plt.show()



# %% ======================== Visualise Extreme on Maps ========================
plot = 'report'
title_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 11}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 10}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 13, 'fontweight': 'bold'}

fig, axes = plt.subplots(2, 2, figsize=(5, 5), subplot_kw={'projection': ccrs.SouthPolarStereo()})
plt.subplots_adjust(wspace=0.05, hspace=0.3)

theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)


vmin, vmax = -2, 2
cmap = 'coolwarm'

# Select data
year= 2018
month = 1
month_name =["November", "December", "January", "February", "March", "April"]
temp_roms_extreme = temp_monthly_regrid.where(thermal_ext_monthly_roms).isel(season_year=1980-year, month=month)
temp_obs_extreme = sst_seasonal_ds["sst"].where(thermal_ext_monthly_obs).isel(season_year=1980-year, month=month)

temp_roms = temp_monthly_regrid.isel(season_year=1980-year, month=month)
temp_obs = sst_seasonal_ds["sst"].isel(season_year=1980-year, month=month)

# 1st Row: SST
ax0 = axes[0, 0]
pcm1 = ax0.pcolormesh(temp_roms.longitude, temp_roms.latitude, temp_roms,
                     transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                     shading='auto')
ax0.set_title('ROMS', **title_kwargs)

ax1 = axes[0, 1]
psm2 = ax1.pcolormesh(temp_obs.longitude, temp_obs.latitude, temp_obs, 
                    transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                    shading='auto')
ax1.set_title('OSTIA', **title_kwargs)

# 2nd Row: Extremes
# -- 1. ROMS
ax2 = axes[1, 0]
pcm1 = ax2.pcolormesh(temp_roms_extreme.longitude, temp_roms_extreme.latitude, temp_roms_extreme,
                     transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                     shading='auto')
ax2.set_title('Extremes ROMS', **title_kwargs)

# -- 2. OBS
ax3 = axes[1, 1]
pcm2 = ax3.pcolormesh(temp_obs_extreme.longitude, temp_obs_extreme.latitude, temp_obs_extreme,
                     transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                     shading='auto')
ax3.set_title('Extremes OSTIA', **title_kwargs)

for ax in axes.flat:
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.set_facecolor('white')
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7, zorder=7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
# Colorbars

cbar1 = fig.colorbar(pcm1, ax=axes[:2],
                     orientation='horizontal', extend='max',
                     fraction=0.05, pad=0.08)
cbar1.set_label('Temperature [°C]', **label_kwargs)
cbar1.ax.tick_params(**tick_kwargs)


fig.suptitle(f'Monthly Temperature during MHWs\n {month_name[month]} {year}', 
             y=1.02, **suptitle_kwargs)

# plt.tight_layout()
plt.show()
# plt.savefig(f'D_Paper_Scripts/figures/sup_mat/temperatures_ostia_roms_{month_name[month]}_{year}.pdf', dpi=200, format='pdf', bbox_inches='tight')

# %% ======================== Visualise Chla during Extreme on Maps ========================
plot = 'report'
title_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 11}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 10}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 13, 'fontweight': 'bold'}

fig, axes = plt.subplots(6, 3, figsize=(5, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})
plt.subplots_adjust(wspace=0.5, hspace=0.3)

theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

vmin, vmax = 0, 1
colors = ["#0E1B11", "#4A8956", "#73A942", "#E7D20D", "#FBB02D"]
color_positions = np.linspace(vmin, vmax, len(colors))
normalized_positions = (color_positions - vmin) / (vmax - vmin)  # Normalize to [0, 1]
cmap_var = LinearSegmentedColormap.from_list("blue_green_yellow_buffered", list(zip(normalized_positions, colors)), N=256)
# bias_lim = np.nanmax(np.abs(bias.values))
bias_lim = 1

months=[11, 12, 1, 2, 3, 4]
months_names = ['November', 'December', 'January', 'February', 'March', 'April']
for i, m in enumerate(months):
    
    # -- 1. OBS
    ax0 = axes[i, 0]
    pcm1 = ax0.pcolormesh(chla_ext_monthly_obs_yearly.longitude, chla_ext_monthly_obs_yearly.latitude, chla_ext_monthly_obs_yearly.sel(month=m),
                        transform=ccrs.PlateCarree(), cmap=cmap_var, vmin=vmin, vmax=vmax,
                        shading='auto')
    if i==0:
        ax0.set_title('GlobColour', **title_kwargs)

    # -- 2. ROMS
    ax1 = axes[i, 1]
    pcm2 = ax1.pcolormesh(chla_ext_monthly_roms_yearly.longitude, chla_ext_monthly_roms_yearly.latitude, chla_ext_monthly_roms_yearly.sel(month=m),
                        transform=ccrs.PlateCarree(), cmap=cmap_var, vmin=vmin, vmax=vmax,
                        shading='auto')
    if i==0:
        ax1.set_title('ROMS', **title_kwargs)

    # -- 3. Bias (ROMS - OBS)
    ax2 = axes[i, 2]
    pcm3 = ax2.pcolormesh(bias_chla_ext_yearly.longitude, bias_chla_ext_yearly.latitude, bias_chla_ext_yearly.sel(month=m),
                        transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                        vmin=-bias_lim, vmax=bias_lim, shading='auto')
    if i==0: 
        ax2.set_title('Difference\n(ROMS - GlobColour)', **title_kwargs)

# Row title
for i, name in enumerate(months_names):
    fig.text(0.04, 0.86 - i * 0.14, # y-position (row spacing)
        name, va='center', ha='left', rotation=90,
        **title_kwargs)
    
for ax in axes.flat:
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.set_facecolor('white')
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7, zorder=7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
# Colorbars
cbar_ax1 = fig.add_axes([0.1, 0.08, 0.5, 0.01])  #[left, bottom, width, height]
cbar1 = fig.colorbar(pcm1, cax=cbar_ax1,
                     orientation='horizontal', extend='max',
                     fraction=0.05, pad=0.08)
cbar1.set_label('Chla [mg m$^{-3}$]', **label_kwargs)
cbar1.ax.tick_params(**tick_kwargs)

cbar_ax2 = fig.add_axes([0.65, 0.08, 0.3, 0.01])
cbar2 = fig.colorbar(pcm3, cax=cbar_ax2,
                     orientation='horizontal', extend='both',
                     fraction=0.05, pad=0.08)
cbar2.set_label('Bias [mg m$^{-3}$]', **label_kwargs)
cbar2.ax.tick_params(**tick_kwargs)

fig.suptitle('Mean Chlorophyll-a Concentration during MHWs\n'
             'Comparison between observations and ROMS (1998–2019)',
             y=0.96, **suptitle_kwargs)

# plt.tight_layout()
plt.show()
# plt.savefig(f'D_Paper_Scripts/figures/sup_mat/temperatures_ostia_roms_monthly.pdf', dpi=200, format='pdf', bbox_inches='tight')

# %%

# Retrieve cells where MHWs happend, i.e. fraction > 0
mhw_cells_regrid = mhw_monthly_frac_regrid > 0
mhw_cells_tot_regrid = mhw_cells_regrid.any(dim=['season_year', 'month']) #shape (120, 1440)
# mhw_cells = mhw_monthly_frac_reformat > 0
# mhw_cells_tot = mhw_cells.any(dim=['season_year', 'month']) #shape (231, 1442)

sst_obs_mhw_cells = sst_seasonal_ds["sst"].isel(season_year=slice(1997-1980, None)).where(mhw_cells_tot_regrid) #shape (120, 1440, 21, 6)
sst_roms_mhw_cells = temp_monthly_regrid.where(mhw_cells_tot_regrid) #shape (21, 6, 120, 1440)

chla_obs_mhw_cells = chla_monthly_regrid_obs.where(mhw_cells_tot_regrid) #shape (21, 6, 120, 1440)
chla_roms_mhw_cells = chla_monthly_regrid.where(mhw_cells_tot_regrid) #shape (21, 6, 120, 1440)



# %% ======================== Mean Chla during MHWs ========================
# Take the values only when MHWs happend - take the mean over these values otherwise signal influence by non-MHWs period
mhw_mask = mhw_monthly_frac_regrid>0
chla_obs_mhw = chla_monthly_regrid_obs.where(mhw_mask) #shape (21, 6, 120, 1440)
chla_roms_mhw = chla_monthly_regrid.where(mhw_mask) #shape (21, 6, 120, 1440)

# ---- Mean over full hindcast ----
chla_obs_mhw_mean  = chla_obs_mhw.mean(dim=['season_year', 'month'], skipna=True)
chla_roms_mhw_mean = chla_roms_mhw.mean(dim=['season_year', 'month'], skipna=True)

# ---- Bias ----
bias = chla_roms_mhw_mean - chla_obs_mhw_mean

# %% ======================== Plot ========================
plot = 'report'
title_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 11}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 10}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 13, 'fontweight': 'bold'}

fig, axes = plt.subplots(1, 3, figsize=(10, 4), subplot_kw={'projection': ccrs.SouthPolarStereo()})
plt.subplots_adjust(wspace=0.5, hspace=0.4)

theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

vmin, vmax = 0, 1
colors = ["#0E1B11", "#4A8956", "#73A942", "#E7D20D", "#FBB02D"]
color_positions = np.linspace(vmin, vmax, len(colors))
normalized_positions = (color_positions - vmin) / (vmax - vmin)  # Normalize to [0, 1]
cmap_var = LinearSegmentedColormap.from_list("blue_green_yellow_buffered", list(zip(normalized_positions, colors)), N=256)
# bias_lim = np.nanmax(np.abs(bias.values))
bias_lim = 1

# -- 1. OBS
ax0 = axes[0]
pcm1 = ax0.pcolormesh(chla_obs_mhw_mean.longitude, chla_obs_mhw_mean.latitude, chla_obs_mhw_mean,
                     transform=ccrs.PlateCarree(), cmap=cmap_var, vmin=vmin, vmax=vmax,
                     shading='auto')
ax0.set_title('GlobColour', **title_kwargs)

# -- 2. ROMS
ax1 = axes[1]
pcm2 = ax1.pcolormesh(chla_roms_mhw_mean.longitude, chla_roms_mhw_mean.latitude, chla_roms_mhw_mean,
                     transform=ccrs.PlateCarree(), cmap=cmap_var, vmin=vmin, vmax=vmax,
                     shading='auto')
ax1.set_title('ROMS', **title_kwargs)

# -- 3. Bias (ROMS - OBS)
ax2 = axes[2]
pcm3 = ax2.pcolormesh(bias.longitude, bias.latitude, bias,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                     vmin=-bias_lim, vmax=bias_lim, shading='auto')
ax2.set_title('Difference (ROMS - GlobColour)', **title_kwargs)

for ax in axes:
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.set_facecolor('white')
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7, zorder=7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
# Colorbars
cbar1 = fig.colorbar(pcm1, ax=axes[:2],
                     orientation='horizontal', extend='max',
                     fraction=0.05, pad=0.08)
cbar1.set_label('Chla [mg m$^{-3}$]', **label_kwargs)
cbar1.ax.tick_params(**tick_kwargs)

cbar2 = fig.colorbar(pcm3, ax=axes[2],
                     orientation='horizontal', extend='both',
                     fraction=0.05, pad=0.08)
cbar2.set_label('Bias [mg m$^{-3}$]', **label_kwargs)
cbar2.ax.tick_params(**tick_kwargs)

fig.suptitle('Mean Chlorophyll-a Concentration during MHWs\n'
             'Comparison between observations and ROMS (1998–2019)',
             y=0.9, **suptitle_kwargs)

# plt.tight_layout()
plt.show()

# %% ======================== Comparison ========================
# Question to answer: Is the response of the model going in the same direction as the observations?     
# If Temperature and Chla concentration are going up together. 
# not interested in magnitude but in the trend
# “Do ecosystems respond similarly to the same physical forcing?”

# %%
def correlation(forcing, chl):

    mask = (~np.isnan(forcing)) & (~np.isnan(chl))

    if np.sum(mask) < 3:
        return np.nan

    if np.nanstd(forcing[mask]) == 0:
        return np.nan

    return np.corrcoef(forcing[mask], chl[mask])[0, 1]


def correlation_map(forcing, chl):

    return xr.apply_ufunc(
        correlation,
        forcing,
        chl,
        input_core_dims=[["season_year"], ["season_year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

months = sst_seasonal_ds.month
months_order = [11, 12, 1, 2, 3, 4]

chla_obs_monthly_response = {}
chla_roms_monthly_response = {}


for m in months_order:    
    print('--- Month:', m, '---\n')
    # m=11
    # ---- Select month ----
    sst_obs_m = sst_obs_mhw_cells.where(sst_obs_mhw_cells.month == m, drop=True)
    sst_roms_m = sst_roms_mhw_cells.where(sst_roms_mhw_cells.month == m, drop=True)

    chla_obs_m = chla_obs_mhw_cells.where(chla_obs_mhw_cells.month == m, drop=True)
    chla_roms_m = chla_roms_mhw_cells.where(chla_roms_mhw_cells.month == m, drop=True)

    # ---- Compute responses ----
    chla_obs_monthly_response[m] = correlation_map(sst_obs_m, chla_obs_m)
    chla_roms_monthly_response[m] = correlation_map(sst_roms_m, chla_roms_m)

# %% =============== SST–CHL sensitivity maps =========
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.path as mpath
import cartopy.feature as cfeature

months_order = [11, 12, 1, 2, 3, 4]
month_names  = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

fig, axes = plt.subplots(
    3, 6,
    figsize=(18, 9),
    subplot_kw={'projection': ccrs.SouthPolarStereo()}
)

theta = np.linspace(0, 2*np.pi, 200)
circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)

vmin, vmax = -0.05, 0.05  # trend scale (adjust if needed)
cmap = 'RdBu_r'

for col, (m, mname) in enumerate(zip(months_order, month_names)):

    # ================= OBS =================
    ax = axes[0, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    da = chla_obs_monthly_response[m].squeeze()

    pcm1 = ax.pcolormesh(
        da.lon_rho if 'lon_rho' in da.coords else da.longitude,
        da.lat_rho if 'lat_rho' in da.coords else da.latitude,
        da,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    ax.set_title(mname, fontsize=9)
    if col == 0:
        ax.text(-0.2, 0.5, "OBS correlation", transform=ax.transAxes,
                rotation=90, va='center')

    # ================= ROMS =================
    ax = axes[1, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    da = chla_roms_monthly_response[m].squeeze()

    pcm2 = ax.pcolormesh(
        da.lon_rho if 'lon_rho' in da.coords else chla_roms_mhw_cells.longitude,
        da.lat_rho if 'lat_rho' in da.coords else chla_roms_mhw_cells.latitude,
        da,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    if col == 0:
        ax.text(-0.2, 0.5, "ROMS correlation", transform=ax.transAxes,
                rotation=90, va='center')

    # ================= BIAS =================
    ax = axes[2, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    obs = chla_obs_monthly_response[m].squeeze()
    roms = chla_roms_monthly_response[m].squeeze()

    # align grids (assumes same grid already; if not, you must regrid)
    bias = roms - obs

    pcm3 = ax.pcolormesh(
        obs.lon_rho if 'lon_rho' in obs.coords else chla_roms_mhw_cells.longitude,
        obs.lat_rho if 'lat_rho' in obs.coords else chla_roms_mhw_cells.latitude,
        bias,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r', vmin=vmin, vmax=vmax
    )

    if col == 0:
        ax.text(-0.2, 0.5, "Bias", transform=ax.transAxes,
                rotation=90, va='center')

# --- colorbars ---
cbar1 = fig.colorbar(pcm1, ax=axes[0, :], shrink=0.7, pad=0.02)
cbar1.set_label("OBS correlation")

cbar2 = fig.colorbar(pcm2, ax=axes[1, :], shrink=0.7, pad=0.02)
cbar2.set_label("ROMS correlation")

cbar3 = fig.colorbar(pcm3, ax=axes[2, :], shrink=0.7, pad=0.02)
cbar3.set_label("Bias (ROMS - OBS)")

fig.suptitle("Chla trends during MHW cells (Nov–Apr)", fontsize=12)
plt.tight_layout()
plt.show()

# %% ======================== Comparison SST ========================
year_sel = 2016
months_order = [11, 12, 1, 2, 3, 4]
month_names  = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

fig, axes = plt.subplots(3, 6, figsize=(18, 9), subplot_kw={'projection': ccrs.SouthPolarStereo()})

vmin, vmax = -2, 8
cmap = 'RdBu_r'

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

for col, (m, mname) in enumerate(zip(months_order, month_names)):

    # -- Row 0: OSTIA observations
    ax = axes[0, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
    ax.set_facecolor('lightgrey')
    data_obs = sst_obs_mhw_cells.sel(season_year=year_sel, month=m)
    pcm = ax.pcolormesh(data_obs.longitude, data_obs.latitude, data_obs,
                        transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    if col == 0:
        ax.set_ylabel('OSTIA obs', fontsize=8)
    ax.set_title(mname, fontsize=9)

    # -- Row 1: ROMS regridded
    ax = axes[1, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
    ax.set_facecolor('lightgrey')
    data_roms = sst_roms_mhw_cells.sel(season_year=year_sel, month=m)
    pcm_r = ax.pcolormesh(data_roms.longitude, data_roms.latitude, data_roms,
                          transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    if col == 0:
        ax.set_ylabel('ROMS', fontsize=8)

    # -- Row 2: Difference (ROMS - OBS)
    ax = axes[2, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
    ax.set_facecolor('lightgrey')
    diff = data_roms - data_obs
    pcm_d = ax.pcolormesh(diff.longitude, diff.latitude, diff,
                          transform=ccrs.PlateCarree(), cmap='BrBG', vmin=-3, vmax=3)
    if col == 0:
        ax.set_ylabel('ROMS — OBS', fontsize=8)

# -- Colorbars
cbar_ax1 = fig.add_axes([0.92, 0.68, 0.01, 0.25])
cbar_ax2 = fig.add_axes([0.92, 0.38, 0.01, 0.25])
cbar_ax3 = fig.add_axes([0.92, 0.08, 0.01, 0.25])
fig.colorbar(pcm, cax=cbar_ax1, label='SST [°C]')
fig.colorbar(pcm_r, cax=cbar_ax2, label='SST [°C]')
fig.colorbar(pcm_d, cax=cbar_ax3, label='Bias [°C]')

fig.suptitle(f'Chla during MHW cells — season {year_sel}/{year_sel+1}', fontsize=12)
fig.subplots_adjust(left=0.06, right=0.91, top=0.91, bottom=0.05, wspace=0.05, hspace=0.15)
plt.show()

# %% ======================== Comparison Chla ========================
month_names  = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

fig, axes = plt.subplots(3, 6, figsize=(18, 9), subplot_kw={'projection': ccrs.SouthPolarStereo()})

vmin, vmax = 0, 1
cmap = 'viridis'

theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

for col, (m, mname) in enumerate(zip(months_order, month_names)):

    # -- Row 0: Chla OBS (curvilinear)
    ax = axes[0, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
    ax.set_facecolor('lightgrey')
    data_obs = chla_obs_mhw_cells.sel(month=m)
    pcm_o = ax.pcolormesh(data_obs.lon_rho, data_obs.lat_rho, data_obs,
                          transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    if col == 0:
        ax.text(-0.15, 0.5, 'Chla OBS', transform=ax.transAxes,
                fontsize=8, va='center', rotation=90)
    ax.set_title(mname, fontsize=9)

    # -- Row 1: Chla ROMS regridded (regular grid)
    ax = axes[1, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
    ax.set_facecolor('lightgrey')
    data_roms = chla_roms_mhw_cells.sel(month=m)
    pcm_r = ax.pcolormesh(chla_roms_mhw_cells.longitude, chla_roms_mhw_cells.latitude, data_roms,
                          transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    if col == 0:
        ax.text(-0.15, 0.5, 'Chla ROMS', transform=ax.transAxes,
                fontsize=8, va='center', rotation=90)

    # -- Row 2: Difference (ROMS - OBS) — need both on same grid first
    ax = axes[2, col]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
    ax.set_facecolor('lightgrey')
    ax.text(0.5, 0.5, 'Regrid needed\nfor diff', transform=ax.transAxes,
            fontsize=7, ha='center', va='center', color='gray')
    if col == 0:
        ax.text(-0.15, 0.5, 'ROMS - OBS', transform=ax.transAxes,
                fontsize=8, va='center', rotation=90)

# -- Colorbars
cbar_ax1 = fig.add_axes([0.92, 0.68, 0.01, 0.22])
cbar_ax2 = fig.add_axes([0.92, 0.38, 0.01, 0.22])
fig.colorbar(pcm_o, cax=cbar_ax1, label='Chla [mg/m³]')
fig.colorbar(pcm_r, cax=cbar_ax2, label='Chla [mg/m³]')

fig.suptitle(f'Chla during MHW cells — season {year_sel}/{year_sel+1}', fontsize=12)
fig.subplots_adjust(left=0.06, right=0.91, top=0.93, bottom=0.05, wspace=0.05, hspace=0.15)
plt.show()

# %%
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# fig, axes = plt.subplots(1, 2, figsize=(14, 5),
#                          subplot_kw={'projection': ccrs.SouthPolarStereo()})

# year_idx, month_sel = 0, 11
# # Circular boundary
# theta = np.linspace(0, 2 * np.pi, 200)
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * 0.5 + 0.5)

# # -- Before: ROMS curvilinear
# ax = axes[0]
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.coastlines(linewidth=0.5)
# ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
# ax.set_facecolor('lightgrey')
# data_before = temp_monthly.sel(month=month_sel).isel(years=year_idx)
# pcm = ax.pcolormesh(temp_surf_roms.lon_rho, temp_surf_roms.lat_rho, data_before,
#                     transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-2, vmax=8)
# ax.set_title(f'ROMS (curvilinear)\nNov {int(temp_roms.years[year_idx].values)}', fontsize=10)
# fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, label='SST [°C]')

# # -- After: regridded onto OSTIA grid
# ax = axes[1]
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.coastlines(linewidth=0.5)
# ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
# ax.set_facecolor('lightgrey')
# data_after = temp_monthly_regrid.sel(month=month_sel).isel(years=year_idx)
# pcm = ax.pcolormesh(sst_seasonal_ds.longitude, sst_seasonal_ds.latitude, data_after,
#                     transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-2, vmax=8)
# ax.set_title(f'ROMS regridded (0.25°)\nNov {int(temp_roms.years[year_idx].values)}', fontsize=10)
# fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, label='SST [°C]')

# fig.suptitle('Regridding check: ROMS → OSTIA grid', fontsize=11)
# plt.show()


"""
Created on Frid 05 Dec 09:07:45 2025

Removing MHWs from the temperature signal -- fake world 

@author: Marguerite Larriere (mlarriere)
"""

# %% -------------------------------- PACKAGES ------------------------------------
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
# %% -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

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
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_masslength = os.path.join(path_surrogates, f'mass_length')
path_trend = os.path.join(path_surrogates, 'detrended_signal')


# %% ======================== Load data ========================
# --- Drivers
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 365, 231, 1442)
temp_avg_100m_SO_allyrs = temp_avg_100m_SO_allyrs.rename({'year':'years'})

# --- MHW events
mhw_duration_seasonal = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))) #shape (39, 181, 231, 1442)
mhw_duration_seasonal = mhw_duration_seasonal.drop_vars('days')              # remove old days coordinate/variable
mhw_duration_seasonal = mhw_duration_seasonal.rename({'days_of_yr': 'days'}) # rename new axis to days

# == Climatological Drivers -> Mean T°C  (days, eta, xi) period: 1980-2009
temp_clim = temp_avg_100m_SO_allyrs.isel(years=slice(0,30)) #shape: (30, 365, 231, 1442)
temp_clim_mean = temp_clim.mean(dim=['years']) #shape: (365, 231, 1442)

# %% ======================== Visualization daily ========================
eta_choice = 200
xi_choice = 1100

lat = temp_clim_mean.isel(eta_rho=eta_choice, xi_rho=xi_choice).lat_rho.values
lon = temp_clim_mean.isel(eta_rho=eta_choice, xi_rho=xi_choice).lon_rho.values

data_clim = temp_clim_mean.isel(eta_rho=200, xi_rho=1100).avg_temp #shape (365)
data_temp = temp_avg_100m_SO_allyrs.isel(eta_rho=eta_choice, xi_rho=xi_choice).avg_temp  #shape (40, 365)

# Full hindcast daily: 365*40days
n_years, n_days = data_temp.shape
data_temp_full = data_temp.values.flatten()  # 365*40 = 14600 days
data_clim_full = np.tile(data_clim.values, n_years)  # repeat 40 times

# Continuous time series
n_years, n_days = data_temp.shape
days_full = np.arange(1, n_years*n_days + 1)  

# Year positions for x-ticks
years = temp_avg_100m_SO_allyrs.years.values
xticks = np.arange(0, n_years*n_days, 5*n_days)  # every 5 years
xticklabels = [str(y) for y in years[::5]]

plt.figure(figsize=(16,5))
plt.plot(days_full, data_clim_full, "-", label="Climatology (1980-2009)", color="#0A9396", alpha=0.7)
plt.plot(days_full, data_temp_full, "r--", label="Modelled hindcast", linewidth=1)
plt.xticks(xticks, xticklabels, fontsize=14)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)
plt.title(f"Daily ROMS Temperature\nLoc: ({-lat:.2f}°S, {lon:.2f}°E)", fontsize=18)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# %% ======================== Functions ========================
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

# %% ======================== Trending Climatological signal ========================
output_file = os.path.join(path_trend, 'clim_with_trend.nc')

if not os.path.exists(output_file):
    # -- Load data
    temp_trend = xr.open_dataset(os.path.join(path_trend, 'temp_linear_trend_100mavg.nc')) #shape (231, 1442), in [°C/yr]
    temp_clim_expanded = temp_clim_mean.expand_dims({'years': temp_avg_100m_SO_allyrs.years})

    # -- Adding warming trend to climatological signal
    reference_year = 1980 
    temp_no_mhws = temp_clim_expanded + temp_trend.slope * (temp_avg_100m_SO_allyrs.years - reference_year) #shape (40, 365, 231, 1442)

    # -- Growth seasonal
    temp_no_mhws_seasons = define_season_all_years_parallel(temp_no_mhws, max_workers=6) #(39, 181, 231, 1442)

    # Add metadata
    temp_no_mhws_seasons = temp_no_mhws_seasons.rename({'season_year': 'season_year_temp'})
    temp_no_mhws_seasons = temp_no_mhws_seasons.drop_vars('years')
    temp_no_mhws_seasons = temp_no_mhws_seasons.rename({'season_year_temp': 'years'})
    temp_no_mhws_seasons.attrs['description'] = ("Climatological temperature signal with warming trend.\n"
                                                 "The trend is estimated using OLS (fitted on annual means).\n"
                                                 "Temporal extent: Growth season (Nov 1 – Apr 30)")
    
    # -- Save to file
    temp_no_mhws_seasons.to_netcdf(output_file)
else:
    # Load Seasonal datasets
    temp_no_mhws_seasons = xr.open_dataset(output_file)

    temp_avg_100m_SO_allyrs_season = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (39, 181, 231, 1442)
    temp_clim_season = temp_avg_100m_SO_allyrs_season.isel(years=slice(0,30)).mean(dim=['years']) #shape: (181, 231, 1442)


# %% ======================== Visualisation yearly ========================
eta_choice = 200
xi_choice = 1100

lat = temp_clim_mean.isel(eta_rho=eta_choice, xi_rho=xi_choice).lat_rho.values
lon = temp_clim_mean.isel(eta_rho=eta_choice, xi_rho=xi_choice).lon_rho.values

data_clim = temp_clim_season.isel(eta_rho=200, xi_rho=1100).avg_temp #shape (181, )
data_temp = temp_avg_100m_SO_allyrs_season.isel(eta_rho=eta_choice, xi_rho=xi_choice).avg_temp  #shape (39, 181)

# data_temp_yearly = np.array([data_temp.values[years == yr].mean() for yr in np.unique(years)])
data_temp_yearly = np.array([data_temp.sel(years=yr).mean().values for yr in data_temp.years.values])

# Full hindcast daily: 365*40days
n_years, n_days = data_temp.shape
data_temp_full = data_temp.values.flatten()  # 365*40 = 14600 days
data_clim_full = np.tile(data_clim.values, n_years)  # repeat 40 times

# Continuous time series
n_years, n_days = data_temp.shape
days_full = np.arange(1, n_years*n_days + 1)  

# Year positions for x-ticks
years = temp_avg_100m_SO_allyrs.years.values
xticks = np.arange(0, n_years*n_days, 5*n_days)  # every 5 years
xticklabels = [str(y) for y in years[::5]]

plt.figure(figsize=(16,5))
plt.plot(days_full, data_clim_full, "-", label="Climatology (1980-2009)", color="#0A9396", alpha=0.7)
plt.plot(days_full, data_temp_full, "r--", label="Modelled hindcast", linewidth=1)
plt.xticks(xticks, xticklabels, fontsize=14)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)
plt.title(f"Daily ROMS Temperature\nLoc: ({-lat:.2f}°S, {lon:.2f}°E)", fontsize=18)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% ================== Identify MHW events for 1 cell ==================
from scipy.ndimage import label
def identify_events(data, threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']):
    threshold_events = {}

    hobday_mask = data.duration.values > 0  # relative threshold (Hobday)
    days = np.arange(len(hobday_mask))

    for var in threshold_vars:
        # var='det_1deg'
        abs_mask = data[var].fillna(0).astype(bool).values  # absolute threshold

        # Combine relative & absolute thresholds
        mhw_mask = hobday_mask & abs_mask

        # Label consecutive days
        labeled_array, num_events = label(mhw_mask)

        events = []
        for event_id in range(1, num_events + 1):
            idx = np.where(labeled_array == event_id)[0] #idx of days when MHWs is happening
            # print(idx)
            if len(idx) == 0:
                continue
            
            duration = len(idx)
            # print(duration)

            if duration > 10:
                events.append({
                    "event_id": event_id,
                    "start_day": int(idx[0]),
                    "end_day": int(idx[-1]),
                    "duration": duration
                })
            # print(event_duration)

            # Store results
        threshold_events[var] = {"events": events, "days": labeled_array}

    return threshold_events


# %% ======================== Plot before VS after ========================
from matplotlib.patches import Patch
# ================= Prepare data =================
eta_choice = 190 #190 200
xi_choice = 1000 #1000 1100
year_choice = 9 #9 36

lat = temp_clim_mean.isel(eta_rho=eta_choice, xi_rho=xi_choice).lat_rho.values
lon = temp_clim_mean.isel(eta_rho=eta_choice, xi_rho=xi_choice).lon_rho.values

days = temp_avg_100m_SO_allyrs.days

# --- Prepare data
temp_year_plot = temp_avg_100m_SO_allyrs_season.avg_temp.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)
temp_non_mhws_plot = temp_no_mhws_seasons.avg_temp.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)
temp_clim_plot = temp_clim_season.avg_temp.isel(eta_rho=eta_choice, xi_rho=xi_choice)
mhw_events_plot = mhw_duration_seasonal.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)

# --- Settings
threshold_vars   = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = [r'$\geq$ 90th perc and 1°C', r'$\geq$ 90th perc and 2°C', r'$\geq$ 90th perc and 3°C', r'$\geq$ 90th perc and 4°C']
threshold_events = identify_events(mhw_events_plot, threshold_vars=threshold_vars)

# --- x-axis
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15) #ticks every 15days
tick_labels = [date_dict.get(day, '') for day in tick_positions]

# --- Figure
fig, (ax_timeline, ax_temp) = plt.subplots(2, 1, figsize=(18, 6), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# --- MHW timeline
for i, var in enumerate(threshold_vars):
    color = threshold_colors[i]
    for ev in threshold_events[var]["events"]:
        ax_timeline.axvspan(ev["start_day"], ev["end_day"], color=color, alpha=0.8)

ax_timeline.set_ylabel('MHWs', fontsize=12)
ax_timeline.set_yticks([])

# --- Temperature signals
ax_temp.plot(days_xaxis, temp_non_mhws_plot, label='Non-MHW (Clim + Trend)', color='#CA6702')
ax_temp.plot(days_xaxis, temp_clim_plot, label='Climatology (1980-2009)', color='#D8E2DC', linewidth=2)
ax_temp.plot(days_xaxis, temp_year_plot, label=f'Temp with MHWs', color='#B10CA1', linestyle='--')
ax_temp.set_xlabel('Days', fontsize=12)
ax_temp.set_ylabel('100m-avg Temperature [°C]', fontsize=12)
ax_temp.grid(alpha=0.3)

# X-axis
ax_temp.set_xticks(tick_positions)
ax_temp.set_xticklabels(tick_labels, rotation=45, fontsize=10)
ax_temp.legend(fontsize=12, loc='upper left')

# --- Legend
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=threshold_colors[i], label=threshold_labels[i]) for i in range(len(threshold_vars))]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.1))

fig.suptitle(f"Temperature signal without MHWs\nLocation: ({lat:.2f}°S, {lon:.2f}°E), in {1980+year_choice}", fontsize=16, y=0.99)

plt.tight_layout()
plt.show()


# %% ======================== Replacing MHWs by climatology ========================
def replace_MHW_clim(yr):
    print(f'Processing year {yr+1980}')

    # Mask where MHWs occured
    mhw_mask = mhw_duration_seasonal.duration.isel(years=yr) != 0 #shape (181, 231, 1442)

    # Replace MHWs data by climatology
    temp_without_MHWs = temp_avg_100m_SO_allyrs.avg_temp.isel(years=yr).where(~mhw_mask, temp_clim_mean.avg_temp)

    return temp_without_MHWs.assign_coords(years=temp_avg_100m_SO_allyrs.years.isel(years=yr)).expand_dims("years")

output_file = os.path.join(path_growth_inputs, 'temp_avg100m_noMHWs.nc')
if not os.path.exists(output_file): 
    # Run in parallel
    results = process_map(replace_MHW_clim, range(0, 39), max_workers=10, desc="Processing year")  

    # Combine years together
    temp_without_MHWs_allyrs = xr.concat(results, dim="years")

    # --- Save to file 
    ds = xr.Dataset(
        {"avg_temp": (["years", "days", "eta_rho", "xi_rho"], temp_without_MHWs_allyrs.data)},
        coords={
            "years": temp_avg_100m_SO_allyrs.years,
            "days": temp_avg_100m_SO_allyrs.days,
            "lon_rho": (["eta_rho", "xi_rho"], temp_avg_100m_SO_allyrs.lon_rho.data),
            "lat_rho": (["eta_rho", "xi_rho"], temp_avg_100m_SO_allyrs.lat_rho.data),
        },
        attrs={
            "description": (
                "Temperature (100m avg) without MHWs. "
                "When MHWs are detected at the surface, the 100m-avg temperature signal is replaced by the  100m-avg temperature climatology (1980–2009)."
            )
        }
    )

    ds.to_netcdf(output_file)

else: 
    temp_without_MHWs_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_noMHWs.nc'))



# %% ======================== Plot before VS after ========================
eta_choice = 200
xi_choice = 1100
year_choice = 36

data_before = temp_avg_100m_SO_allyrs.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)
data_after = temp_without_MHWs_allyrs.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)
mhw_events = mhw_duration_seasonal.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)

# === Prepare time axis ===
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15) #ticks every 15days
tick_labels = [date_dict.get(day, '') for day in tick_positions]

threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['$\\geq$ 90th perc and 1°C', '$\\geq$ 90th perc and 2°C', '$\\geq$ 90th perc and 3°C', '$\\geq$ 90th perc and 4°C']

fig, (ax_timeline, ax_temp) = plt.subplots(2, 1, figsize=(15, 5), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# ======= MHW timeline =======
threshold_events = identify_events(mhw_events)
threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
for i, var in enumerate(threshold_vars):
    color = threshold_colors[i]
    for ev in threshold_events[var]["events"]:
        ax_timeline.axvspan(ev["start_day"], ev["end_day"], color=color, alpha=0.8)

ax_timeline.set_ylabel("Surface \n MHW Events", fontsize=11)
ax_timeline.set_yticks([])

# ======= Temp time series =======
ax_temp.plot(days_xaxis, data_before.avg_temp.values, label='With MHWs', linewidth=1, color='#B10CA1')
ax_temp.plot(days_xaxis, data_after.avg_temp.values, label='Without MHWs', linewidth=1, color="black", linestyle='--')

ax_temp.set_ylabel("100m-avg temperature [°C]", fontsize=12)
ax_temp.set_xlabel("Date", fontsize=12)

# X-axis
ax_temp.set_xticks(tick_positions)
ax_temp.set_xticklabels(tick_labels, rotation=45, fontsize=10)

ax_temp.legend(fontsize=12, loc='upper left')


# Add thresholds legend
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=threshold_colors[i], label=threshold_labels[i]) for i in range(len(threshold_vars))]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.1))

# Common title
fig.suptitle("Deleting MHWs signal", fontsize=16, y=0.99)


plt.tight_layout()
plt.show()

# %%

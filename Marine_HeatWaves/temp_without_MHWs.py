"""
Created on Frid 05 Dec 09:07:45 2025

Removing MHWs from the temperature signal -- fake world 

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
# --- Drivers
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (39, 181, 231, 1442)
# chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 

# --- MHW events
mhw_duration_seasonal = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))) #shape (39, 181, 231, 1442)
mhw_duration_seasonal = mhw_duration_seasonal.drop_vars('days')              # remove old days coordinate/variable
mhw_duration_seasonal = mhw_duration_seasonal.rename({'days_of_yr': 'days'}) # rename new axis to days

# == Climatological Drivers -> Mean T°C  (days, eta, xi) period: 1980-2009
temp_clim = temp_avg_100m_SO_allyrs.isel(years=slice(0,30)) #shape: (30, 181, 231, 1442)
temp_clim_mean = temp_clim.mean(dim=['years']) #shape: (181, 231, 1442)


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


# %% ================== Identify events 1 cell ==================
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

            if duration > 0:
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
# ax_temp.plot(days_xaxis, data_after.avg_temp.values, label='Without MHWs', linewidth=1, color="black", linestyle='--')

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

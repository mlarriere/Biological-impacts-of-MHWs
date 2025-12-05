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

# %% ======================== Find MHWs and replace by climatology ========================
#test
eta_choice = 200
xi_choice = 1000
year_choice=9
temp_test = temp_avg_100m_SO_allyrs.isel(years=year_choice, eta_rho=eta_choice, xi_rho =xi_choice)
mhw_test = mhw_duration_seasonal.isel(years=year_choice, eta_rho=eta_choice, xi_rho =xi_choice)
temp_clim_mean_test = temp_clim_mean.isel(eta_rho=eta_choice, xi_rho =xi_choice)
print('MHWs duration:', mhw_test.duration.values)
# print('MHWs 4deg:', mhw_test.det_4deg.values)

# Mask where MHWs occured
mhw_mask = mhw_test.duration != 0 #shape (181,)

# Mask temp where MHW happend
temp_mhw = temp_test.where(mhw_test.duration.values!=0).avg_temp #value where MHW happend -- shape (181,)
print('Temperature during MHWs', temp_mhw.values)

# Where MHWs happend -- replace temperature values by climatological ones
temp_clim_mhws = temp_clim_mean_test.where(mhw_test.duration.values!=0).avg_temp #value where MHW happend -- shape (181,)
print('Climatology temperature during MHWs', temp_clim_mhws.values)

# Replace
temp_replaced_test = temp_test.avg_temp.where(~mhw_mask, temp_clim_mean_test.avg_temp)


# %% All cells Southern Ocean
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
data_before = temp_avg_100m_SO_allyrs.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)
data_after = temp_without_MHWs_allyrs.isel(years=year_choice, eta_rho=eta_choice, xi_rho=xi_choice)
# === Prepare time axis ===
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15) #ticks every 15days
tick_labels = [date_dict.get(day, '') for day in tick_positions]

# === Plot both time series on the SAME graph ===
fig, ax = plt.subplots(figsize=(15, 6))  # returns fig and ax

ax.plot(days_xaxis, data_before.avg_temp.values, label='Temperature with MHWs', linewidth=1, color='#9D4EDD')
ax.plot(days_xaxis, data_after.avg_temp.values, label='Temperature without MHWs', linewidth=1, color='#9B2226')

ax.set_title('Temperature Time Series (100m avg): With vs. Without MHWs', fontsize=15, y=1.1)
fig.text(0.5, 0.9, "Reminder: MHWs are defined at the surface.", ha='center', fontsize=12)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Temperature [°C]', fontsize=12)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()






# %%

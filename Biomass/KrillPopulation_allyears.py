"""
Created on Tue 31 July 15:06:14 2025

Length-Mass relationship

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
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

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


# %% ============== Load data ==============
def subset_spatial_domain(ds, lat_range=(-80, -60), lon_range=(270, 360)): #, (0, 30)
    lat_min, lat_max = lat_range
    lon_range1, lon_range2 = lon_range

    lat_mask = (ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max)
    lon_mask = ((ds['lon_rho'] >= lon_range1) & (ds['lon_rho'] <= lon_range2)) #| ((ds['lon_rho'] >= lon_range2[0]) & (ds['lon_rho'] <= lon_range2[1]))

    return ds.where(lat_mask & lon_mask, drop=True)

# === Data with MHW scenarios masks
# Output file paths
spatial_average = False  # Set to False for gridded output
suffix = '_ts' if spatial_average else ''
temp_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/temp_avg100m_daily_mhw{suffix}.nc")
chla_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/chla_surf_daily_mhw{suffix}.nc")
temp_non_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/temp_avg100m_daily_nomhw{suffix}.nc")
chla_non_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/chla_surf_daily_nomhw{suffix}.nc")

# Open files
temp_mhw = xr.open_dataset(temp_mhw_file)
chla_mhw = xr.open_dataset(chla_mhw_file)
temp_non_mhw = xr.open_dataset(temp_non_mhw_file)
chla_non_mhw = xr.open_dataset(chla_non_mhw_file)

# === Data without MHW scenarios masks
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) 
temp_avg_100m_study_area_allyrs = subset_spatial_domain(temp_avg_100m_SO_allyrs) 
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 
chla_surf_study_area_allyrs = subset_spatial_domain(chla_surf_SO_allyrs) 

# %% ============== Length evolution for all years ==============
from Growth_Model.growth_model import length_Atkison2006  

# Function to select pixels where growth happened, i.e. MHWs are detected and returning mean time serie
def compute_mean_ts(length_array):
    growth_pixels = length_array.max(dim='days') > length_array.min(dim='days')
    return length_array.where(growth_pixels).mean(dim=('eta_rho', 'xi_rho'))

# --- Parameters
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP accorindg to Tarling et al 2006 - graph under 0°C
mhw_thresholds = [1, 2, 3, 4]
mhw_thresholds_str = [f"{t}deg" for t in mhw_thresholds]
years = np.arange(1980, 2019)
year_indices = years - 1980  # 0 to 38


# --- Initialize dictionnaries
length_unmasked_all = {stage: [] for stage in stage_lengths}
mean_length_unmasked_all = {stage: [] for stage in stage_lengths}
length_non_MHWs_all = {stage: [] for stage in stage_lengths}
mean_length_non_MHWs_all = {stage: [] for stage in stage_lengths}
length_MHWs_all = {stage: {f"{thresh}deg": [] for thresh in [1, 2, 3, 4]} for stage in stage_lengths}
mean_length_MHWs_all = {stage: {f"{thresh}deg": [] for thresh in [1, 2, 3, 4]} for stage in stage_lengths}

for year_index in year_indices:
    print(f'---Processing {year_index+1980}')
    # === NON MHW FILTERING ===
    # Extract data
    temp_unmasked_1yr = temp_avg_100m_study_area_allyrs.isel(years=year_index)
    chla_unmasked_1yr = chla_surf_study_area_allyrs.isel(years=year_index)

    for stage in stage_lengths:
        length_unmasked = length_Atkison2006(chla=chla_unmasked_1yr['chla'], temp=temp_unmasked_1yr['avg_temp'],
                                             initial_length=stage_lengths[stage], intermoult_period=stage_IMP[stage],
                                             maturity_stage=stage)
        length_unmasked_all[stage].append(length_unmasked)

        # --- Mean Time Series
        mean_length_unmasked_all[stage].append(compute_mean_ts(length_unmasked))

    # === NON-MHW events ===
    # --- Select temporal extent
    temp_non_1yr = temp_non_mhw.isel(years=year_index)
    chla_non_1yr = chla_non_mhw.isel(years=year_index)

    # --- Compute Length 
    for stage in stage_lengths:
        length_non_mhw = length_Atkison2006(chla=chla_non_1yr[f'chla_nonmhw'], temp=temp_non_1yr[f'temp_nonmhw'],
                                    initial_length=stage_lengths[stage],
                                    intermoult_period=stage_IMP[stage],
                                    maturity_stage=stage)
        
        length_non_MHWs_all[stage].append(length_non_mhw)

        # --- Mean Time Series
        mean_length_non_MHWs_all[stage].append(compute_mean_ts(length_non_mhw))

    # === MHW events ===
    # --- Select temporal extent
    temp_mhw_1yr = temp_mhw.isel(years=year_index)
    chla_mhw_1yr = chla_mhw.isel(years=year_index)

    # --- Compute Length (for each intensity)
    for thresh in mhw_thresholds:
        chla_data = chla_mhw_1yr[f'chla_{thresh}deg']
        temp_data = temp_mhw_1yr[f'temp_{thresh}deg']
        for stage in stage_lengths:
            length_mhw = length_Atkison2006(chla=chla_data, temp=temp_data,
                                        initial_length=stage_lengths[stage],
                                        intermoult_period=stage_IMP[stage],
                                        maturity_stage=stage)
            length_MHWs_all[stage][f'{thresh}deg'].append(length_mhw)
            
            # --- Mean Time Series
            mean_length_MHWs_all[stage][f'{thresh}deg'].append(compute_mean_ts(length_mhw))


# %% ======================= Write To DataSet =======================
# --- Initialize dictionnaries
datasets_full_by_stage = {}
datasets_mean_by_stage = {}

for stage in stage_lengths:
    # --------- Spatial-length data
    data_vars_full = {'reference': xr.concat(length_non_MHWs_all[stage], dim='years')}
    data_vars_full = {'non_mhw': xr.concat(length_non_MHWs_all[stage], dim='years')}
    for thresh in mhw_thresholds_str:
        data_vars_full[f'mhw_{thresh}'] = xr.concat(length_MHWs_all[stage][f"{thresh}"], dim='years')
    
    ds_full = xr.Dataset(data_vars_full)
    ds_full = ds_full.assign_coords(years=years)
    ds_full = ds_full.assign_coords(
        lon_rho=(["eta_rho", "xi_rho"], temp_mhw.lon_rho.values),
        lat_rho=(["eta_rho", "xi_rho"], temp_mhw.lat_rho.values),
    )
    ds_full.attrs.update({
        'description': ('Krill length during MHW, non-MHW, and unmasked periods (full spatial data). '
                        'Includes intensity-specific MHW scenarios (1–4°C thresholds), non-MHW growth, '
                        'and a reference dataset without any MHW masking.'),
        'maturity_stage': stage,
        'temporal_window': 'Growth season'})

    datasets_full_by_stage[stage] = ds_full

    # --------- Mean time series data
    data_vars_mean = {
        'non_mhw': xr.concat(mean_length_non_MHWs_all[stage], dim='years')
    }
    for thresh in mhw_thresholds_str:
        data_vars_mean[f'mhw_{thresh}'] = xr.concat(mean_length_MHWs_all[stage][f"{thresh}"], dim='years')
    
    ds_mean = xr.Dataset(data_vars_mean)
    ds_mean = ds_mean.assign_coords(years=years)
    ds_mean.attrs.update({
        'description': ('Mean krill length time series during MHW, non-MHW, and unmasked periods (full spatial data). '
                        'Includes intensity-specific MHW scenarios (1–4°C thresholds), non-MHW growth, '
                        'and a reference dataset without any MHW masking.'),
        'maturity_stage': stage,
        'temporal_window': 'Growth season'
    })
    
    datasets_mean_by_stage[stage] = ds_mean


# --- Save to file
# Save full spatial datasets
for stage, ds_full in datasets_full_by_stage.items():
    fname = f"length_full_{stage}.nc"
    ds_full.to_netcdf(os.path.join(path_length, fname))
    print(f"Saved full dataset: {fname}")

# Save mean time series datasets
for stage, ds_mean in datasets_mean_by_stage.items():
    fname = f"length_mean_{stage}.nc"
    ds_mean.to_netcdf(os.path.join(path_length, fname))
    print(f"Saved mean dataset: {fname}")

# %% ============== Mass evolution for all years ==============
def length_to_mass(p, length_array, r):
    mass_array = p*length_array**r
    return mass_array

# --- Defining constants - Accroding to mass length coefficient of Atkison et al (2006)
p = 10**(-4.19)
r = 3.89

# --- Parameters
stages = ['juvenile', 'immature', 'mature', 'gravid']
mhw_thresholds = ['1deg', '2deg', '3deg', '4deg']

# --- Initialize dictionaries
mass_full_by_stage = {}
mass_mean_by_stage = {}

for stage in stages:
    # stage='juvenile'
    print(f'Processing stage: {stage}')
    
    # === Load full length dataset ===
    ds_length = xr.open_dataset(os.path.join(path_length, f"length_full_{stage}.nc"))

    # === Create dicts for mass datasets
    mass_vars_full = {}
    mass_vars_mean = {}

    for var in ds_length.data_vars:
        length_data = ds_length[var]

        # -- Full spatial mass
        mass_data = length_to_mass(p, length_data, r)
        var_mass_name = var.replace('length', 'mass')
        mass_vars_full[var_mass_name] = mass_data

        # -- Compute spatial mean time series (only where growth occurs)
        growth_pixels = mass_data.max(dim='days') > mass_data.min(dim='days')
        mean_ts = mass_data.where(growth_pixels).mean(dim=('eta_rho', 'xi_rho'))
        mass_vars_mean[var_mass_name] = mean_ts

    # === Create and save full mass dataset
    ds_mass_full = xr.Dataset(mass_vars_full, attrs=ds_length.attrs)
    ds_mass_full.attrs['description'] = ds_mass_full.attrs.get('description', '').replace('length', 'mass')
    fname_full = f"mass_full_{stage}.nc"
    ds_mass_full.to_netcdf(os.path.join(path_mass, fname_full))
    print(f"Saved full mass dataset: {fname_full}")
    mass_full_by_stage[stage] = ds_mass_full

    # === Create and save mean mass time series
    ds_mass_mean = xr.Dataset(mass_vars_mean, attrs=ds_length.attrs)
    ds_mass_mean.attrs['description'] = 'Mean krill mass time series during MHW and non-MHW periods'
    fname_mean = f"mass_mean_{stage}.nc"
    ds_mass_mean.to_netcdf(os.path.join(path_mass, fname_mean))
    print(f"Saved mean mass dataset: {fname_mean}")
    mass_mean_by_stage[stage] = ds_mass_mean


# %% =========== Print final values ===========
year_index = 36
year = 1980 + year_index

print(f"--- Final day length and mass for {year} ---")

for stage in stage_lengths:
    print(f"\nStage: {stage.capitalize()}")
    
    # Load full length data for the stage
    ds = xr.open_dataset(os.path.join(path_length, f"length_full_{stage}.nc"))
    
    # Initial values
    init_len = stage_lengths[stage]
    init_mass = length_to_mass(p, init_len, r)
    print(f"  Initial length: {init_len:.1f} mm")
    print(f"  Initial mass:   {init_mass:.4f} mg")
    
    # Non-MHW scenario
    length_nonmhw = ds['non_mhw'].isel(years=year_index)
    length_final_nonmhw = length_nonmhw.isel(days=-1)
    growth_pixels_nonmhw = length_nonmhw.max(dim='days') > length_nonmhw.min(dim='days')
    final_length_mean_nonmhw = length_final_nonmhw.where(growth_pixels_nonmhw).mean().item()
    mass_final_nonmhw = length_to_mass(p, length_final_nonmhw, r)
    final_mass_mean_nonmhw = mass_final_nonmhw.where(growth_pixels_nonmhw).mean().item()
    print(f"  Non-MHW final length: {final_length_mean_nonmhw:.2f} mm")
    print(f"  Non-MHW final mass:   {final_mass_mean_nonmhw:.4f} mg")
    
    # Loop over MHW thresholds
    for thresh in mhw_thresholds:
        length_mhw = ds[f'mhw_{thresh}'].isel(years=year_index)
        length_final_mhw = length_mhw.isel(days=-1)
        growth_pixels_mhw = length_mhw.max(dim='days') > length_mhw.min(dim='days')
        final_length_mean_mhw = length_final_mhw.where(growth_pixels_mhw).mean().item()
        mass_final_mhw = length_to_mass(p, length_final_mhw, r)
        final_mass_mean_mhw = mass_final_mhw.where(growth_pixels_mhw).mean().item()
        
        print(f"  MHW {thresh} final length: {final_length_mean_mhw:.2f} mm")
        print(f"  MHW {thresh} final mass:   {final_mass_mean_mhw:.4f} mg")


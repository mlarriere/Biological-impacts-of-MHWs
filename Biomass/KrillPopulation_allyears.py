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


temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) 
temp_avg_100m_SO_noMHWs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_noMHWs.nc')) 

chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 

#%% ====================================================================== 
#           Calculating length for each maturity stage
#       Under different environmental conditions ('worlds')
# ====================================================================== 
# %% ================= Preparing data =================
from Growth_Model.growth_model import length_Atkison2006 
def length_to_mass(length_array, p = 10**(-4.19), r=3.89):
    ''' Constants from Atkinson et al (2006)'''
    mass_array = p*length_array**r
    return mass_array

# --- Parameters
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP accorindg to Tarling et al 2006 - graph under 0°C

# ==== Climatological Drivers: Mean Chla and T°C 
# -- Southern Ocean
temp_clim = temp_avg_100m_SO_allyrs.isel(years=slice(0,30)) #shape: (30, 181, 231, 1442)
temp_clim_mean = temp_clim.mean(dim=['years']) #shape: (181, 231, 1442)
chla_clim = chla_surf_SO_allyrs.isel(years=slice(0,30))
chla_clim_mean = chla_clim.mean(dim=['years'])


# %%  ================= Lengths for each grid cell -- Southern Ocean =================
from functools import partial
from tqdm.contrib.concurrent import process_map

# Function to run years in parallel
def length_single_year(yr, stage, chla_ds, temp_ds):
    chla = chla_ds.chla.isel(years=yr)
    temp = temp_ds.avg_temp.isel(years=yr)

    length = length_Atkison2006(chla=chla, temp=temp, initial_length=stage_lengths[stage], intermoult_period=stage_IMP[stage], maturity_stage=stage)   # → (181, 231, 1442)
    mass = length_to_mass(length)

    return length, mass

def compute_stage(stage):
    print(f"Processing stage: {stage}\n")

    # --------- World1. Climatology
    print('1. Climatology')
    climatological_length = length_Atkison2006(chla=chla_clim_mean.chla, temp=temp_clim_mean.avg_temp, initial_length=stage_lengths[stage], intermoult_period=stage_IMP[stage], maturity_stage=stage) # shape (181, 231, 1442)
    climatological_mass = length_to_mass(climatological_length) # shape (181, 231, 1442)

    # --------- World2. Actual
    print('2. Actual')
    # Run years in parallel
    func = partial(length_single_year, stage=stage, chla_ds=chla_surf_SO_allyrs, temp_ds=temp_avg_100m_SO_allyrs)
    actual_results = process_map(func, range(39), max_workers=10, desc=f"Actual growth | stage={stage}")

    actual_length_list, actual_mass_list = zip(*actual_results)
    actual_length = xr.concat(actual_length_list, dim="years") # shape (39, 181, 231, 1442)
    actual_mass = xr.concat(actual_mass_list, dim="years") # shape (39, 181, 231, 1442)

    # --------- World3. No MHWs 
    print('3. No MHWs')
    # Run years in parallel
    func = partial(length_single_year, stage=stage, chla_ds=chla_surf_SO_allyrs, temp_ds=temp_avg_100m_SO_noMHWs)
    noMHWs_results = process_map(func, range(39), max_workers=10, desc=f"No MHWs growth | stage={stage}")

    noMHWs_length_list, noMHWs_mass_list = zip(*noMHWs_results)
    noMHWs_length = xr.concat(noMHWs_length_list, dim="years") # shape (39, 181, 231, 1442)
    noMHWs_mass = xr.concat(noMHWs_mass_list, dim="years") # shape (39, 181, 231, 1442)

    # --------- World4. Global Warming
    # TODO

    return {"stage": stage,
            "clim_length": climatological_length, "clim_mass": climatological_mass,
            "actual_length": actual_length, "actual_mass": actual_mass,
            "nomhw_length": noMHWs_length, "nomhw_mass": noMHWs_mass,}


# Put to datasets with stages as variables and store to file
stage_list = list(stage_lengths.keys())
def save_if_not_exists(ds, filepath, description=None, units_dict=None):
    if os.path.exists(filepath):
        print(f"File already exists.")
    else:
        # Add description
        if description is not None:
            ds.attrs["description"] = description

        # Add units
        if units_dict is not None:
            for var, unit in units_dict.items():
                if var in ds:
                    ds[var].attrs["units"] = unit

        # Save
        ds.to_netcdf(filepath)
        print(f"Saved: {filepath}")


files = [os.path.join(path_biomass, "fake_worlds/clim_length_stages_SO.nc"),
         os.path.join(path_biomass,   "fake_worlds/clim_mass_stages_SO.nc"),
         os.path.join(path_biomass, "fake_worlds/actual_length_stages_SO.nc"),
         os.path.join(path_biomass,   "fake_worlds/actual_mass_stages_SO.nc"),
         os.path.join(path_biomass, "fake_worlds/noMHWs_length_stages_SO.nc"),
         os.path.join(path_biomass,   "fake_worlds/noMHWs_mass_stages_SO.nc"),]

# Run only if files don't exist
if all(os.path.exists(f) for f in files):
    print("All output files already written. Load data")
    # -- 1. Climatology
    clim_length_ds = xr.open_dataset(files[0])
    clim_mass_ds = xr.open_dataset(files[1])

    # -- 2. Actual
    actual_length_ds = xr.open_dataset(files[2])
    actual_mass_ds = xr.open_dataset(files[3])

    # -- 3. No MHWs world
    noMHWs_length_ds = xr.open_dataset(files[4])
    noMHWs_mass_ds = xr.open_dataset(files[5])
else:
    print("Missing output files → running full processing pipeline...\n")
        
    # --- Call function to compute the different worlds for each stage
    stage_results = process_map(compute_stage, list(stage_lengths.keys()), max_workers=8, desc="Stages") # --> computing time ~50min

    # --- Extract results
    # Initialisation
    clim_length_stages = []
    clim_mass_stages = []
    actual_length_stages = []
    actual_mass_stages = []
    noMHWs_length_stages = []
    noMHWs_mass_stages = []

    # Extract results and put to list
    for res in stage_results:
        clim_length_stages.append(res["clim_length"])
        clim_mass_stages.append(res["clim_mass"])
        actual_length_stages.append(res["actual_length"])
        actual_mass_stages.append(res["actual_mass"])
        noMHWs_length_stages.append(res["nomhw_length"])
        noMHWs_mass_stages.append(res["nomhw_mass"])


    # -- Store results to datasets and save file
    # 1. Climatology
    clim_length_ds = xr.Dataset({f"{stage}": clim_length_stages[i] for i, stage in enumerate(stage_list)})
    clim_mass_ds = xr.Dataset({f"{stage}": clim_mass_stages[i] for i, stage in enumerate(stage_list)})
    save_if_not_exists(clim_length_ds, files[0], description="Climatological Krill Length (30yrs: 1980-2009)", units_dict='mm')
    save_if_not_exists(clim_mass_ds, files[1], description="Climatological Krill Dry Weight (30yrs: 1980-2009)", units_dict='mg')

    # 2. Actual
    actual_length_ds = xr.Dataset({f"{stage}": actual_length_stages[i] for i, stage in enumerate(stage_list)})
    actual_mass_ds = xr.Dataset({f"{stage}": actual_mass_stages[i] for i, stage in enumerate(stage_list)})
    save_if_not_exists(actual_length_ds, files[2], description='Krill Length (1980-2018)', units_dict='mm')
    save_if_not_exists(actual_mass_ds, files[3], description='Krill Dry weight (1980-2018)', units_dict='mg')

    # 3. No MHWs world
    noMHWs_length_ds = xr.Dataset({f"{stage}": noMHWs_length_stages[i] for i, stage in enumerate(stage_list)})
    noMHWs_mass_ds = xr.Dataset({f"{stage}": noMHWs_mass_stages[i] for i, stage in enumerate(stage_list)})
    save_if_not_exists(noMHWs_length_ds, files[4], description="Krill Length without influence of MHWs.\nThe temperature input doesn't contain MHWs (replaced by clim)", units_dict='mm') 
    save_if_not_exists(noMHWs_mass_ds, files[5], description= "Krill Dry weight without influence of MHWs.\nTemperature input don't contain MHWs (replaced by clim)", units_dict='mg')


# %%  ================= Save to Datasets =================
# def create_dataset(stage_lengths, list_length_stages, list_mass_stages, 
#                    description_length="", description_weight=""):
#     # Initialise dict
#     length_vars = {}
#     mass_vars = {}

#     # Unwrapp
#     # for i, stage in enumerate(stage_lengths):
#     i=0
#     stage='juvenile'
#     length_vars[stage] = list_length_stages[i].rename(stage)
#     mass_vars[stage] = list_mass_stages[i].rename(stage)

#     # To dataset
#     length_ds = xr.Dataset(length_vars)
#     mass_ds = xr.Dataset(mass_vars)

#     # Shared metadata
#     shared_coords = {'days': chla_surf_SO_allyrs.days, 'lon_rho': chla_surf_SO_allyrs.lon_rho, 'lat_rho': chla_surf_SO_allyrs.lat_rho,}
#     length_ds = length_ds.assign_coords(shared_coords)
#     mass_ds = mass_ds.assign_coords(shared_coords)

#     # Add extra Metadata
#     length_ds.attrs = {
#         'description': description_length,
#         'time window': 'Growth Season (1st Nov to 30th Apr)',
#         'extent': 'Southern Ocean',
#         'growth_model': 'Atkinson et al., 2006',
#         'created_on': '2025-12-05',
#         'units': 'mm',
#     }

        
#     mass_ds.attrs = {
#         'description': description_weight,
#         'time window': 'Growth Season (1st Nov to 30th Apr)',
#         'extent': 'Southern Ocean',
#         'mass_model': 'Atkinson et al., 2006',
#         'created_on': '2025-12-05',
#         'units': 'mg',
#     }


#     return length_ds, mass_ds


# clim_length_ds, clim_mass_ds = create_dataset(stage_lengths, clim_length_stages, clim_mass_stages, 
#                                               description_length='Climatological Krill Length (30yrs: 1980-2009)', 
#                                               description_weight='Climatological Krill Dry weight (30yrs: 1980-2009)')
# actual_length_ds, actual_mass_ds = create_dataset(stage_lengths, actual_length_stages, actual_mass_stages, 
#                                               description_length='Krill Length (1980-2018)', 
#                                               description_weight='Krill Dry weight (1980-2018)')
# noMHWs_length_ds, noMHWs_mass_ds = create_dataset(stage_lengths, noMHWs_length_stages, noMHWs_mass_stages, 
#                                               description_length="Krill Length without influence of MHWs.\nThe temperature input doesn't contain MHWs (replaced by clim)", 
#                                               description_weight= "Krill Dry weight without influence of MHWs.\nTemperature input don't contain MHWs (replaced by clim)")

# # -- Save to file
# fname = f"clim_length_traj_mat_stages_SO.nc"
# clim_length_ds.to_netcdf(os.path.join(path_length, fname))
# print(f"Saved mass dataset: {fname}")

# fname = f"clim_mass_traj_mat_stages_SO.nc"
# clim_mass_ds.to_netcdf(os.path.join(path_mass, fname))
# print(f"Saved mass dataset: {fname}")

#%% ====================================================================== 
# Calculating length for each maturity stage YEARLY 
# ====================================================================== 
# %%  ================= Southern Ocean =================
# # == Lengths traj for each year and grid cell -- shape (39, 181, 231, 360)
# years = np.arange(1980, 2019)
# year_indices = years - 1980  # 0 to 38

# lengths_by_stage = {stage: [] for stage in stage_lengths}
# mass_by_stage = {stage: [] for stage in stage_lengths}


# for yr in year_indices:
#     print(yr+1980)
#     for stage in stage_lengths:
#         print(stage)
#         length_yr = length_Atkison2006(chla=chla_surf_SO_allyrs.chla.isel(years=yr), 
#                                         temp=temp_avg_100m_SO_allyrs.avg_temp.isel(years=yr), 
#                                         initial_length=stage_lengths[stage], 
#                                         intermoult_period=stage_IMP[stage],
#                                         maturity_stage=stage)
#         # --- Check shape and fix if necessary ---
#         if length_yr.dims != ('days', 'eta_rho', 'xi_rho'):
#             length_yr = length_yr.transpose('days', 'eta_rho', 'xi_rho')

#         # Add coordinate year
#         length_yr = length_yr.expand_dims(dim={'years': [yr + 1980]})

#         # Converting to mass
#         mass_yr = length_to_mass(length_yr)   

#         # Store rsults
#         lengths_by_stage[stage].append(length_yr)
#         mass_by_stage[stage].append(mass_yr)

# length_datasets = {}
# mass_datasets = {}
# for stage in stage_lengths:
#     # Concatenate
#     length_stack = xr.concat(lengths_by_stage[stage], dim='years')
#     mass_stack = xr.concat(mass_by_stage[stage], dim='years')
    
#     # Adding coords
#     length_stack = length_stack.assign_coords(years=years)
#     mass_stack = mass_stack.assign_coords(years=years)

#     length_datasets[stage] = length_stack
#     mass_datasets[stage] = mass_stack

# # -- To Dataset
# krill_length_ds = xr.Dataset(length_datasets)
# krill_mass_ds = xr.Dataset(mass_datasets)

# # Metadata
# shared_coords = {
#     'days': chla_surf_SO_allyrs.days,
#     'eta_rho': chla_surf_SO_allyrs.eta_rho,
#     'xi_rho': chla_surf_SO_allyrs.xi_rho,
#     'lon_rho': chla_surf_SO_allyrs.lon_rho,
#     'lat_rho': chla_surf_SO_allyrs.lat_rho,
# }
# krill_length_ds = krill_length_ds.assign_coords(shared_coords)
# krill_mass_ds = krill_mass_ds.assign_coords(shared_coords)


# krill_length_ds.attrs = {
#     'title': 'Krill Length Trajectories (1980–2018)',
#     'description': 'Daily krill body lengths for different maturity stages using Atkinson (2006) growth model',
#     'time window': 'Growth Season (1st Nov to 30th Apr)',
#     'extent': 'Southern Ocean',
#     'growth_model': 'Atkinson et al., 2006',
#     'created_on': '2025-08-06',
#     'units': 'mm',
# }

# krill_mass_ds.attrs = {
#     'title': 'Krill Dry Mass Trajectories (1980–2018)',
#     'description': 'Dry weight estimates derived from krill length for different maturity stages',
#     'time window': 'Growth Season (1st Nov to 30th Apr)',
#     'extent': 'Southern Ocean',
#     'mass_model': 'Atkinson et al., 2006',
#     'created_on': '2025-08-06',
#     'units': 'mg',
# }

# # -- Save to file
# fname = f"length_traj_mat_stages_SO.nc"
# krill_length_ds.to_netcdf(os.path.join(path_length, fname))
# print(f"Saved mass dataset: {fname}")

# fname = f"mass_traj_mat_stages_SO.nc"
# krill_mass_ds.to_netcdf(os.path.join(path_mass, fname))
# print(f"Saved mass dataset: {fname}")




# temp_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/temp_avg100m_daily_mhw.nc")
# temp_mhw = xr.open_dataset(temp_mhw_file)

# # ==== Calculate mean length trajectories
# length_daily_trajectories_all = {}
# stage_daily_growths_all = {}
# growth_by_stage_and_mhw_all = {}
# length_by_stage_and_mhw_all={}

# for yr_index in year_indices: 
#     # yr_index = 0
#     print(f'Processing {yr_index+1980}') 
#     # -- Select drivers
#     temp = temp_avg_100m_study_area_allyrs.isel(years=yr_index)
#     chla = chla_surf_study_area_allyrs.isel(years=yr_index)

#     # Initalize dictionnaries
#     length_daily_trajectories = {}
#     stage_daily_growths = {}
#     growth_by_stage_and_mhw = {}

#     # -- Calculate length Daily and associated growth increments
#     for stage in stage_lengths:
#         length_stage_daily = length_Atkison2006(chla=chla.chla, temp=temp.avg_temp, 
#                                                 initial_length=stage_lengths[stage], 
#                                                 intermoult_period=1, 
#                                                 maturity_stage=stage) #shape (181, 231, 360)
#         growth = length_stage_daily.diff(dim='days') #shape (180, 231, 360)

#         # Store results
#         length_daily_trajectories[stage] = length_stage_daily
#         stage_daily_growths[stage] = growth

#     # -- Defining MHWs scenarios
#     mhw_1 = xr.where(~np.isnan(temp_mhw.temp_1deg.isel(years=yr_index)), 1, 0)
#     mhw_2 = xr.where(~np.isnan(temp_mhw.temp_2deg.isel(years=yr_index)), 1, 0)
#     mhw_3 = xr.where(~np.isnan(temp_mhw.temp_3deg.isel(years=yr_index)), 1, 0)
#     mhw_4 = xr.where(~np.isnan(temp_mhw.temp_4deg.isel(years=yr_index)), 1, 0)

#     for stage, daily_growth in stage_daily_growths.items():
#         growth_by_mhw = {1: daily_growth.where(mhw_1 == 1),
#                          2: daily_growth.where(mhw_2 == 1),
#                          3: daily_growth.where(mhw_3 == 1),
#                          4: daily_growth.where(mhw_4 == 1),
#                          0: daily_growth.where((mhw_1 + mhw_2 + mhw_3 + mhw_4) == 0)}
        
#         # Store results 
#         growth_by_stage_and_mhw[stage] = growth_by_mhw #shape (180, 231, 360)

#     # Store per year
#     length_daily_trajectories_all[yr_index+1980] = length_daily_trajectories
#     stage_daily_growths_all[yr_index+1980] = stage_daily_growths
#     growth_by_stage_and_mhw_all[yr_index+1980] = growth_by_stage_and_mhw

#     # -- Length trajectories under the different scenarios
#     n_days = 181
#     length_by_stage_and_mhw = {}
    
#     for stage in stage_lengths:
#         initial_length = stage_lengths[stage]
#         intermoult_period = stage_IMP[stage]
#         length_by_mhw_level = {}

#         for level in range(5):
#             growth = growth_by_stage_and_mhw[stage][level]  # (days, eta_rho, xi_rho)

#             # 1. Mean growth across space for each day
#             daily_mean_growth = growth.mean(dim=["eta_rho", "xi_rho"], skipna=True)

#             # 2. Intermoult-block logic with last valid step growth
#             growth_blocks = []
#             for i in range(0, n_days, intermoult_period):
#                 block = daily_mean_growth.isel(days=slice(i, min(i + intermoult_period, n_days)))
                
#                 # Take last valid growth value in the block, or 0 if none
#                 valid_growth = block.dropna(dim='days')
#                 if valid_growth.size > 0:
#                     block_growth = valid_growth[-1].item()
#                 else:
#                     block_growth = 0.0
#                 growth_blocks.extend([block_growth] * len(block))

#             # 3. Length trajectory
#             length_series = [initial_length]
#             current_length = initial_length
#             for i in range(1, n_days):
#                 if i % intermoult_period == 0:
#                     current_length += growth_blocks[i - 1]
#                 length_series.append(current_length)

#             # 4. Store as DataArray
#             length_by_mhw_level[level] = xr.DataArray(data=length_series,
#                                                       dims=["days"],
#                                                       coords={"days": np.arange(181)}) 

#         # Save all MHW levels for this stage
#         length_by_stage_and_mhw[stage] = length_by_mhw_level

#     length_by_stage_and_mhw_all[yr_index+1980] = length_by_stage_and_mhw


# # %% ===== Save to Dataset =====
# # --- Initialize dictionaries
# datasets_mean_by_stage = {}

# for stage in stage_lengths: # 1 dataset for each maturity stage    
#     data_vars = {}
#     for level in range(5):
#         values = []
#         for yr in years:
#             da = length_by_stage_and_mhw_all[yr][stage][level]
#             values.append(da.values)  # shape: (181,)
#         stacked = np.stack(values, axis=0)  # shape: (n_years, 181)
#         data_vars[f"length_cat{level}"] = (("years", "days"), stacked)

#     # Create dataset
#     ds = xr.Dataset(data_vars=data_vars,
#                     coords={"years": years, "days": np.arange(181)})    
#     ds.attrs.update({
#         'description': 'Mean krill length time series during MHW and non-MHW.',
#         'maturity_stage': stage,
#         'temporal_window': 'Growth season (daily steps)',
#         'category': '0: Non MHWs, 1: 1°C MHW, 2: 2°C MHW, ...',
#         'units': 'mm',
#     })
    
#     datasets_mean_by_stage[stage] = ds

# # --- Save to file
# for stage, ds_mean in datasets_mean_by_stage.items():
#     fname = f"length_{stage}.nc"
#     ds_mean.to_netcdf(os.path.join(path_length, fname))
#     print(f"Saved mean dataset: {fname}")


# # %% ============== Mass evolution for all years ==============
# def length_to_mass(p, length_array, r):
#     mass_array = p*length_array**r
#     return mass_array

# # --- Defining constants - Accroding to mass length coefficient of Atkison et al (2006)
# p = 10**(-4.19)
# r = 3.89

# datasets_mass_by_stage = {}
# for stage in stage_lengths:
#     ds_length = datasets_mean_by_stage[stage]
#     data_vars_mass = {}

#     for level in range(5):
#         length_data = ds_length[f"length_cat{level}"].values  # shape: (n_years, 181)
#         mass_data = length_to_mass(p, length_data, r)         # same shape
#         data_vars_mass[f"mass_cat{level}"] = (("years", "days"), mass_data)

#     # Create dataset
#     ds_mass = xr.Dataset(data_vars=data_vars_mass,
#                          coords={"years": years, "days": np.arange(181)})
#     ds_mass.attrs.update({
#         'description': 'Mean krill mass time series during MHW and non-MHW.',
#         'maturity_stage': stage,
#         'temporal_window': 'Growth season (daily steps)',
#         'category': '0: Non MHWs, 1: 1°C MHW, 2: 2°C MHW, ...',
#         'units': 'mg dry mass',
#         'conversion_equation': 'mass = p * length^r',
#         'p': p,
#         'r': r,
#     })

#     datasets_mass_by_stage[stage] = ds_mass

# # -- Save to file
# for stage, ds_mass in datasets_mass_by_stage.items():
#     fname = f"mass_{stage}.nc"
#     ds_mass.to_netcdf(os.path.join(path_mass, fname))
#     print(f"Saved mass dataset: {fname}")



# # %% =========== Print final values for selected years ===========
# selected_years = [2016]
# selected_indices = [yr - 1980 for yr in selected_years]

# print(f"--- Final day krill length and mass for years: {selected_years} ---")

# for stage in stage_lengths:
#     print(f"\nStage: {stage.capitalize()}")

#     # Load saved datasets
#     ds_length = xr.open_dataset(os.path.join(path_length, f"length_{stage}.nc"))
#     ds_mass = xr.open_dataset(os.path.join(path_mass, f"mass_{stage}.nc"))

#     for yr, yr_idx in zip(selected_years, selected_indices):
#         print(f"\n  Year: {yr}")
#         for level in range(5):
#             length_final = ds_length[f"length_cat{level}"].isel(years=yr_idx, days=-1).item()
#             mass_final = ds_mass[f"mass_cat{level}"].isel(years=yr_idx, days=-1).item()
#             print(f"    MHW {level}: Final length = {length_final:.2f} mm, Final mass = {mass_final:.4f} mg")

# # %%

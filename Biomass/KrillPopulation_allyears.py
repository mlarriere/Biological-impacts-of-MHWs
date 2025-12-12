"""
Created on Tue 31 July 15:06:14 2025

Length-Mass relationship

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import sys
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


# %% ================= Load data =================
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) 
temp_avg_100m_SO_noMHWs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_noMHWs.nc')) 
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 

# ==== Climatological Drivers: Mean Chla and T°C 
# -- Southern Ocean
temp_clim = temp_avg_100m_SO_allyrs.isel(years=slice(0,30)) #shape: (30, 181, 231, 1442)
temp_clim_mean = temp_clim.mean(dim=['years']) #shape: (181, 231, 1442)
chla_clim = chla_surf_SO_allyrs.isel(years=slice(0,30))
chla_clim_mean = chla_clim.mean(dim=['years'])

#%% ====================================================================== 
#           Calculating length for each maturity stage
#       Under different environmental conditions ('worlds')
# ====================================================================== 
# %% ================= Settings =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Growth_Model.growth_model import length_Atkison2006 

# --- Parameters
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
# stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP accorindg to Tarling et al 2006 - graph under 0°C
stage_IMP = {'juvenile': 1, 'immature': 1, 'mature': 1, 'gravid': 1} # Remove IMP as temperature dependent (not constant)

def length_to_mass(length_array, p = 10**(-4.19), r=3.89):
    ''' Constants from Atkinson et al (2006)'''
    mass_array = p*length_array**r
    return mass_array

# %%  ================================== Functions to run length and mass under different scenarios for krill population ==================================
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
def save_if_not_exists(ds, filepath, description=None, units=None):
    if os.path.exists(filepath):
        print(f"File already exists.")
    else:
        # Add description
        if description is not None:
            ds.attrs["description"] = description

        # Add units
        if units is not None:
            ds.attrs["units"] = units
        
        ds.attrs["date"] = '08 December 2025'

        # Save
        ds.to_netcdf(filepath)
        print(f"Saved: {filepath}")

# %%  ================================== Krill Population Southern Ocean ==================================
files = [os.path.join(path_biomass, "fake_worlds/clim_length_stages_SO_noIMP.nc"),
         os.path.join(path_biomass,   "fake_worlds/clim_mass_stages_SO_noIMP.nc"),
         os.path.join(path_biomass, "fake_worlds/actual_length_stages_SO_noIMP.nc"),
         os.path.join(path_biomass,   "fake_worlds/actual_mass_stages_SO_noIMP.nc"),
         os.path.join(path_biomass, "fake_worlds/noMHWs_length_stages_SO_noIMP.nc"),
         os.path.join(path_biomass,   "fake_worlds/noMHWs_mass_stages_SO_noIMP.nc"),]

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
    stage_results = process_map(compute_stage, list(stage_lengths.keys()), max_workers=8, desc="Stages") # --> computing time ~20min

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
    save_if_not_exists(clim_length_ds, files[0], description="Climatological Krill Length (30yrs: 1980-2009)", units='mm')
    save_if_not_exists(clim_mass_ds, files[1], description="Climatological Krill Dry Weight (30yrs: 1980-2009)", units='mg')

    # 2. Actual
    actual_length_ds = xr.Dataset({f"{stage}": actual_length_stages[i] for i, stage in enumerate(stage_list)})
    actual_mass_ds = xr.Dataset({f"{stage}": actual_mass_stages[i] for i, stage in enumerate(stage_list)})
    save_if_not_exists(actual_length_ds, files[2], description='Krill Length (1980-2018)', units='mm')
    save_if_not_exists(actual_mass_ds, files[3], description='Krill Dry weight (1980-2018)', units='mg')

    # 3. No MHWs world
    noMHWs_length_ds = xr.Dataset({f"{stage}": noMHWs_length_stages[i] for i, stage in enumerate(stage_list)})
    noMHWs_mass_ds = xr.Dataset({f"{stage}": noMHWs_mass_stages[i] for i, stage in enumerate(stage_list)})
    save_if_not_exists(noMHWs_length_ds, files[4], description="Krill Length without influence of MHWs.\nThe temperature input doesn't contain MHWs (replaced by clim)", units='mm') 
    save_if_not_exists(noMHWs_mass_ds, files[5], description= "Krill Dry weight without influence of MHWs.\nTemperature input don't contain MHWs (replaced by clim)", units='mg')



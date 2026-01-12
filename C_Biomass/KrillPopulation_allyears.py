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
from datetime import datetime
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
path_trend = os.path.join(path_biomass, f'fake_worlds/trends')


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


# ==== Detrended signal
detrended_temp_seasons_file = os.path.join(path_biomass, f'surrogates/detrended_signal/temp_detrended_seasonal.nc')
detrended_temp = xr.open_dataset(detrended_temp_seasons_file) #shape (40, 181, 231, 1442)


#%% ====================================================================== 
#           Calculating length for each maturity stage
#       Under different environmental conditions ('worlds')
# ====================================================================== 
# %% ================= Settings =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from B_Growth_Model.Atkinson2006_model import length_Atkinson2006 

# --- Parameters
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}

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

    length = length_Atkinson2006(chla=chla, temp=temp, initial_length=stage_lengths[stage], maturity_stage=stage)   # → (181, 231, 1442)
    mass = length_to_mass(length)

    return length, mass


def compute_world(stage, chla_ds, temp_ds, label, max_workers=4):
    # test
    # stage='juvenile'
    # chla_ds=chla_surf_SO_allyrs
    # temp_ds=temp_avg_100m_SO_allyrs
    # label='actual'
    # max_workers=4
    
    print(f"  → {label}")

    # COmpute length and mass - results=list of length 39
    func = partial(length_single_year, stage=stage, chla_ds=chla_ds, temp_ds=temp_ds)
    results = process_map(func, range(39), max_workers=max_workers,
                          chunksize=1, desc=f"{label} | {stage}")

    # Initialisation
    nyears, nday, neta, nxi = 39, 181, 231, 1442
    length_out_da = xr.DataArray(
        np.empty((nyears, nday, neta, nxi), dtype=np.float32),
        dims=("years", "days", "eta_rho", "xi_rho"),
        coords={
            "years": np.arange(nyears),
            "lat": chla_ds.lat_rho,
            "lon": chla_ds.lon_rho,
        },
    )

    mass_out_da = xr.DataArray(
        np.empty((nyears, nday, neta, nxi), dtype=np.float32),
        dims=("years", "days", "eta_rho", "xi_rho"),
        coords=length_out_da.coords,
    )

    for y, (length, mass) in enumerate(results):
        length_out_da[y, :, :, :] = length.values.astype(np.float32)
        mass_out_da[y, :, :, :]   = mass.values.astype(np.float32)

        del length, mass

    
    return length_out_da, mass_out_da

def compute_stage(stage):
    # stage='juvenile'

    # ---------- World 1: Climatology
    print("  → Climatology")
    clim_length = length_Atkinson2006(chla=chla_clim_mean.chla, temp=temp_clim_mean.avg_temp,
                                     initial_length=stage_lengths[stage], maturity_stage=stage)
    clim_mass = length_to_mass(clim_length)

    # ---------- World 2: Actual
    actual_length, actual_mass = compute_world(stage, chla_surf_SO_allyrs, temp_avg_100m_SO_allyrs, label="Actual", max_workers=4)

    # ---------- World 3: No MHWs
    nomhw_length, nomhw_mass = compute_world(stage, chla_surf_SO_allyrs, temp_avg_100m_SO_noMHWs, label="No MHWs", max_workers=4)

    # ---------- World 4: No Warming
    warming_length, warming_mass = compute_world(stage, chla_surf_SO_allyrs, detrended_temp, label="No Warming", max_workers=4)

    return {"stage": stage,
            "clim_length": clim_length, "clim_mass": clim_mass,
            "actual_length": actual_length, "actual_mass": actual_mass,
            "nomhw_length": nomhw_length, "nomhw_mass": nomhw_mass,
            "warming_length": warming_length, "warming_mass": warming_mass,
            }


# Put to datasets with stages as variables and store to file
stage_list = list(stage_lengths.keys())
def save_if_not_exists(ds, filepath, description=None, units=None):
    # create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # if the file exists, remove it 
    if os.path.exists(filepath):
        print(f"Removing existing file: {filepath}")
        os.remove(filepath)

    # Metadata
    if description is not None:
        ds.attrs["description"] = description
    if units is not None:
        ds.attrs["units"] = units
    ds.attrs["date"] = datetime.today().strftime('%d %B %Y')

    # Write
    ds.to_netcdf(filepath, mode="w", engine="netcdf4")
    print(f"Saved: {filepath}")

# %%  ================================== Krill Population Southern Ocean ==================================
path_biomass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_surrogates_krill = os.path.join(path_biomass, f'surrogates/mass_length')

files = [os.path.join(path_surrogates_krill, "clim_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "clim_mass_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "actual_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "actual_mass_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "noMHWs_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "noMHWs_mass_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "warming_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "warming_mass_stages_SO.nc"),]

# Run only if files don't exist
if all(os.path.exists(f) for f in files):
    print("All output files already written. Load data")
    # -- 1. Climatology
    clim_length_ds = xr.open_dataset(files[0])
    clim_mass_ds = xr.open_dataset(files[1])

    # -- 2. Actual
    actual_length_ds = xr.open_dataset(files[2])
    actual_mass_ds = xr.open_dataset(files[3])

    # -- 3. No MHWs
    noMHWs_length_ds = xr.open_dataset(files[4])
    noMHWs_mass_ds = xr.open_dataset(files[5])

    # 4. No Global Warming
    nowarming_length_ds = xr.open_dataset(files[6])
    nowarming_mass_ds = xr.open_dataset(files[7])

else:
    print("Missing output files → running full processing pipeline...\n")
    
    # Initialisation
    clim_length_ds = xr.Dataset()
    clim_mass_ds = xr.Dataset()
    actual_length_ds = xr.Dataset()
    actual_mass_ds = xr.Dataset()
    noMHWs_length_ds = xr.Dataset()
    noMHWs_mass_ds = xr.Dataset()
    nowarming_length_ds = xr.Dataset()
    nowarming_mass_ds = xr.Dataset()

    # Loop over matury stage and compute surrogates (computing time ~30min for each stage)
    for stage in stage_lengths:
        # Test
        # stage='juvenile'

        print(f"\n=== Stage: {stage} ===")

        # --- Call function to compute the different worlds for each stage (~15min per stage)
        res = compute_stage(stage)

        # --- Extract results
        clim_length_ds[stage] = res["clim_length"]
        clim_mass_ds[stage]   = res["clim_mass"]

        actual_length_ds[stage] = res["actual_length"]
        actual_mass_ds[stage]   = res["actual_mass"]

        noMHWs_length_ds[stage] = res["nomhw_length"]
        noMHWs_mass_ds[stage]   = res["nomhw_mass"]

        nowarming_length_ds[stage] = res["warming_length"]
        nowarming_mass_ds[stage]   = res["warming_mass"]

    # -- Store results to datasets and save file
    # 1. Climatology
    save_if_not_exists(clim_length_ds, files[0], description="Climatological Krill Length (30yrs: 1980-2009).\n" \
                                                              "Length calculated using the Daily Growth Rate (daily time scale) from Atkinson et al. 2006 (model 4)", units='mm')
    save_if_not_exists(clim_mass_ds, files[1], description="Climatological Krill Dry Weight (30yrs: 1980-2009)\n" \
                                                              "Weight conversion from Atkinson et al. 2006 ('All stage' configuration)", 
                                                              units='mg')

    # 2. Actual
    save_if_not_exists(actual_length_ds, files[2], description='Krill Length (1980-2018)', units='mm')
    save_if_not_exists(actual_mass_ds, files[3], description='Krill Dry weight (1980-2018)', units='mg')

    # 3. No MHWs
    save_if_not_exists(noMHWs_length_ds, files[4], description="Krill Length without influence of MHWs.\nThe temperature input doesn't contain MHWs (replaced by clim)", units='mm') 
    save_if_not_exists(noMHWs_mass_ds, files[5], description= "Krill Dry weight without influence of MHWs.\nTemperature input don't contain MHWs (replaced by clim)", units='mg')

    # 4. No Global Warming
    save_if_not_exists(nowarming_length_ds, files[6], description="Krill Length without influence of global warming.\nThe temperature input is detrended.", units='mm') 
    save_if_not_exists(nowarming_mass_ds, files[7], description= "Krill Dry weight without influence of global warming.\nTemperature input is detrended", units='mg')




# %%

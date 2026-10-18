"""
Created on Tue 31 July 15:06:14 2025

Length-Mass relationship

@author: Marguerite Larriere (mlarriere)
"""

# %% -------------------------------- Packages ------------------------------------
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
path_surrogates = os.path.join(path_biomass, 'surrogates')
path_trend = os.path.join(path_surrogates, f'detrended_signal')


# %% ================= Load data =================
# ==== Actual Conditions
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) 
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 

# ==== Climatological Drivers
temp_clim = temp_avg_100m_SO_allyrs.isel(years=slice(0,30)) #shape: (30, 181, 231, 1442)
temp_clim_mean = temp_clim.mean(dim=['years']) #shape: (181, 231, 1442)
chla_clim = chla_surf_SO_allyrs.isel(years=slice(0,30))
chla_clim_mean = chla_clim.mean(dim=['years'])

# ... adding years dimension
chla_clim_expanded = chla_clim_mean.expand_dims({'years': chla_surf_SO_allyrs.years})
temp_clim_expanded = temp_clim_mean.expand_dims({'years': chla_surf_SO_allyrs.years})

# ==== No MHWs signal
temp_avg_100m_SO_noMHWs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_noMHWs.nc')) #signal where temp is set to clim when MHWs happend
temp_clim_trended = xr.open_dataset(os.path.join(path_trend, 'clim_with_trend.nc')) #clim with warming trend 

# ==== Detrended signal
detrended_temp_seasons_file = os.path.join(path_biomass, f'surrogates/detrended_signal/temp_detrended_seasonal.nc')
detrended_temp = xr.open_dataset(detrended_temp_seasons_file) #shape (40, 181, 231, 1442)


#%% ====================================================================== 
#           Calculating length for each maturity stage
#       Under different environmental conditions ('worlds')
# ====================================================================== 
# %% ================= Krill Settings =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from B_Growth_Model.Atkinson2006_model import length_Atkinson2006 

import importlib
import functions_length
importlib.reload(functions_length)

from functions_length import (
    STAGE_LENGTHS,
    STAGE_LIST,
    length_to_mass,
    length_single_year,
    compute_world,
    compute_stage,
    save_if_not_exists,
)

from functools import partial
from tqdm.contrib.concurrent import process_map

stage_lengths = STAGE_LENGTHS
stage_list = STAGE_LIST

WORLDS = {
    "clim": (chla_clim_expanded, temp_clim_expanded, "Climatology", 10,),
    "actual": (chla_surf_SO_allyrs, temp_avg_100m_SO_allyrs, "Actual", 10,),
    "clim_trended": (chla_surf_SO_allyrs, temp_clim_trended, "Clim Trended", 10,),
    "no_warming": (chla_surf_SO_allyrs, detrended_temp, "No Warming", 10,),
}


# # --- Parameters
# stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}

# def length_to_mass(length_array, p = 10**(-4.19), r=3.89):
#     ''' Constants from Atkinson et al (2006)'''
#     mass_array = p*length_array**r #in mg
#     return mass_array

# %%  ================================== Functions to run length and mass under different scenarios for krill population ==================================

# Function to run years in parallel
# def length_single_year(yr, stage, chla_ds, temp_ds, sensitivity_param=None, sensitivity=0):
#     # yr=0
#     chla = chla_ds.chla.isel(years=yr)
#     temp = temp_ds.avg_temp.isel(years=yr)

#     length = length_Atkinson2006(chla=chla, temp=temp, initial_length=stage_lengths[stage], maturity_stage=stage, sensitivity_param=sensitivity_param, sensitivity=sensitivity) #(181, 231, 1442)
#     mass = length_to_mass(length)

#     return length, mass


# def compute_world(stage, chla_ds, temp_ds, label, max_workers=4, sensitivity_param=None, sensitivity=0):
#     # test
#     # stage='juvenile'
#     # chla_ds=chla_surf_SO_allyrs
#     # temp_ds=temp_avg_100m_SO_allyrs
#     # label='actual'
#     # max_workers=4
    
#     # print(f"  → {label}")
#     print(f"  → {label} | "
#         f"{sensitivity_param or 'baseline'} | {sensitivity:+.0%}")

#     # COmpute length and mass - results=list of length 39
#     func = partial(length_single_year, stage=stage, chla_ds=chla_ds, temp_ds=temp_ds, sensitivity_param=sensitivity_param, sensitivity=sensitivity)
#     results = process_map(func, range(39), max_workers=max_workers, chunksize=1, desc=f"{label} | {stage}")

#     # Initialisation
#     nyears, nday, neta, nxi = 39, 181, 231, 1442
#     length_out_da = xr.DataArray(np.empty((nyears, nday, neta, nxi), dtype=np.float32),
#                                  dims=("years", "days", "eta_rho", "xi_rho"),
#                                  coords={"years": np.arange(nyears),
#                                          "lat": chla_ds.lat_rho,
#                                          "lon": chla_ds.lon_rho,},)

#     mass_out_da = xr.DataArray(np.empty((nyears, nday, neta, nxi), dtype=np.float32),
#                                dims=("years", "days", "eta_rho", "xi_rho"),
#                                coords=length_out_da.coords,)

#     for y, (length, mass) in enumerate(results):
#         length_out_da[y, :, :, :] = length.values.astype(np.float32)
#         mass_out_da[y, :, :, :]   = mass.values.astype(np.float32)

#         del length, mass

#     return length_out_da, mass_out_da


# def compute_stage(stage, sensitivity_param=None, sensitivity=0):
#     # stage='juvenile'

#     # ---------- World 1: Climatology
#     print("  → Climatology")
#     clim_length, clim_mass = compute_world(stage, chla_clim_expanded, temp_clim_expanded, label="Climatology", max_workers=10, sensitivity_param=sensitivity_param, sensitivity=sensitivity)

#     # ---------- World 2: Actual
#     actual_length, actual_mass = compute_world(stage, chla_surf_SO_allyrs, temp_avg_100m_SO_allyrs, label="Actual", max_workers=10, sensitivity_param=sensitivity_param, sensitivity=sensitivity)

#     # ---------- World 3: No MHWs
#     clim_trended_length, clim_trended_mass = compute_world(stage, chla_surf_SO_allyrs, temp_clim_trended, label="Clim Trended", max_workers=10, sensitivity_param=sensitivity_param, sensitivity=sensitivity)

#     # ---------- World 4: No Warming
#     warming_length, warming_mass = compute_world(stage, chla_surf_SO_allyrs, detrended_temp, label="No Warming", max_workers=10, sensitivity_param=sensitivity_param, sensitivity=sensitivity)

#     return {"stage": stage,
#             "clim_length": clim_length, "clim_mass": clim_mass,
#             "actual_length": actual_length, "actual_mass": actual_mass,
#             "clim_trended_length": clim_trended_length, "clim_trended_mass": clim_trended_mass,
#             "warming_length": warming_length, "warming_mass": warming_mass,
            # }





# def compute_stage(stage, sensitivity_param=None, sensitivity=0):
#     results = {}

#     for world_key, (chla_ds, temp_ds, label, workers) in WORLDS.items():

#         length, mass = compute_world(stage=stage, chla_ds=chla_ds, temp_ds=temp_ds, label=label, max_workers=workers,
#                                      sensitivity_param=sensitivity_param, sensitivity=sensitivity,)

#         results[f"{world_key}_length"] = length
#         results[f"{world_key}_mass"] = mass

#     return results


# # Put to datasets with stages as variables and store to file
# stage_list = list(stage_lengths.keys())
# def save_if_not_exists(ds, filepath, description=None, temperature=None, chla = None, units=None):
#     # create directory if needed
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)

#     # if the file exists, remove it 
#     if os.path.exists(filepath):
#         print(f"Removing existing file: {filepath}")
#         os.remove(filepath)

#     # Metadata
#     if description is not None:
#         ds.attrs["description"] = description
#     if units is not None:
#         ds.attrs["units"] = units
#     if temperature is not None:
#         ds.attrs["temperature"] = temperature
#     if chla is not None:
#         ds.attrs["chla"] = chla
#     ds.attrs["date"] = datetime.today().strftime('%d %B %Y')

#     # Write
#     ds.to_netcdf(filepath, mode="w", engine="netcdf4")
#     print(f"Saved: {filepath}")

# %%  ================================== Krill Population Southern Ocean ==================================
path_surrogates_krill = os.path.join(path_surrogates, f'mass_length')

files = [os.path.join(path_surrogates_krill, "clim_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "clim_mass_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "actual_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "actual_mass_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "clim_trended_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "clim_trended_mass_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "nowarming_length_stages_SO.nc"),
         os.path.join(path_surrogates_krill, "nowarming_mass_stages_SO.nc"),]

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
    clim_trended_length_ds = xr.open_dataset(files[4])
    clim_trended_mass_ds = xr.open_dataset(files[5])

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
    clim_trended_length_ds = xr.Dataset()
    clim_trended_mass_ds = xr.Dataset()
    nowarming_length_ds = xr.Dataset()
    nowarming_mass_ds = xr.Dataset()

    # Loop over matury stage and compute surrogates (computing time ~30min for each stage)
    for stage in stage_lengths:
        # Test
        # stage='juvenile'

        print(f"\n=== Stage: {stage} ===")

        # --- Call function to compute the different worlds for each stage (~15min per stage)
        res = compute_stage(stage, WORLDS)

        # --- Extract results
        clim_length_ds[stage] = res["clim_length"]
        clim_mass_ds[stage]   = res["clim_mass"]

        actual_length_ds[stage] = res["actual_length"]
        actual_mass_ds[stage]   = res["actual_mass"]

        clim_trended_length_ds[stage] = res['clim_trended_length']
        clim_trended_mass_ds[stage] = res['clim_trended_mass']

        nowarming_length_ds[stage] = res["no_warming_length"]
        nowarming_mass_ds[stage]   = res["no_warming_mass"]

    # -- Store results to datasets and save file
    # 1. Climatology
    save_if_not_exists(clim_length_ds, files[0], description="Climatological Krill Length (30yrs: 1980-2009).\n" \
                                                             "Length calculated using the Daily Growth Rate (daily time scale) from Atkinson et al. 2006 (model 4)", 
                                                 temperature = 'Climatological temperature (growth seasons 30years avg).',
                                                 chla = 'Climatological Chla (growth seasons 30years avg).',
                                                 units='mm')
    save_if_not_exists(clim_mass_ds, files[1], description="Climatological Krill Dry Weight (30yrs: 1980-2009)\n" \
                                                           "Weight conversion from Atkinson et al. 2006 ('All stage' configuration)", 
                                               temperature = 'Climatological temperature (growth seasons 30years avg).',
                                               chla = 'Climatological Chla (growth seasons 30years avg).',
                                               units='mg')

    # 2. Actual
    save_if_not_exists(actual_length_ds, files[2], description='Krill Length (1980-2018)', 
                                                   temperature = 'Actual temperature (growth seasons daily).',
                                                   chla = 'Actual Chla (growth seasons daily).',
                                                   units='mm')
    save_if_not_exists(actual_mass_ds, files[3], description='Krill Dry weight (1980-2018)', 
                                                 temperature = 'Actual temperature (growth seasons daily).',
                                                 chla = 'Actual Chla (growth seasons daily).',
                                                 units='mg')

    # 3. No MHWs
    save_if_not_exists(clim_trended_length_ds, files[4], description="Krill Length without influence of MHWs.", 
                                                         temperature = 'Climatological temperature trended with warming trend (growth seasons daily).',
                                                         chla = 'Climatological Chla (growth seasons daily).',
                                                         units='mm') 
    save_if_not_exists(clim_trended_mass_ds, files[5], description= "Krill Dry weight without influence of MHWs.", 
                                                       temperature = 'Climatological temperature trended with warming trend (growth seasons daily).',
                                                       chla = 'Climatological Chla (growth seasons daily).',
                                                       units='mg')

    # 4. No Global Warming
    save_if_not_exists(nowarming_length_ds, files[6], description="Krill Length without influence of global warming.", 
                                                      temperature = 'Actual temperature detrended, i.e. warming trend removed (growth seasons daily).',
                                                      chla = 'Actual Chla (growth seasons daily).',
                                                      units='mm') 
    save_if_not_exists(nowarming_mass_ds, files[7], description= "Krill Dry weight without influence of global warming.", 
                                                     temperature = 'Actual temperature detrended, i.e. warming trend removed (growth seasons daily).',
                                                     chla = 'Actual Chla (growth seasons daily).',
                                                     units='mg')


# %%  ================================== Sensitivity analysis of the growth model ==================================
stage_list = list(stage_lengths.keys())

# Sensitivity scenarios
experiments = [
    ("d_minus10", "d", -0.10),
    ("d_plus10",  "d",  0.10),
    ("e_minus10", "e", -0.10),
    ("e_plus10",  "e",  0.10),
    ("f_minus10", "f", -0.10),
    ("f_plus10",  "f",  0.10),
    ("g_minus10", "g", -0.10),
    ("g_plus10",  "g",  0.10),
]



# Metadata for files
WORLD_METADATA = {
    "clim": {
        "label": "Climatology",
        "temperature": "Climatological temperature (growth seasons 30-year average).",
        "chla": "Climatological Chla (growth seasons 30-year average).",
        "period": "1980-2009",
    },
    "actual": {
        "label": "Actual",
        "temperature": "Actual temperature (growth seasons daily).",
        "chla": "Actual Chla (growth seasons daily).",
        "period": "1980-2018",
    },
    "clim_trended": {
        "label": "Clim Trended (No MHWs)",
        "temperature": "Climatological temperature with warming trend applied.",
        "chla": "Climatological Chla (growth seasons daily).",
        "period": "1980-2018",
    },
    "no_warming": {
        "label": "No Global Warming",
        "temperature": "Actual temperature detrended to remove the warming trend.",
        "chla": "Actual Chla (growth seasons daily).",
        "period": "1980-2018",
    },
}


path_sensitivity = os.path.join(path_surrogates, f'sensitivity_analysis')
files_sensitivity = [
    os.path.join(
        path_sensitivity,
        f"{exp_name}_{world_key}_mass.nc"
    )
    for exp_name, _, _ in experiments
    for world_key in WORLD_METADATA
]

if all(os.path.exists(f) for f in files_sensitivity):
    print("All sensitivity mass files already exist.")

else:
    for exp_name, param, change in experiments:

        print(f"\n========== Experiment: {exp_name} ==========")

        stage_results = {}

        # Run all maturity stages for this experiment
        for stage in stage_list:
            print(f"  Stage: {stage}")

            stage_results[stage] = compute_stage(
                stage=stage,
                sensitivity_param=param,
                sensitivity=change,
            )

        sensitivity_description = (f"Parameter {param} changed by {change:+.0%}. All other parameters held at baseline values (model4).")

        # Save mass only, with maturity stages as variables
        for world_key, metadata in WORLD_METADATA.items():

            stage_data = {
                stage: stage_results[stage][f"{world_key}_mass"]
                for stage in stage_list
            }

            ds = xr.Dataset(stage_data)

            filepath = os.path.join(
                path_sensitivity,
                f"{exp_name}_{world_key}_mass.nc"
            )

            description = (
                f"{metadata['label']} Krill Dry Weight "
                f"({metadata['period']}).\n"
                "Weight calculated using Atkinson et al. (2006).\n"
                f"Sensitivity experiment: {sensitivity_description}"
            )

            save_if_not_exists(
                ds,
                filepath,
                description=description,
                temperature=metadata["temperature"],
                chla=metadata["chla"],
                units="mg",
            )

            del ds, stage_data

        del stage_results

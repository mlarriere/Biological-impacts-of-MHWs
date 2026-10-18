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
import importlib
import functions_length
importlib.reload(functions_length)

from functions_length import (STAGE_LENGTHS, STAGE_LIST, compute_stage, save_if_not_exists)

stage_lengths = STAGE_LENGTHS
stage_list = STAGE_LIST

WORLDS = {
    "clim": (chla_clim_expanded, temp_clim_expanded, "Climatology", 10,),
    "actual": (chla_surf_SO_allyrs, temp_avg_100m_SO_allyrs, "Actual", 10,),
    "clim_trended": (chla_surf_SO_allyrs, temp_clim_trended, "Clim Trended", 10,),
    "no_warming": (chla_surf_SO_allyrs, detrended_temp, "No Warming", 10,),
}

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


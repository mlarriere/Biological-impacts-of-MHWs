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



# %% ======================== Initial Biomass and Carbon fraction ========================
# -- Load data
# Biomass are climatological products [mgC/m3]
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')
biomass_regrid_interp = xr.open_dataset(os.path.join(path_cephalopod, 'regrid_interp/euphausia_biomass_no_eke.nc')) #shape (181, 10, 231, 1442)

# -- Convert biomass from [mgC/m3] to [mg/m3]
# Carbon fraction in Euphausia Superba -- Data from Farber-Lorda et al. (2009)
C_frac_stage = {'juvenile': (0.4989, 0.0250), 'males': (0.4756, 0.0266), 'mature females': (0.4756, 0.0297), 'spent females': (0.5299, 0.0267),} # [fraction C / mg biomass]
mean_C_fraction = np.mean(np.array([v[0] for v in C_frac_stage.values()]))
propagated_sd = np.sqrt(np.sum(np.array([v[1] for v in C_frac_stage.values()])**2) / len(np.array([v[1] for v in C_frac_stage.values()])))

biomass_regrid_interp_dry = biomass_regrid_interp / mean_C_fraction #[mg/m3]


#%% ======================== Length and Biomass Calculation ========================
# -- Load functions and settings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
import functions_length
importlib.reload(functions_length)
from functions_length import STAGE_LIST, compute_stage
from functions_biomass import DEFAULT_PROPORTION, compute_biomass_surrogates

path_sensitivity = os.path.join(path_surrogates, f'sensitivity_analysis')

# The 4 different surrogates
WORLDS = {"clim": (chla_clim_expanded, temp_clim_expanded, "Climatology", 10,),
          "actual": (chla_surf_SO_allyrs, temp_avg_100m_SO_allyrs, "Actual", 10,),
          "clim_trended": (chla_surf_SO_allyrs, temp_clim_trended, "Clim Trended", 10,),
          "no_warming": (chla_surf_SO_allyrs, detrended_temp, "No Warming", 10,)}

# Metadata for the different environmental surrogates
WORLD_METADATA = {
    "clim": {"label": "Climatology",
             "temperature": "Climatological temperature (growth seasons 30-year average).",
             "chla": "Climatological Chla (growth seasons 30-year average).",
             "period": "1980-2009"},
    "actual": {"label": "Actual",
               "temperature": "Actual temperature (growth seasons daily).",
               "chla": "Actual Chla (growth seasons daily).",
               "period": "1980-2018"},
    "clim_trended": {"label": "Clim Trended (No MHWs)",
                     "temperature": "Climatological temperature with warming trend applied.",
                     "chla": "Climatological Chla (growth seasons daily).",
                     "period": "1980-2018"},
    "no_warming": {"label": "No Global Warming",
                   "temperature": "Actual temperature detrended to remove the warming trend.",
                   "chla": "Actual Chla (growth seasons daily).",
                   "period": "1980-2018"}
                   }

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

for exp_name, param, change in experiments:
    # test
    # exp_name = 'f_plus10'
    # param = 'f'
    # change=0.1
    # print(f"\n========== Experiment: {exp_name} ==========")

    # run only if necessary
    sensitivity_path = os.path.join(path_surrogates, 'sensitivity_analysis')
    existing_files = [f for f in os.listdir(sensitivity_path) if f.startswith(f'{exp_name}') and f.endswith('_biomass.nc')]

    if existing_files:
        print(f"Biomass file already exists ({existing_files[0]}). Skipping experiment: {exp_name}")

    else:
        # --------------------------------------------------
        # 1. Calculate lengths and masses -- keep in memory
        # --------------------------------------------------
        def run_stage_for_sensitivity(args):
            stage, worlds, param, change = args
            result = compute_stage(stage=stage, worlds=worlds, sensitivity_param=param, sensitivity=change)
            return stage, result

        stage_args = [(stage, WORLDS, param, change) for stage in STAGE_LIST]
        stage_outputs = process_map(run_stage_for_sensitivity, stage_args, max_workers=4, chunksize=1, desc=f"Stages | {exp_name}")
        stage_results = dict(stage_outputs)

        # Run all maturity stages for this experiment
        # stage_results = {}
        # for stage in STAGE_LIST:
        #     # test
        #     # stage='immature'
        #     print(f"  Stage: {stage}")
        #     stage_results[stage] = compute_stage(stage=stage, worlds=WORLDS, sensitivity_param=param, sensitivity=change)

        # --------------------------------------------------
        # 2. Calculate biomass and save to file
        # --------------------------------------------------
        for world_key, metadata in WORLD_METADATA.items():
            # test
            # world_key='clim'
            # metadata=WORLD_METADATA[world_key]

            output_file = os.path.join(path_surrogates, f'sensitivity_analysis/{exp_name}_{world_key}_biomass.nc')

            if os.path.exists(output_file):
                print(f"  Already exists, skipping: {output_file}")
                continue

            print(f"Computing biomass: {world_key} - {exp_name}")

            # Retrieve mass data
            stage_data = {stage: stage_results[stage][f"{world_key}_mass"] for stage in STAGE_LIST}
            ds_mass = xr.Dataset(stage_data)
            rename_coords = {}
            if "lat" in ds_mass.coords and "lat_rho" not in ds_mass.coords:
                rename_coords["lat"] = "lat_rho"

            if "lon" in ds_mass.coords and "lon_rho" not in ds_mass.coords:
                rename_coords["lon"] = "lon_rho"

            if rename_coords:
                ds_mass = ds_mass.rename(rename_coords)

            description = (
                    f"{metadata['label']} krill biomass ({metadata['period']}). "
                    f"Sensitivity experiment: parameter {param} changed by "
                    f"{change:+.0%}; all other parameters held at baseline."
                )
            
            biomass_ds = compute_biomass_surrogates(
                        ds_mass=ds_mass,
                        initial_biomass_ds=biomass_regrid_interp_dry,
                        label=f"{metadata['label']} {exp_name}",
                        proportion=DEFAULT_PROPORTION,
                        output_file=output_file,
                        mean_C_fraction=mean_C_fraction,
                        propagated_sd=propagated_sd,
                        description=description,
                        max_workers=10,
                    )

            del ds_mass, stage_data, biomass_ds
            gc.collect()
        del stage_results
        gc.collect()

#%% ====================================================================== 
#           Biomass Calculation 
# # ======================================================================
# # --- Load data 
# parameters =['d', 'e', 'f', 'g']
# variations = ['minus10', 'plus10']

# clim_krillmass = {}
# actual_krillmass = {}
# clim_trended_krillmass = {}
# nowarming_krillmass = {}

# def load_dataset(base_path, filename):
#     return xr.open_dataset(os.path.join(base_path, "sensitivity_analysis", filename)).rename({"lat": "lat_rho", "lon": "lon_rho"})

# # 1. Climatology
# for param in parameters:
#     for variation in variations:
#         key=f"{param}_{variation}"
#         clim_krillmass[key]= load_dataset(path_surrogates, f"{param}_{variation}_clim_mass.nc")

# # 2. Actual
# for param in parameters:
#     for variation in variations:
#         key=f"{param}_{variation}"
#         actual_krillmass[key]= load_dataset(path_surrogates, f"{param}_{variation}_actual_mass.nc")

# # 3. No MHWs (MHWs replaced by clim)
# for param in parameters:
#     for variation in variations:
#         key=f"{param}_{variation}"
#         clim_trended_krillmass[key]= load_dataset(path_surrogates, f"{param}_{variation}_clim_trended_mass.nc")

# # 4. No warming (temperature signal detrended)
# for param in parameters:
#     for variation in variations:
#         key=f"{param}_{variation}"
#         nowarming_krillmass[key]= load_dataset(path_surrogates, f"{param}_{variation}_no_warming_mass.nc")


# # --- Compute biomass for each of the variation
# surrogate_inputs = {"clim": clim_krillmass, "actual": actual_krillmass, "clim_trended": clim_trended_krillmass, "nowarming": nowarming_krillmass}
# label = {"clim": "Climatology", "actual": "Actual", "clim_trended": "Clim Trended", "nowarming": "No Warming"}
# biomass = {"clim": {}, "actual": {}, "clim_trended": {}, "nowarming": {}} #initialisation

# for surrogate, input_dict in surrogate_inputs.items():
#     for param in parameters:
#         for variation in variations:
#             # test
#             surrogate='clim_trended'
#             input_dict=clim_trended_krillmass
#             param='g'
#             variation='minus10'

#             key=f"{param}_{variation}"

#             output_file = os.path.join(path_surrogates, f'sensitivity_analysis/biomass/{key}_{surrogate}_biomass.nc')

#             if not os.path.exists(output_file):
#                 print(f"Computing biomass: {surrogate} - {key}")

#                 result = compute_biomass_surrogates(input_dict[key], label=f"{label[surrogate]} {key}", label_da=surrogate, proportion=proportion, 
#                                                    output_file=output_file, max_workers=20)

#             else:
#                 print(f'File for {key} already exists')
    
# # %%

"""
Created on Tue 30 July 17:04:45 2025

Calculating the Biomass of E. Superba

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
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_biomass_ts = os.path.join(path_surrogates, f'biomass_timeseries')
path_biomass_ts_SO = os.path.join(path_biomass_ts, f'SouthernOcean')
path_masslength = os.path.join(path_surrogates, f'mass_length')
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')


# %% ======================== Parameters ========================
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
proportion = {'juvenile': 0.20, 'immature': 0.3, 'mature': 0.3, 'gravid':0.2}

# %% ======================== Area ========================
# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc') #in km2

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)

# %% ======================== Roms grid cell volume ========================
# --- Load data
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc') #in km3
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') #volume first 100m

# --- Southern Ocean south of 60°S
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)

# %% ======================== Biomass from CEPHALOPOD ========================
# -- Load data
# Biomass are climatological products [mgC/m3]
biomass_regrid_interp = xr.open_dataset(os.path.join(path_cephalopod, 'regrid_interp/euphausia_biomass_no_eke.nc')) #shape (181, 10, 231, 1442)

# -- Convert biomass from [mgC/m3] to [mg/m3]
# Carbon fraction in Euphausia Superba -- Data from Farber-Lorda et al. (2009)
C_frac_stage = {'juvenile': (0.4989, 0.0250), 'males': (0.4756, 0.0266), 'mature females': (0.4756, 0.0297), 'spent females': (0.5299, 0.0267),} # [fraction C / mg biomass]
mean_C_fraction = np.mean(np.array([v[0] for v in C_frac_stage.values()]))
propagated_sd = np.sqrt(np.sum(np.array([v[1] for v in C_frac_stage.values()])**2) / len(np.array([v[1] for v in C_frac_stage.values()])))

biomass_regrid_interp_dry = biomass_regrid_interp / mean_C_fraction #[mg/m3]

# %% ======================== Load data ========================
# --- Load mass data [mg] for each maturity stage -- Southern Ocean  
# 1. Climatology
clim_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_mass_stages_SO.nc")) #shape (39, 181, 231, 1442) - repeated info on years
clim_krillmass_SO = clim_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 2. Actual
actual_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "actual_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
actual_krillmass_SO = actual_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 3. No MHWs (MHWs replaced by clim)
clim_trended_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_trended_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
clim_trended_krillmass_SO = clim_trended_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 4. No warming (temperature signal detrended)
nowarming_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "nowarming_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
nowarming_krillmass_SO = nowarming_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})


# %% ======================== Defining Functions ========================
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import functions_biomass
importlib.reload(functions_biomass)

from functions_biomass import (DEFAULT_PROPORTION, compute_biomass_surrogates)

proportion = DEFAULT_PROPORTION.copy()

# %% ==================================== Compute Biomass for surrogates - Regridded and Interpolated ====================================
files_interp = [os.path.join(path_biomass_ts_SO, "biomass_clim.nc"),
                os.path.join(path_biomass_ts_SO, "biomass_actual.nc"),
                os.path.join(path_biomass_ts_SO, "biomass_clim_trend.nc"),
                os.path.join(path_biomass_ts_SO, "biomass_nowarming.nc")]

if not all(os.path.exists(f) for f in files_interp):
    print("Missing output files → running full processing pipeline...\n")

    description = "Evolution of krill biomass weighted according to population proportions.\nBiomass timeseries computed for each bootstrap of the CEPHALOPOD model output.",

    # ---------- Surrogate 1: Climatology
    clim_biomass_interp = compute_biomass_surrogates(clim_krillmass_SO, initial_biomass_ds=biomass_regrid_interp_dry,
                                                      label="Climatology", proportion=proportion, output_file=files_interp[0],
                                                     mean_C_fraction=mean_C_fraction, propagated_sd=propagated_sd, 
                                                     description=description, max_workers=20)
    
    # ---------- Surrogate 2: Actual
    actual_biomass_interp = compute_biomass_surrogates(actual_krillmass_SO, initial_biomass_ds=biomass_regrid_interp_dry,
                                                       label="Actual", proportion=proportion, output_file=files_interp[1],
                                                       mean_C_fraction=mean_C_fraction, propagated_sd=propagated_sd, 
                                                       description=description, max_workers=20)

    # ---------- Surrogate 3: No MHWs
    clim_trended_biomass_interp = compute_biomass_surrogates(clim_trended_krillmass_SO, initial_biomass_ds=biomass_regrid_interp_dry,
                                                             label="Clim Trended", proportion=proportion, output_file=files_interp[2],
                                                             mean_C_fraction=mean_C_fraction, propagated_sd=propagated_sd, 
                                                             description=description, max_workers=20)
    
    # ---------- Surrogate 4: No Warming
    nowarming_biomass_interp = compute_biomass_surrogates(nowarming_krillmass_SO, initial_biomass_ds=biomass_regrid_interp_dry,
                                                          label="No Warming", proportion=proportion, output_file=files_interp[3],
                                                             mean_C_fraction=mean_C_fraction, propagated_sd=propagated_sd, 
                                                             description=description, max_workers=20)

else:
    print("All files already written → Load data")
    # Load data
    clim_biomass_interp = xr.open_dataset(files_interp[0])
    actual_biomass_interp = xr.open_dataset(files_interp[1])
    clim_trended_biomass_interp= xr.open_dataset(files_interp[2])
    nowarming_biomass_interp = xr.open_dataset(files_interp[3])


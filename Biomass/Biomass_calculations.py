"""
Created on Tue 30 July 17:04:45 2025

Calcuating the Biomass of E. Superba

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


# %% ======================== Parameters ========================
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP accorindg to Tarling et al 2006 - graph under 0°C

proportion = {'juvenile': 0.20, 'immature': 0.3, 'mature': 0.3, 'gravid':0.2}
area_atl_sect = 5707378 * 10**6 # Area in m2
area_SO = 20265858.00* 10**6 # Area in m2
N = 17.85 # Abundance ind/m2



# %% ======================== Load data ========================
# Load mass data for each maturity stage
mass_juvenile = xr.open_dataset(os.path.join(path_mass, 'mass_juvenile.nc'))
mass_immature = xr.open_dataset(os.path.join(path_mass, 'mass_immature.nc'))
mass_mature = xr.open_dataset(os.path.join(path_mass, 'mass_mature.nc'))
mass_gravid = xr.open_dataset(os.path.join(path_mass, 'mass_gravid.nc'))


# Extract final mass for each maturity stage, MHW conditions and year
mass_final = {stage: [] for stage in stage_lengths}

for stage, ds in zip(['juvenile', 'immature', 'mature', 'gravid'],
                     [mass_juvenile, mass_immature, mass_mature, mass_gravid]):
    # Select the last day for each year, assuming 'days' is sorted
    final_mass = ds.isel(days=-1)
    
    # Keep only the mass variables (non_mhw and mhw_*)
    mass_vars = [v for v in ds.data_vars]
    final_mass_vars = final_mass[mass_vars]
    
    mass_final[stage] = final_mass_vars #shape: (39, )



# %% ======================== Biomass Calculation ========================
# Calculate total biomass (mg) for each year
biomass_total = {}
for var in ['mass_cat0', 'mass_cat1', 'mass_cat2', 'mass_cat3', 'mass_cat4']:
    B_j = area_atl_sect * sum(proportion[stage] * N * mass_final[stage][var] for stage in proportion)
    B_j_Mt = B_j * 1e-15  # Convert from mg to Mt
    biomass_total[var] = B_j_Mt


# %% ======================== PLOT ========================
years_to_plot = [1989, 2000, 2016]
year_indices = [year - 1980 for year in years_to_plot]

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, year, idx in zip(axs, years_to_plot, year_indices):
    biomass_year = {var: biomass_total[var].isel(years=idx).item() for var in biomass_total}
    bars = ax.bar(biomass_year.keys(), biomass_year.values(), color='teal')
    ax.set_title(f"Biomass in {year}")
    ax.set_xlabel("MHW Scenario")
    ax.set_xticklabels(biomass_year.keys(), rotation=45)
    if ax == axs[0]:
        ax.set_ylabel("Biomass (Mt)")

    # Add text labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:.3e}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# %%

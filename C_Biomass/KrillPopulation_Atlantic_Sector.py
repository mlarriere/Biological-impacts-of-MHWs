"""
Created on Tue 31 July 15:06:14 2025

Length-Mass relationship - Atlantic Sector only

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
temp_avg_100m_study_area_allyrs = subset_spatial_domain(temp_avg_100m_SO_allyrs) 

chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 
chla_surf_study_area_allyrs = subset_spatial_domain(chla_surf_SO_allyrs) 


# %% ================= Preparing data =================
# Atlantic Sector for 1 season of interest
from Growth_Model.Atkinson2006_model import length_Atkison2006 
def length_to_mass(length_array, p = 10**(-4.19), r=3.89):
    ''' Constants from Atkinson et al (2006)'''
    mass_array = p*length_array**r
    return mass_array

# --- Parameters
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP accorindg to Tarling et al 2006 - graph under 0°C

# ==== Climatological Drivers -> Mean Chla and T°C  (days, eta, xi)
# -- Atlantic sector
temp_clim_atl = temp_avg_100m_study_area_allyrs.isel(years=slice(0,30)) #shape: (30, 181, 231, 360)
temp_clim_atl_mean = temp_clim_atl.mean(dim=['years']) #shape: (181, 231, 360)
chla_clim_atl = chla_surf_study_area_allyrs.isel(years=slice(0,30))
chla_clim_atl_mean = chla_clim_atl.mean(dim=['years'])


#%% ====================================================================== 
# Calculating length for each maturity stage CLIMATOLOGY 
# ====================================================================== 
# %%  ================= Climatological Lengths for each grid cell -- Atlantic Sector =================
clim_length_stages =[]
clim_mass_stages =[]
for stage in stage_lengths:
    print(stage)
    climatological_length = length_Atkison2006(chla=chla_clim_atl_mean.chla, 
                                                temp=temp_clim_atl_mean.avg_temp,
                                                initial_length=stage_lengths[stage],
                                                intermoult_period=stage_IMP[stage],
                                                maturity_stage=stage)
    climatological_mass = length_to_mass(climatological_length)

    # Store results
    clim_length_stages.append(climatological_length)
    clim_mass_stages.append(climatological_mass)

# -- To Dataset
length_vars = {}
mass_vars = {}
for i, stage in enumerate(stage_lengths):
    length_vars[stage] = clim_length_stages[i].rename(stage)
    mass_vars[stage] = clim_mass_stages[i].rename(stage)

clim_length_ds = xr.Dataset(length_vars)
clim_mass_ds = xr.Dataset(mass_vars)

# Metadata
shared_coords = {
    'days': chla_surf_study_area_allyrs.days,
    'lon_rho': chla_surf_study_area_allyrs.lon_rho,
    'lat_rho': chla_surf_study_area_allyrs.lat_rho,
}
clim_length_ds = clim_length_ds.assign_coords(shared_coords)
clim_mass_ds = clim_mass_ds.assign_coords(shared_coords)

clim_length_ds.attrs = {
    'description': 'Climatological Krill Length Trajectories (30yrs : 1980-2009)',
    'time window': 'Growth Season (1st Nov to 30th Apr)',
    'extent': 'Atlantic Sector',
    'growth_model': 'Atkinson et al., 2006',
    'created_on': '2025-08-06',
    'units': 'mm',
}

clim_mass_ds.attrs = {
    'description': 'Climatological Krill Dry weight Trajectories (30yrs : 1980-2009)',
    'time window': 'Growth Season (1st Nov to 30th Apr)',
    'extent': 'Atlantic Sector',
    'mass_model': 'Atkinson et al., 2006',
    'created_on': '2025-08-06',
    'units': 'mg',
}

# -- Save to file
fname = f"clim_length_traj_mat_stages_Atl.nc"
clim_length_ds.to_netcdf(os.path.join(path_length, fname))
print(f"Saved mass dataset: {fname}")

fname = f"clim_mass_traj_mat_stages_Atl.nc"
clim_mass_ds.to_netcdf(os.path.join(path_mass, fname))
print(f"Saved mass dataset: {fname}")

#%% ====================================================================== 
# Calculating length for each maturity stage YEARLY 
# ====================================================================== 
# %%  ================= Atlantic Sector =================
# == Lengths traj for each year and grid cell -- shape (39, 181, 231, 360)
years = np.arange(1980, 2019)
year_indices = years - 1980  # 0 to 38

lengths_by_stage = {stage: [] for stage in stage_lengths}
mass_by_stage = {stage: [] for stage in stage_lengths}


for yr in year_indices:
    print(yr+1980)
    for stage in stage_lengths:
        print(stage)
        length_yr = length_Atkison2006(chla=chla_surf_study_area_allyrs.chla.isel(years=yr), 
                                        temp=temp_avg_100m_study_area_allyrs.avg_temp.isel(years=yr), 
                                        initial_length=stage_lengths[stage], 
                                        intermoult_period=stage_IMP[stage],
                                        maturity_stage=stage)
        # --- Check shape and fix if necessary ---
        if length_yr.dims != ('days', 'eta_rho', 'xi_rho'):
            length_yr = length_yr.transpose('days', 'eta_rho', 'xi_rho')

        # Add coordinate year
        length_yr = length_yr.expand_dims(dim={'years': [yr + 1980]})

        # Converting to mass
        mass_yr = length_to_mass(length_yr)   

        # Store rsults
        lengths_by_stage[stage].append(length_yr)
        mass_by_stage[stage].append(mass_yr)

length_datasets = {}
mass_datasets = {}
for stage in stage_lengths:
    # Concatenate
    length_stack = xr.concat(lengths_by_stage[stage], dim='years')
    mass_stack = xr.concat(mass_by_stage[stage], dim='years')
    
    # Adding coords
    length_stack = length_stack.assign_coords(years=years)
    mass_stack = mass_stack.assign_coords(years=years)

    length_datasets[stage] = length_stack
    mass_datasets[stage] = mass_stack

# -- To Dataset
krill_length_ds = xr.Dataset(length_datasets)
krill_mass_ds = xr.Dataset(mass_datasets)

# Metadata
shared_coords = {
    'days': chla_surf_study_area_allyrs.days,
    'eta_rho': chla_surf_study_area_allyrs.eta_rho,
    'xi_rho': chla_surf_study_area_allyrs.xi_rho,
    'lon_rho': chla_surf_study_area_allyrs.lon_rho,
    'lat_rho': chla_surf_study_area_allyrs.lat_rho,
}
krill_length_ds = krill_length_ds.assign_coords(shared_coords)
krill_mass_ds = krill_mass_ds.assign_coords(shared_coords)


krill_length_ds.attrs = {
    'title': 'Krill Length Trajectories (1980–2018)',
    'description': 'Daily krill body lengths for different maturity stages using Atkinson (2006) growth model',
    'time window': 'Growth Season (1st Nov to 30th Apr)',
    'extent': 'Atlantic Sector',
    'growth_model': 'Atkinson et al., 2006',
    'created_on': '2025-08-06',
    'units': 'mm',
}

krill_mass_ds.attrs = {
    'title': 'Krill Dry Mass Trajectories (1980–2018)',
    'description': 'Dry weight estimates derived from krill length for different maturity stages',
    'time window': 'Growth Season (1st Nov to 30th Apr)',
    'extent': 'Atlantic Sector',
    'mass_model': 'Atkinson et al., 2006',
    'created_on': '2025-08-06',
    'units': 'mg',
}

# -- Save to file
fname = f"length_traj_mat_stages_Atl.nc"
krill_length_ds.to_netcdf(os.path.join(path_length, fname))
print(f"Saved mass dataset: {fname}")

fname = f"mass_traj_mat_stages_Atl.nc"
krill_mass_ds.to_netcdf(os.path.join(path_mass, fname))
print(f"Saved mass dataset: {fname}")


# %%

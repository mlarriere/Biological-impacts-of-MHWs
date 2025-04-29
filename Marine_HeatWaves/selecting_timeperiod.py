"""
Created on Mon 29 Apr 08:54:05 2025

Selecting time period -- austral summer

@author: Marguerite Larriere (mlarriere)
"""
# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath

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
    "axes.titlesize": 10,
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

path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
joel_path ='/home/jwongmeng/work/ROMS/scripts/mhw_krill/' #codes joel
output_path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

# %% Select time period according to Bahlburg et al. (2023)
# We only are interested by the summer and early spring in the austral summer, i.e. 1stNov (day=305) until 1st May (day=121)
os.makedirs(os.path.join(output_path, "det_depth/austral_summer"), exist_ok=True)

det_files = glob.glob(os.path.join(output_path, "det_depth/det_*.nc"))

def austral_sumer(file):
    start_time = time.time()

    # Retrieve depth of file as string
    basename = os.path.basename(file)
    depth_str = basename.split('_')[1].replace('m.nc', '')  # gets '5', '11', etc.
    print(f'Depth being processed: {depth_str}\n')

    # Read data
    # file =  '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/det_11m.nc'
    det_ds = xr.open_dataset(file)

    # Select only summer and early spring
    det_austral = xr.concat([
        det_ds.isel(days=slice(0, 121)),     # 1 Jan to 1 May (Day 0–120)
        det_ds.isel(days=slice(305, 365))    # 1 Nov to 31 Dec (Day 305–364)
    ], dim='days') #shape: (40, 181, 434, 1442)
    
    # Save file
    output_file = os.path.join(output_path, "det_depth/austral_summer", f"det_depth{depth_str}m_summer.nc")
    if not os.path.exists(output_file):
        try:
            det_austral.to_netcdf(output_file, engine="netcdf4")
            print(f"File written: {depth_str}")
        except Exception as e:
            print(f"Error writing {depth_str}: {e}")    
    
    elapsed_time = time.time() - start_time
    print(f"Processing time for {depth_str}m: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")

    # Free memory
    del det_austral, det_ds
    gc.collect()
    
process_map(austral_sumer, det_files, max_workers=30, desc="Processing file")  # in parallel - computing time ~5min per file




# %%

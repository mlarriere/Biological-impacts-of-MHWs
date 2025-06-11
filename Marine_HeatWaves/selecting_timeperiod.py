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

path_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
joel_path ='/home/jwongmeng/work/ROMS/scripts/mhw_krill/' #codes joel
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'

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
os.makedirs(os.path.join(path_det, "austral_summer"), exist_ok=True)

det_files = glob.glob(os.path.join(path_det, "det_*.nc"))

def austral_sumer(file):
    start_time = time.time()

    absolute_thresh_extended=True

    if absolute_thresh_extended:
        # For testing    
        file='/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/det5m_extended.nc'

        # Retrieve depth of file as string
        basename = os.path.basename(file)
        depth_str = basename.split('_')[0][3:-1].replace('m_extended.nc', '')
        print(f'Depth being processed: {depth_str}\n')

    else:
        # For testing
        # file =  '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/det_11m.nc'

        # Retrieve depth of file as string
        basename = os.path.basename(file)
        depth_str = basename.split('_')[1].replace('m.nc', '')
        print(f'Depth being processed: {depth_str}\n')

    # Read data
    det_ds = xr.open_dataset(file) #days ranging from idx0 to idx364

    # Select only austral summer and early spring
    jan_april = det_ds.sel(days=slice(0, 119)) # 1 Jan to 30 April (Day 0-119) - last idx excluded
    jan_april.coords['days'] = jan_april.coords['days'] #keep info on day
    
    nov_dec = det_ds.sel(days=slice(304, 365)) # 1 Nov to 31 Dec (Day 304–364) - last idx excluded
    nov_dec.coords['days'] = np.arange(304, 365) #keep info on day
    
    det_austral = xr.concat([nov_dec, jan_april], dim="days") #181days

    # Save to file
    if absolute_thresh_extended:
        output_file = os.path.join(path_det, "austral_summer", f"det_depth{depth_str}m_extended.nc")
    else:
        output_file = os.path.join(path_det, "austral_summer", f"det_depth{depth_str}m.nc")

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
    
# Calling function in parallel
process_map(austral_sumer, det_files, max_workers=30, desc="Processing file")  #computing time ~1-4min per file

# %%

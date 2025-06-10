# -*- coding: utf-8 -*-
"""
Created on Tues 10 June 11:29:36 2025

Extending MHWs detection 

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
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

import time
from tqdm.contrib.concurrent import process_map
from functools import partial

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
    "font.size": 10,           
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,   
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

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold
percentile = 90 

# Handling time
from datetime import datetime, timedelta
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0, 121)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)


# %% ============== LOAD DATA ==============
def fill_short_gaps(arr, max_gap=5):
    """
    arr: 1D boolean numpy array (True/False)
    max_gap: maximum gap length to fill
    Returns: new array with short False gaps filled with True
    """
    arr = arr.copy()
    n = len(arr)
    # Find transitions: True->False or False->True
    diff = np.diff(arr.astype(int))
    # Start indices of False runs (+1 because diff is shifted)
    false_starts = np.where(diff == -1)[0] + 1
    # End indices of False runs
    false_ends = np.where(diff == 1)[0] + 1
    
    # Handle edge cases where start or end with False
    if arr[0] == False:
        false_starts = np.insert(false_starts, 0, 0)
    if arr[-1] == False:
        false_ends = np.append(false_ends, n)
    
    # Fill short gaps
    for start, end in zip(false_starts, false_ends):
        length = end - start
        if length <= max_gap:
            arr[start:end] = True
    return arr

det5m = xr.open_dataset(os.path.join(path_det,'det_5m.nc')) #boolean dataset with detection 

def extended_detection(ieta, varname):
    # ieta=220
    det5m_yr = det5m.isel(eta_rho=ieta)
    data=det5m_yr[varname]

    data_stacked = data.stack(time=['years','days'])

    # Fill when gaps<=5days, i.e. replacing False by True -> extending the detection
    # Example: T T T T T T T F F T T T becomes only T 
    # data_test = data[950, 36*365-100: 37*365-330]
    # print(data_test)

    extended_array = np.empty_like(data_stacked) 
    for xi in range(data_stacked.shape[0]):
        extended_array[xi] = fill_short_gaps(data_stacked[xi], max_gap=5) #shape: (1442, 14600)
    
    return ieta, extended_array

# --- Initialisation
threshold_vars = ['mhw_abs_threshold_1_deg', 'mhw_abs_threshold_2_deg', 'mhw_abs_threshold_3_deg', 'mhw_abs_threshold_4_deg']
results_dict = {var: np.empty((neta, nxi, nyears*ndays), dtype=bool) for var in threshold_vars}
# extended_det_ds = np.empty((neta, nxi, nyears*ndays), dtype=bool)

# --- Calling function to combine eta (process_map)
for varname in threshold_vars:
    extended_detection_ieta = partial(extended_detection, varname=varname)
    for ieta, extended_det in process_map(extended_detection_ieta, range(neta), max_workers=30, desc=f"Processing {varname}"):
        results_dict[varname][ieta] = extended_det

# extended_detection_ieta = partial(extended_detection)
# for ieta, extended_det  in process_map(extended_detection_ieta, range(neta), max_workers=30, desc="Processing eta"):
#     extended_det_ds[ieta] = extended_det

# --- Reshaping
da_dict = {}
for varname in threshold_vars:
    extended_det_ds = results_dict[varname]

    # Flip eta position
    extended_det_transposed = extended_det_ds.transpose(2, 0, 1)
    
    # To DataArray  
    da = xr.DataArray(
        extended_det_transposed,
        coords={
            'time': det5m[varname].stack(time=['years', 'days']).time,
            'eta_rho': det5m[varname]['eta_rho'],
            'xi_rho': det5m[varname]['xi_rho'],
        },
        dims=['time', 'eta_rho', 'xi_rho']
    )

    # Reshape time
    da_unstacked = da.unstack('time')
    da_dict[varname] = da_unstacked

# Flip eta position
# extended_det_transposed = extended_det_ds.transpose(2, 0, 1) #shape: (14600, 434, 1442)

# To DataArray
# extended_da = xr.DataArray(
#         extended_det_transposed,
#         coords={
#             'time': det5m.mhw_abs_threshold_1_deg.stack(time=['years', 'days']).time,
#             'eta_rho': det5m.mhw_abs_threshold_1_deg['eta_rho'],
#             'xi_rho': det5m.mhw_abs_threshold_1_deg['xi_rho']
#         },
#         dims=['time', 'eta_rho', 'xi_rho'],
#     )

# Reshape time
# extended_det_reshaped = extended_da.unstack('time')

# --- To DataSet
extended_ds = xr.Dataset(da_dict)
# Rename variables
extended_ds['det_1deg_extended'] = extended_ds['mhw_abs_threshold_1_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_1_deg'])
extended_ds['det_2deg_extended'] = extended_ds['mhw_abs_threshold_2_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_2_deg'])
extended_ds['det_3deg_extended'] = extended_ds['mhw_abs_threshold_3_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_3_deg'])
extended_ds['det_4deg_extended'] = extended_ds['mhw_abs_threshold_4_deg']
extended_ds = extended_ds.drop(['mhw_abs_threshold_4_deg'])


# Add coordinates
extended_ds = extended_ds.assign_coords(
    lon_rho=(('eta_rho', 'xi_rho'), ds_roms.lon_rho.values),
    lat_rho=(('eta_rho', 'xi_rho'), ds_roms.lat_rho.values),
    years=extended_ds.coords['years']+1980,
    days=extended_ds.coords['days'],
)


# extended_ds = xr.Dataset(
#     {"extended_detection": extended_det_reshaped},
#     coords={
#         "lon_rho": (("eta_rho", "xi_rho"), ds_roms.lon_rho.values),
#         "lat_rho": (("eta_rho", "xi_rho"), ds_roms.lat_rho.values),
#         "years": extended_det_reshaped.coords['years'],
#         "days": extended_det_reshaped.coords['days'],
#     }
# )

# --- Save file
output_file = os.path.join(path_det, f"det5m_extended.nc")
if not os.path.exists(output_file):
    extended_ds.to_netcdf(output_file, engine="netcdf4")


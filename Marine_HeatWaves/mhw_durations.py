# -*- coding: utf-8 -*-
"""
Created on Mon 19 May 09:37:56 2025

Calculate MHW durations for the first 100m depth 

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory

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
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'

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

var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold
percentile = 90 
pmod = 'perc' + str(percentile)


# -- Handling time
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) #defining months with days within a year
month_names = np.array(['Jan','Feb','Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov','Dec'])

season_bins = np.array([0, 90, 181, 273, 365]) #defining seasons with days within a year
season_names = np.array(['DJF (Summer)', 'MAM (Fall)', 'JJA (Winter)', 'SON (Spring)']) #southern ocean!


# %% -------------------------------- Durations Events --------------------------------
# According to Hobday et al. (2016) - MHW needs to persist for at least five days (5days of TRUE)
def compute_mhw_durations(arr):
    """
    Compute duration of consecutive Trues (MHW) and Falses (non-MHW) in a 1D boolean array (time, ).
    Return two arrays of same shape: mhw_durations, non_mhw_durations.
    """
    # arr= bool_series

    n = arr.shape[0]
    durations = np.zeros(n, dtype=np.int32)
    
    if n == 0:  # Empty case
        return durations, durations

    # Find run starts and lengths
    is_diff = np.diff(arr.astype(int), prepend=~arr[0]) != 0  # Detect transitions
    run_ids = np.cumsum(is_diff)  # Label runs
    run_lengths = np.bincount(run_ids, minlength=run_ids[-1] + 1)  # Length of each run

    # Map lengths back
    durations = run_lengths[run_ids]

    mhw_durations = np.where(arr, durations, 0)
    non_mhw_durations = np.where(~arr, durations, 0)

    return mhw_durations, non_mhw_durations


ds = xr.open_dataset(path_mhw + file_var + 'eta200.nc') #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
all_depths= ds['z_rho'].values

def mhw_duration(depth_idx):
# for depth_idx in range(14): # computing time ~10min per depth
    # depth_idx=0

    print(f'---------DEPTH{depth_idx}---------')

    start_time = time.time()
    
    # Read data - only relative threshold
    det_ds_depth= xr.open_dataset(os.path.join(output_path, f"det_depth/det_{-all_depths[depth_idx]}m.nc")) #read each depth

    # Intersection between relative and 1°C threshold (min abs thresh) to define mhw duration
    det_rel_thres = det_ds_depth.mhw_rel_threshold # shape: (40, 365, 434, 1442). Boolean
    det_abs_1deg = det_ds_depth.mhw_abs_threshold_1_deg # shape: (40, 365, 434, 1442). Boolean
    mhw_both_thresholds = np.logical_and(det_rel_thres, det_abs_1deg) #rel thresh is True if >1°C else False ~1min computing
    
    mhw_both_thresholds_stacked = mhw_both_thresholds.stack(time=('years', 'days')) #consider time as continuous
    mhw_both_thresholds_stacked = mhw_both_thresholds_stacked.transpose('time', 'eta_rho', 'xi_rho') #time as first dim

    # Initialization ~10s
    ntime, neta, nxi = mhw_both_thresholds_stacked.shape
    # mhw_stacked = np.zeros((ntime, neta, nxi), dtype=np.int32)
    # non_mhw_stacked = np.zeros((ntime, neta, nxi), dtype=np.int32)
    mhw_both_thresh_reshaped = mhw_both_thresholds_stacked.values.reshape(ntime, neta * nxi) #(14600, 625828)
    mhw_durations_all = np.zeros_like(mhw_both_thresh_reshaped, dtype=np.int32)
    non_mhw_durations_all = np.zeros_like(mhw_both_thresh_reshaped, dtype=np.int32)

    # Calculating duration -- ~ 11min per depth
    # for ixi in range(nxi):
    #     for ieta in range(neta):
    #         # ixi=27
    #         # ieta = 200
    #         bool_series = det_rel_thres_stacked[:, ieta, ixi].values  # 1D time series 
    #         mhw_dur, non_mhw_dur = compute_mhw_durations(bool_series)
    #         mhw_stacked[:, ieta, ixi] = mhw_dur
    #         non_mhw_stacked[:, ieta,ixi] = non_mhw_dur

    # Calculating duration -- ~ 4min per depth
    print('Calculating duration')
    for i in range(mhw_both_thresh_reshaped.shape[1]): # Loop over grid points (1D)
        mhw_durations_all[:, i], non_mhw_durations_all[:, i] = compute_mhw_durations(mhw_both_thresh_reshaped[:, i])

    # Reshape to (eta, xi)
    mhw_durations = mhw_durations_all.reshape(ntime, neta, nxi)
    non_mhw_durations = non_mhw_durations_all.reshape(ntime, neta, nxi)

    # Check
    # np.all(mhw_durations == mhw_stacked) #True

    # Check for bad values
    # assert np.all(np.isfinite(mhw_stacked)), "mhw_durations has non-finite values"
    # assert np.all(np.isfinite(non_mhw_stacked)), "non_mhw_durations has non-finite values"

    # To dataset
    ds_intermediate = xr.Dataset({
        "mhw_durations": (("time", "eta_rho", "xi_rho"), mhw_durations),
        "non_mhw_durations": (("time", "eta_rho", "xi_rho"), non_mhw_durations),
    }, coords={
        "lon_rho":(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
        "lat_rho":(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),  # (434, 1442)
    })

    # print(ds_intermediate.mhw_durations.isel(time=slice(38*365+70, 39*365-70), eta_rho=200, xi_rho=1000).values)
    # print(ds_intermediate.non_mhw_durations.isel(time=slice(38*365+70, 39*365-70), eta_rho=200, xi_rho=1000).values)
          
    # Clean memory
    del non_mhw_durations, mhw_durations, det_ds_depth #,non_mhw_stacked, mhw_stacked
    gc.collect()
    print(f"Memory used: {psutil.virtual_memory().percent}%")


    # --- Rules on duration
    print('Rules duration')

    # MHW last at least 5 days
    mhw_event = ds_intermediate['mhw_durations'] >= 5 
    # print(mhw_event.isel(time=slice(38*365+70, 39*365-70), eta_rho=200, xi_rho=1000).values)

    # Gap of 1 day -- ~15min
    # gap_1_day = (ds_intermediate['non_mhw_durations'] == 1) & (mhw_event.shift(time=1).astype(bool)) & (mhw_event.shift(time=-1).astype(bool))
    # gap_1_day = gap_1_day.astype(bool)

    # # Gap of 2 days -- ~15min
    # gap_2_day = (ds_intermediate['non_mhw_durations'] == 2) & (mhw_event.shift(time=2).astype(bool)) & (mhw_event.shift(time=-2).astype(bool))
    # gap_2_day = gap_2_day.astype(bool)

    # Extract arrays
    mhw_array = mhw_event.values  # (time, eta_rho, xi_rho)
    non_mhw_array = ds_intermediate['non_mhw_durations'].values  # (time, eta_rho, xi_rho)
    gap_1_day = np.zeros_like(mhw_array, dtype=bool)
    gap_2_day = np.zeros_like(mhw_array, dtype=bool)

    # Gap of 1 day -- ~1min
    mhw_prev = np.roll(mhw_array, shift=1, axis=0)
    mhw_next = np.roll(mhw_array, shift=-1, axis=0)
    mhw_prev[0, :, :] = False  # First timestep invalid (1 January 1980)
    mhw_next[-1, :, :] = False  # Last timestep invalid (31 Dec 2019)
    gap_1_day = (non_mhw_array == 1) & mhw_prev & mhw_next
    
    # np.all(gap_1_day==gap_1_day_test) #True
    
    # Gap of 2 days -- ~1min
    mhw_prev2 = np.roll(mhw_array, shift=2, axis=0)
    mhw_next2 = np.roll(mhw_array, shift=-2, axis=0)
    mhw_prev2[0:2, :, :] = False  # First two timesteps invalid
    mhw_next2[-2:, :, :] = False  # Last two timesteps invalid
    gap_2_day = (non_mhw_array == 2) & mhw_prev2 & mhw_next2

    # np.all(gap_2_day==gap_2_day_test) #True

    # Combine events
    mhw_event_combined_stacked = mhw_event | gap_1_day | gap_2_day
    mhw_event_combined_stacked = mhw_event_combined_stacked.astype(bool)
    # print(mhw_event_combined_stacked.isel(time=slice(38*365+70, 39*365-70), eta_rho=200, xi_rho=1000).values)

    # Initialization
    ntime, neta, nxi = mhw_event_combined_stacked.shape
    # mhw_recalc = np.zeros((ntime, neta, nxi), dtype=np.int32)
    # non_mhw_recalc = np.zeros((ntime, neta, nxi), dtype=np.int32)
    mhw_event_combined_reshaped = mhw_event_combined_stacked.values.reshape(ntime, neta * nxi) #(14600, 625828)
    mhw_durations_recalc_all = np.zeros_like(mhw_event_combined_reshaped, dtype=np.int32)
    non_mhw_durations_recalc_all = np.zeros_like(mhw_event_combined_reshaped, dtype=np.int32)

    # Recalculating duration -- ~11min
    # for ixi in range(nxi):
    #     for ieta in range(neta):
    #         bool_series = mhw_event_combined_stacked[:,ieta,ixi].values  # 1D time series
    #         mhw_dur_ext, non_mhw_dur_ext = compute_mhw_durations(bool_series)
    #         mhw_recalc[:,ieta,ixi] = mhw_dur_ext
    #         non_mhw_recalc[:,ieta,ixi] = non_mhw_dur_ext

    # Recalculating duration -- ~ 4min per depth
    print('Recalculating duration')
    for i in range(mhw_event_combined_reshaped.shape[1]): # Loop over grid points (1D)
        mhw_durations_recalc_all[:, i], non_mhw_durations_recalc_all[:, i] = compute_mhw_durations(mhw_event_combined_reshaped[:, i])

    # Reshape to (eta, xi)
    mhw_recalc = mhw_durations_recalc_all.reshape(ntime, neta, nxi)
    non_mhw_recalc= non_mhw_durations_recalc_all.reshape(ntime, neta, nxi)
    # print(mhw_recalc[38*365+70: 39*365-70, 200, 1000])

    # Reformat with years and days in dim
    mhw_recalc_reshaped = mhw_recalc.reshape((40, 365, neta, nxi))
    non_mhw_recalc_reshaped = non_mhw_recalc.reshape((40, 365, neta, nxi))

    # To dataset
    ds_out = xr.Dataset({
        "mhw_durations": (("years", "days", "eta_rho", "xi_rho"), mhw_recalc_reshaped),
        "non_mhw_durations": (("years", "days", "eta_rho", "xi_rho"), non_mhw_recalc_reshaped),
        }, 
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
            ), 
        attrs={
            'depth': f"{-all_depths[depth_idx]}m, i.e. layer{depth_idx}",
            'description': 'Durations recalculated using Hobday rules (min 5-day events, 1-2 day gaps allowed). Events T°C are above the relative and absolute (1°C) thresholds'}
        )

    # Clean memory
    del mhw_recalc_reshaped, non_mhw_recalc_reshaped, mhw_recalc, non_mhw_recalc, mhw_event_combined_stacked, mhw_event, gap_2_day, gap_1_day
    gc.collect()
    print(f"Memory used: {psutil.virtual_memory().percent}%")

    # ====== ILLUSTRATION ======
    print('Illustration') 
    # to show duration recalculation (before/after)
    # if depth_idx==0:
    year_to_plot = slice(37, 39) #last idx excluded
    xi_rho_to_plot = 1000  
    eta_rho_to_plot = 200  
    # ---- BEFORE
    # Reformat with years and days in dim
    mhw_intermediate_reshape = xr.Dataset({
        "mhw_durations": (("years", "days", "eta_rho", "xi_rho"), ds_intermediate['mhw_durations'].values.reshape((nyears, ndays, neta, nxi))),
        "non_mhw_durations": (("years", "days", "eta_rho", "xi_rho"),  ds_intermediate['non_mhw_durations'].values.reshape((nyears, ndays, neta, nxi))),
        },
        coords={
            "years": np.arange(1980, 2020),
            "days": np.arange(365),
            "eta_rho": ds_intermediate["eta_rho"],
            "xi_rho": ds_intermediate["xi_rho"]
        }
    )
    
    # Slicing dataset
    mhw_duration_before = mhw_intermediate_reshape['mhw_durations'].isel(years=year_to_plot, xi_rho=xi_rho_to_plot, eta_rho=eta_rho_to_plot)
    mhw_duration_before = mhw_duration_before.stack(time=('years', 'days'))
    non_mhw_duration_before = mhw_intermediate_reshape['non_mhw_durations'].isel(years=year_to_plot, xi_rho=xi_rho_to_plot, eta_rho=eta_rho_to_plot)
    non_mhw_duration_before = non_mhw_duration_before.stack(time=('years', 'days')) 
    # print('BEFORE', mhw_duration_before.isel(time=slice(120,210)).values)

    # ---- AFTER 
    mhw_event_recalculated = ds_out['mhw_durations'].isel(years=year_to_plot, xi_rho=xi_rho_to_plot, eta_rho=eta_rho_to_plot)
    mhw_event_recalculated = mhw_event_recalculated.stack(time=('years', 'days')) 
    # print('AFTER', mhw_event_recalculated.isel(time=slice(120,210)).values)
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_clip_on(False)

    ax.plot(non_mhw_duration_before, label="Non-MHW", color='#A2B36B',linestyle=":")
    ax.plot(mhw_duration_before, label="MHW", color='#DC6D04', linestyle="--")
    ax.plot(mhw_event_recalculated, label="Extended MHW", color='#780000', linestyle="-")

    ax.set_title(f'Illustration of Hobday et al (2016) rules on MHW duration ({all_depths[depth_idx]}m depth)')
    ax.set_xlabel('Days')
    ax.set_ylabel('Duration (days)')
    # ax.set_ylim(-1,10)
    # ax.set_xlim(135,165)

    # Add the year labels below the x-axis
    ax.annotate('', xy=(365, -65), xytext=(0, -65), arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), annotation_clip=False)
    ax.annotate('', xy=(365*2, -65), xytext=(365, -65), arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), annotation_clip=False)
    # ax.annotate('', xy=(365*3, -65), xytext=(365*2, -65), arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), annotation_clip=False)

    ax.text(182, -80, f'{1980 + year_to_plot.start}', ha='center', va='center')
    ax.text(365 + 182, -80, f'{1980 + year_to_plot.start+1}', ha='center', va='center')
    # ax.text(365*2 + 182, -80, f'{1980 + year_to_plot.start+2}', ha='center', va='center')

    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Clean memory
    del mhw_intermediate_reshape, mhw_event_recalculated, non_mhw_duration_before, mhw_duration_before
    gc.collect()
    
    # Save file
    output_file = os.path.join(output_path, "mhw_durations", f"mhw_duration_{-all_depths[depth_idx]}m.nc")
    if not os.path.exists(output_file):
        try:
            ds_out.to_netcdf(output_file, engine="netcdf4")
            print(f"File written: {depth_idx}")
        except Exception as e:
            print(f"Error writing {depth_idx}: {e}")    

    # Clean memory
    del ds_intermediate, ds_out
    gc.collect()

    elapsed_time = time.time() - start_time
    print(f"Processing time for depth {depth_idx}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")

os.makedirs(os.path.join(output_path, "mhw_durations"), exist_ok=True)
process_map(mhw_duration, range(3,6), max_workers=6, desc="Processing depth")  # detects extremes for each latitude in parallel - computing time ~10min total


# %%

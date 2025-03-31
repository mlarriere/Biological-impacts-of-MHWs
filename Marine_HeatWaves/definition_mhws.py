#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 17 Feb 09:32:05 2025

Calculate MHW  durations

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath

import time

from joblib import Parallel, delayed

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



# %% -------------------------------- LOAD DATA --------------------------------
# ------ PER LATITUDE
def detect_absolute_mhw(ieta):
    # ieta =200 #for testing

    print(f"Processing eta {ieta}...")
    start_time = time.time()

    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:,0:365, : ,:]#.squeeze(axis=2) #Extracts daily data : only 40yr + consider 365 days per year
    ds_original_surf = ds_original.isel(z_rho=0) #only surface
    print(np.unique(ds_original_surf.lat_rho.values))

    # Dataset initialization
    nyears, ndays, nxi = ds_original_surf.shape
    mhw_rel_threshold = np.full((nyears, ndays, nxi), np.nan, dtype=bool)

    # Deal with NANs values
    ds_original_surf.values[np.isnan(ds_original_surf.values)] = 0 # Set to 0 so that det = False at nans

    # # -- TEST --
    # mean_temp = ds_original.mean(dim=['xi_rho'])
    # # Save the plot
    # plt.figure()
    # mean_temp.plot()
    # plt.title(f"Mean Temperature for eta {ieta}")
    # plt.savefig(f"outputs/mhw_plot_eta_{ieta}.png")  # Save plot as an image
    # plt.close()  # Close the figure to free memory
    # -----------

    # -- Climatology
    file_clim = os.path.join(output_path_clim, f"clim_{ieta}.nc")
    climatology_surf = xr.open_dataset(file_clim)['climSST']

    # -- Thresholds
    absolute_thresholds = [1, 2, 3, 4] # Absolute threshold
    file_thresh = os.path.join(output_path_clim, f"thresh_90perc_{ieta}.nc") # Relative threshold
    relative_threshold_surf = xr.open_dataset(file_thresh)['relative_threshold']# shape:(day: 365, xi_rho: 1442)

    # -- MHW events detection
    mhw_events = {}
    for thresh in absolute_thresholds:
        mhw_events[thresh] = np.greater(ds_original_surf.values, thresh)  #Boolean
    
    mhw_rel_threshold[:,:,:] = np.greater(ds_original_surf.values, relative_threshold_surf.values) #Boolean

    # -- MHW intensity 
    mhw_intensity = ds_original_surf.values - climatology_surf.values # Anomaly relative to climatology
    # mhw_intensity[~mhw_events] = np.nan  # Mask non-MHW values

    # Reformating
    mhw_events_ds = xr.Dataset(
        data_vars=dict(
            **{
                f"mhw_abs_threshold_{thresh}_deg": (["years", "days", "xi_rho"], mhw_events[thresh]) for thresh in absolute_thresholds},
            mhw_rel_threshold=(["years", "days", "xi_rho"], mhw_rel_threshold), 
            mhw_intensity=(["years", "days", "xi_rho"], mhw_intensity), 
            ),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
            ),
        attrs = {
                'mhw_abs_threshold_i': "Detected events where T°C > absolute threshold  (i°C), boolean array",
                'mhw_rel_threshold': "Detected events where T°C > relative threshold  (90th percentile), boolean array",
                'mhw_intensity': 'Intensity of events defined as SST - climatology SST (median value obtained from a seasonally varying 11‐day moving window with 30yrs baseline (1980-2009))'
                }                
            ) 
    
    # Save output
    output_file_eta = os.path.join(output_path, f"det_{ieta}.nc")
    if not os.path.exists(output_file_eta):
        mhw_events_ds.to_netcdf(output_file_eta, mode='w')  

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for eta {ieta}: {elapsed_time:.2f} seconds")

    return mhw_events_ds 
    
# Calling function
results = Parallel(n_jobs=30)(delayed(detect_absolute_mhw)(ieta) for ieta in range(0, neta)) # detects extremes for each latitude in parallel - results (list) - ~5min computing

# %% Aggregating the different eta values
# Initialization 
det_abs = {thresh: np.full((nyears, ndays, neta, nxi), False, dtype=np.bool_) for thresh in absolute_thresholds}  # Shape: (40, 365, neta, nxi)
det_rel = np.full((nyears, ndays, neta, nxi), False, dtype=np.bool_)  # Shape: (40, 365, neta, nxi)
det_intensity = np.full((nyears, ndays, neta, nxi), False, dtype=np.float32) #Shape: (40, 365, neta, nxi)

# Loop over neta and write all eta in same Datatset - aggregation 
for ieta in range(0,neta):
    for thresh in absolute_thresholds:
        det_abs[thresh][:, :, ieta, :] = results[ieta][f"mhw_abs_threshold_{thresh}_deg"]  # Store detection for each threshold
    
    det_rel[:,:,ieta,:] = results[ieta]["mhw_rel_threshold"]
    det_intensity[:,:,ieta,:] = results[ieta]["mhw_intensity"]

det_ds = xr.Dataset(
    data_vars=dict(
        **{
            f"mhw_abs_threshold_{thresh}_deg": (["years", "days", "eta_rho", "xi_rho"], det_abs[thresh]) for thresh in absolute_thresholds},
        mhw_rel_threshold=(["years", "days", "eta_rho", "xi_rho"], det_rel), 
        mhw_intensity=(["years", "days", "eta_rho", "xi_rho"], det_intensity), 
        ),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs = {
            'mhw_abs_threshold_i': "Detected events where T°C > absolute threshold  (i°C), boolean array",
            'mhw_rel_threshold': "Detected events where T°C > relative threshold  (90th percentile), boolean array",
            'mhw_intensity': 'Intensity of events defined as SST - climatology SST (median value obtained from a seasonally varying 11‐day moving window with 30yrs baseline (1980-2009))'
            }                
        ) 


# Save output
output_file = os.path.join(output_path, f"det_all_eta.nc")
if not os.path.exists(output_file):
    det_ds.to_netcdf(output_file, mode='w') 

# del det, results

# %% ------------------ Duration
# According to Hobday et al. (2016) - MHW needs to persist for at least five days (5days of TRUE)

def compute_mhw_durations(arr):
    """
    Compute duration of consecutive Trues (MHW) and Falses (non-MHW) in a 1D boolean array.
    Return two arrays of same shape: mhw_durations, non_mhw_durations.
    """
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


def compute_mhw_durations_eta(ieta):

    print(f'Processing {ieta}')
    start_time = time.time()
    # ieta=200
    
    # Read data
    det_ds = xr.open_dataset(os.path.join(output_path, "det_all_eta.nc")).isel(eta_rho=ieta)
    det_rel_threshold = det_ds.mhw_rel_threshold # shape: (years: 40, days: 365, eta_rho: 434, xi_rho: 1442). Boolean
    det_rel_threshold_stacked = det_rel_threshold.stack(time=('years', 'days')) #consider time as continuous
    det_rel_threshold_stacked = det_rel_threshold_stacked.transpose('time', 'xi_rho')


    # Initialization
    n_time, n_xi = det_rel_threshold_stacked.shape
    mhw_durations_stacked = np.zeros((n_time, n_xi), dtype=np.int32)
    non_mhw_durations_stacked = np.zeros((n_time, n_xi), dtype=np.int32)


    # Calcualting duration
    for j in range(n_xi):
        bool_series = det_rel_threshold_stacked[:, j].values  # 1D time series 
        mhw_dur, non_mhw_dur = compute_mhw_durations(bool_series)
        mhw_durations_stacked[:, j] = mhw_dur
        non_mhw_durations_stacked[:, j] = non_mhw_dur

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for eta {ieta}: {elapsed_time:.2f} seconds")

    # Into DataArrays
    mhw_durations_da = xr.DataArray(
        mhw_durations_stacked,
        coords=det_rel_threshold_stacked.coords,
        dims=det_rel_threshold_stacked.dims,
        name="mhw_durations"
    )

    non_mhw_durations_da = xr.DataArray(
        non_mhw_durations_stacked,
        coords=det_rel_threshold_stacked.coords,
        dims=det_rel_threshold_stacked.dims,
        name="non_mhw_durations"
    )
    return mhw_durations_da, non_mhw_durations_da

results = Parallel(n_jobs=30)(delayed(compute_mhw_durations_eta)(ieta) for ieta in range(0, neta)) #computing time per eta ~2s,  in total ~4-5min

# %% Combining eta
mhw_durations_stacked, non_mhw_durations_stacked =zip(*results)

# Initialisation 
mhw_duration_all_eta_stacked_before= np.full((nyears* ndays, neta, nxi), False, dtype=np.int32)  # Shape: (40* 365, neta, nxi)
non_mhw_duration_all_eta_stacked_before= np.full((nyears* ndays, neta, nxi), False, dtype=np.int32)  # Shape: (40* 365, neta, nxi)

# Aggregation 
for ieta in range(0,neta):
    mhw_duration_all_eta_stacked_before[:,ieta,:] = mhw_durations_stacked[ieta]
    non_mhw_duration_all_eta_stacked_before[:,ieta,:] = non_mhw_durations_stacked[ieta]

# Reformat with years and days in dim
mhw_duration_all_eta_stacked_before_reshaped = mhw_duration_all_eta_stacked_before.reshape(nyears, ndays, neta, nxi)
non_mhw_duration_all_eta_stacked_before_reshaped = non_mhw_duration_all_eta_stacked_before.reshape(nyears, ndays, neta, nxi)

# Into Datasets
ds_mhw_duration_stacked_before = xr.Dataset(
    data_vars=dict(
        mhw_duration=(["years", "days",  "eta_rho", "xi_rho"], mhw_duration_all_eta_stacked_before_reshaped)
    ),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),  # (434, 1442)
    ),
    attrs={
        'mhw_duration': "Duration of event > 90th perc (not finished), int32 array"
    }
)

ds_non_mhw_duration_stacked_before = xr.Dataset(
    data_vars=dict(
        non_mhw_duration=(["years", "days", "eta_rho", "xi_rho"], non_mhw_duration_all_eta_stacked_before_reshaped)
    ),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),  # (434, 1442)
    ),
    attrs={
        'non_mhw_duration': "Duration of event < 90th perc (not finished), int32 array"
    }
)


# %% Combining MHW events when allowed
def gap_between_events(ieta, mhw, non_mhw):

# mhw = mhw_durations_stacked
# non_mhw = non_mhw_durations_stacked
# for ieta in range(0, neta):

    print(f'Processing {ieta}')
    start_time = time.time()

    # MHW last at least 5 days
    mhw_event = mhw[ieta] >= 5 
    
    # Gap of 1 day
    gap_1_day = (non_mhw[ieta] == 1) & (mhw_event.shift(time=1).astype(bool)) & (mhw_event.shift(time=-1).astype(bool)) #computation time ~4min
    gap_1_day = gap_1_day.astype(bool)

    # Gap of 2 days
    gap_2_day = (non_mhw[ieta] == 2) & (mhw_event.shift(time=2).astype(bool)) & (mhw_event.shift(time=-2).astype(bool)) #computation time ~4min
    gap_2_day = gap_2_day.astype(bool)

    # Combine events
    mhw_event_combined_stacked = mhw_event | gap_1_day | gap_2_day
    mhw_event_combined_stacked = mhw_event_combined_stacked.astype(bool)

    # Initialization
    n_time, n_xi = mhw_event_combined_stacked.shape
    a = np.zeros((n_time, n_xi), dtype=np.int32)
    non_a = np.zeros((n_time, n_xi), dtype=np.int32)

    # Recalculating duration    
    for j in range(n_xi):
        bool_series = mhw_event_combined_stacked[:, j].values  # 1D time series
        mhw_dur_ext, non_mhw_dur_ext = compute_mhw_durations(bool_series)
        a[:, j] = mhw_dur_ext
        non_a[:, j] = non_mhw_dur_ext

   # Into DataArrays
    mhw_durations_stacked_da = xr.DataArray(
        a,
        coords=mhw_event_combined_stacked.coords,
        dims=mhw_event_combined_stacked.dims,
        name="mhw_durations"
    )

    non_mhw_durations_stacked_da = xr.DataArray(
        non_a,
        coords=mhw_event_combined_stacked.coords,
        dims=mhw_event_combined_stacked.dims,
        name="non_mhw_durations"
    )

    end_time = time.time()
    diff_time = end_time - start_time
    print(f"Processing time for eta {ieta}: {diff_time:.2f} seconds")

    return mhw_durations_stacked_da, non_mhw_durations_stacked_da

duration_recalc = Parallel(n_jobs=30)(delayed(gap_between_events)(ieta, mhw_durations_stacked, non_mhw_durations_stacked) for ieta in range(0, neta)) #computing time per eta ~ 5s,  in total ~12min

# %% Combining all eta
mhw_durations_extended_stacked, non_mhw_durations_extended_stacked =zip(*duration_recalc)

# Initialisation 
mhw_duration_all_eta_stacked= np.full((nyears* ndays, neta, nxi), False, dtype=np.int32)  # Shape: (40* 365, neta, nxi)
non_mhw_duration_all_eta_stacked= np.full((nyears* ndays, neta, nxi), False, dtype=np.int32)  # Shape: (40* 365, neta, nxi)

# Aggregation ~7min
for ieta in range(0,neta):
    mhw_duration_all_eta_stacked[:,ieta,:] = mhw_durations_extended_stacked[ieta]
    non_mhw_duration_all_eta_stacked[:,ieta,:] = non_mhw_durations_extended_stacked[ieta]

# Reformat with years and days in dim
mhw_duration_all_eta_stacked_reshaped = mhw_duration_all_eta_stacked.reshape(nyears, ndays, neta, nxi)
non_mhw_duration_all_eta_stacked_reshaped = non_mhw_duration_all_eta_stacked.reshape(nyears, ndays, neta, nxi)

# Into Datasets
ds_mhw_duration_stacked = xr.Dataset(
    data_vars=dict(
        mhw_duration=(["years", "days",  "eta_rho", "xi_rho"], mhw_duration_all_eta_stacked_reshaped)
    ),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),  # (434, 1442)
    ),
    attrs={
        'mhw_duration': "Duration of event > 90th perc (at least 5days and gap of 1 and 2 days allowed), int32 array"
    }
)

ds_non_mhw_duration_stacked = xr.Dataset(
    data_vars=dict(
        non_mhw_duration=(["years", "days", "eta_rho", "xi_rho"], non_mhw_duration_all_eta_stacked_reshaped)
    ),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),  # (434, 1442)
    ),
    attrs={
        'non_mhw_duration': "Duration of event < 90th perc, int32 array"
    }
)

# Save output 
output_file = os.path.join(output_path, f"mhw_durations_extended_all_eta.nc")
if not os.path.exists(output_file):
    ds_mhw_duration_stacked.to_netcdf(output_file, mode='w')

# %% Redefine event detection 
ds_mhw_duration_stacked = xr.open_dataset(os.path.join(output_path, f"mhw_durations_extended_all_eta.nc"))
print('ds_mhw_duration_stacked:', ds_mhw_duration_stacked.isel(years=37, days=slice(190,220), xi_rho=1000, eta_rho=200).mhw_duration.values)
print("")

det_all_eta =  xr.open_dataset(os.path.join('/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/', "det_all_eta.nc"))
print('det_all_eta:', det_all_eta.isel(years=37, days=slice(190,220), xi_rho=1000, eta_rho=200).mhw_rel_threshold.values)
print("")

# If MHW duration > 0, detection=TRUE, else FALSE
mhw_rel_threshold_updated = det_all_eta.mhw_rel_threshold.where(ds_mhw_duration_stacked.mhw_duration > 0, False)
det_all_eta['mhw_rel_threshold'] = mhw_rel_threshold_updated# --- Test
print('det_all_eta after:', det_all_eta.isel(years=37, days=slice(190,220), xi_rho=1000, eta_rho=200).mhw_rel_threshold.values)
print("")

# Write 
det_all_eta.to_netcdf(os.path.join(output_path, "det_all_eta_corrected.nc"), mode='w')

# %% Visualisation of combination of events 
# Read data
ds_mhw_duration_stacked = xr.open_dataset(os.path.join(output_path, f"mhw_durations_extended_all_eta.nc"))

year_to_plot = slice(37, 39) #last idx excluded
eta_rho_to_plot = 200 
xi_rho_to_plot = 1000  

# Slicing the dataset - ~2min of computing
# Before combining events 
mhw_duration_before_to_plot = ds_mhw_duration_stacked_before.isel(years=year_to_plot, eta_rho=eta_rho_to_plot, xi_rho=xi_rho_to_plot)#dtype('int64')
mhw_duration_before_to_plot = mhw_duration_before_to_plot.stack(time=('years', 'days')).mhw_duration 
non_mhw_duration_before_to_plot = ds_non_mhw_duration_stacked_before.isel(years=year_to_plot, eta_rho=eta_rho_to_plot, xi_rho=xi_rho_to_plot) #dtype('int64')
non_mhw_duration_before_to_plot = non_mhw_duration_before_to_plot.stack(time=('years', 'days')).non_mhw_duration 

# After 
mhw_event_extended_to_plot = ds_mhw_duration_stacked.isel(years=year_to_plot, eta_rho=eta_rho_to_plot, xi_rho=xi_rho_to_plot)#dtype('int64')
mhw_event_extended_to_plot = mhw_event_extended_to_plot.stack(time=('years', 'days')).mhw_duration 

lon = mhw_event_extended_to_plot.lon_rho.values.item()
lat = mhw_event_extended_to_plot.lat_rho.values.item()

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_clip_on(False)

ax.plot(non_mhw_duration_before_to_plot, label="Non-MHW", color='#A2B36B',linestyle=":")
ax.plot(mhw_duration_before_to_plot, label="MHW", color='#DC6D04', linestyle="--")
ax.plot(mhw_event_extended_to_plot, label="Extended MHW", color='#780000', linestyle="-")

ax.set_title(f'Detection of events (surface) \nLocation: ({round(lat)}°S, {round(lon)}°E)')
ax.set_xlabel('Days')
ax.set_ylabel('Duration (days)')
# ax.set_ylim(-1,10)
# ax.set_xlim(135,165)

# Add the year labels below the x-axis
ax.annotate('', xy=(365, -65), xytext=(0, -65), arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), annotation_clip=False)
ax.annotate('', xy=(365*2, -65), xytext=(365, -65), arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), annotation_clip=False)
ax.annotate('', xy=(365*3, -65), xytext=(365*2, -65), arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), annotation_clip=False)

ax.text(182, -80, f'{1980 + year_to_plot.start}', ha='center', va='center')
ax.text(365 + 182, -80, f'{1980 + year_to_plot.start+1}', ha='center', va='center')
ax.text(365*2 + 182, -80, f'{1980 + year_to_plot.start+2}', ha='center', va='center')

ax.legend()
plt.tight_layout()
plt.show()

# %% --- Sectors computation
ds_mhw_duration_stacked["lon_rho"] = ds_mhw_duration_stacked["lon_rho"] % 360 # Ensure longitude is wrapped to 0-360° range
print(ds_mhw_duration_stacked.lon_rho.values)
print(np.isnan(ds_mhw_duration_stacked.lon_rho).sum())

spatial_domain_atl = (ds_mhw_duration_stacked.lon_rho >= 290) | (ds_mhw_duration_stacked.lon_rho < 20) #Atlantic sector: From 290°E to 20°E -OK
mask_atl = xr.DataArray(spatial_domain_atl, dims=["eta_rho", "xi_rho"]) #shape: (434, 1442)
spatial_domain_pac = (ds_mhw_duration_stacked.lon_rho >= 150) & (ds_mhw_duration_stacked.lon_rho < 290) #Pacific sector: From 150°E to 290°E -OK
mask_pac = xr.DataArray(spatial_domain_pac, dims=["eta_rho", "xi_rho"])
mask_indian = ~(mask_atl | mask_pac)


def apply_spatial_mask(yr, ds, variable, mask_da, sector):

    # Read only 1 year
    ds_yr = ds.isel(years=yr)

    # Mask
    masked_data = ds_yr[variable].where(mask_da) #, drop=True) 
    # print(masked_data.lon_rho.values)

    # Reformating - new dataset
    data_spat_filtered = xr.Dataset(
        data_vars= dict(duration_sector = (["days", "eta_rho", "xi_rho"], masked_data.data)),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], masked_data.lon_rho.values), 
            lat_rho=(["eta_rho", "xi_rho"], masked_data.lat_rho.values)
            ),
        attrs = dict(description=f'MHW duration in {sector} sector')
        ) 

    return data_spat_filtered

# Spatial mask parallelization on years - computing time ~1min30  each
spatial_mask_atl = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, ds_mhw_duration_stacked, 'mhw_duration', mask_atl, 'atlantic') for yr in range(0, nyears))
spatial_mask_pac = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, ds_mhw_duration_stacked, 'mhw_duration', mask_pac, 'pacific') for yr in range(0, nyears)) 
spatial_mask_ind = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, ds_mhw_duration_stacked, 'mhw_duration', mask_indian, 'indian') for yr in range(0, nyears))

# Put back to original dimensions
spatial_mask_atl_all = xr.concat(spatial_mask_atl, dim='years') #~20s
spatial_mask_pac_all = xr.concat(spatial_mask_pac, dim='years') #~20s
spatial_mask_ind_all = xr.concat(spatial_mask_ind, dim='years') #~7min

# Save output 
output_file1 = os.path.join(output_path, f"mhw_durations_atlantic.nc")
output_file2= os.path.join(output_path, f"mhw_durations_pacific.nc")
output_file3 = os.path.join(output_path, f"mhw_durations_indian.nc")
if not os.path.exists(output_file1):
    spatial_mask_atl_all.to_netcdf(output_file1, mode='w')
if not os.path.exists(output_file2):
    spatial_mask_pac_all.to_netcdf(output_file2, mode='w')
if not os.path.exists(output_file3):
    spatial_mask_ind_all.to_netcdf(output_file3, mode='w')


# %%  Vizualisation sectors 
# Read data
spatial_mask_atl_all = xr.open_dataset(os.path.join(output_path, f"mhw_durations_atlantic.nc"))
spatial_mask_pac_all = xr.open_dataset(os.path.join(output_path, f"mhw_durations_pacific.nc"))
spatial_mask_ind_all = xr.open_dataset(os.path.join(output_path, f"mhw_durations_indian.nc"))

# --- PLOT
from matplotlib.colors import ListedColormap

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Create the mask arrays for each sector with unique values
sector_mask = np.full_like(spatial_mask_atl_all.duration_sector.isel(years=0, days=98), np.nan)

# Assign sector values (1 for Atlantic, 2 for Pacific, 3 for Indian)
sector_mask = np.where(~np.isnan(spatial_mask_atl_all.duration_sector.isel(years=0, days=98)), 1, sector_mask)
sector_mask = np.where(~np.isnan(spatial_mask_pac_all.duration_sector.isel(years=0, days=98)), 2, sector_mask)
sector_mask = np.where(~np.isnan(spatial_mask_ind_all.duration_sector.isel(years=0, days=98)), 3, sector_mask)

sector_cmap = ListedColormap(["#778B04", "#BF3100", "#E09F3E"])

# Plot sectors
c = ax.pcolormesh(spatial_mask_atl_all.lon_rho, spatial_mask_atl_all.lat_rho, sector_mask, transform=ccrs.PlateCarree(), cmap=sector_cmap, vmin=1, vmax=3)

# Features
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
ax.set_facecolor('lightgrey')

# Atlantic-Pacific boundary (near Drake Passage)
ax.plot([-70, -70], [-90, -60], transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=2) #Atlantic sector

# Pacific-Indian boundary
ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=2) #Pacific sector

# Indian-Atlantic boundary
ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=2) #Indian sector
ax.gridlines(draw_labels=True)

plt.show()




# %% Yearly averages (from code: avg_mhw_durations.py)
ds_avg_duration = xr.open_dataset(os.path.join(output_path, f"mhw_avg_duration_yearly.nc"))
ds_avg_duration_SO = ds_avg_duration.avg_dur.mean(dim=['eta_rho', 'xi_rho'])

# Understanding averages
ds_avg_duration_values = [ds_avg_duration.avg_dur.sel(years=year).values.flatten() for year in range(0,40)]
print(f'In 2019, maximum duration in SO: {np.max(ds_avg_duration_values[39])} days')
print(f'In 2017, maximum duration in SO: {np.max(ds_avg_duration_values[37])} days')
print(f'Over the period 1980-2019, event maximum duration in SO: {np.max(ds_avg_duration_values)} days')

# -- PLOT
fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[25, 1], hspace=0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

boxplot_style = {'flierprops': {'marker': ',', 'markerfacecolor': 'black', 'markersize': 5}} #chosing marker style

# plot the same data on both Axes
ax1.boxplot(ds_avg_duration_values, positions=range(1980, 2020),  **boxplot_style,)
ax2.boxplot(ds_avg_duration_values, positions=range(1980, 2020))
ax1.plot(years, ds_avg_duration_SO) #mean line

ax1.set_ylim(1, 1200)  # ax1 ---- outliers only
ax2.set_ylim(-1, 1)  # ax2 ---- most of the data
ax1.set_yscale('log') 

# -- Labels
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)

# xlabel and xticks in ax2
ax2.set_xticks(years)
ax2.set_xticklabels(years, rotation=45)
ax2.set_xlabel('Years')
ax2.xaxis.tick_bottom()

# No ylabel in ax2
ax2.set_yticks([])
ax2.tick_params(left=False)

# ylabel and title in ax1
ax1.set_ylabel('Days (log)')
ax1.set_title("Distribution of MHW Duration Per Year", y=1.05)

# Adding maximum for each year as text
for i in range(0,40):
    ax1.text(1980+i, np.max(ds_avg_duration_values[i]) + 10, f'{int(np.max(ds_avg_duration_values[i]))}', fontsize=8, ha='left', va='bottom')

plt.show()



#%% --- Sectors 
spatial_domain_atl = (ds_avg_duration.lon_rho >= 290) | (ds_mhw_duration_stacked.lon_rho < 20) #Atlantic sector: From 290°E to 20°E -OK
mask_atl = xr.DataArray(spatial_domain_atl, dims=["eta_rho", "xi_rho"]) #shape: (434, 1442)
spatial_domain_pac = (ds_avg_duration.lon_rho >= 150) & (ds_mhw_duration_stacked.lon_rho < 290) #Pacific sector: From 150°E to 290°E -OK
mask_pac = xr.DataArray(spatial_domain_pac, dims=["eta_rho", "xi_rho"])
mask_indian = ~(mask_atl | mask_pac)

def apply_spatial_mask(yr, ds, variable, mask_da, sector):

    # Read only 1 year
    ds_yr = ds.isel(years=yr)

    # Mask
    masked_data = ds_yr[variable].where(mask_da) #, drop=True) 
    # print(masked_data.lon_rho.values)

    # Reformating - new dataset
    data_spat_filtered = xr.Dataset(
        data_vars= dict(duration_sector = (["eta_rho", "xi_rho"], masked_data.data)),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], masked_data.lon_rho.values), 
            lat_rho=(["eta_rho", "xi_rho"], masked_data.lat_rho.values)
            ),
        attrs = dict(description=f'MHW duration in {sector} sector')
        ) 

    return data_spat_filtered

avg_duration_mask_atl = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, ds_avg_duration, 'avg_dur', mask_atl, 'atlantic') for yr in range(0, nyears))
avg_duration_mask_pac = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, ds_avg_duration, 'avg_dur', mask_pac, 'pacific') for yr in range(0, nyears))
avg_duration_mask_ind = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, ds_avg_duration, 'avg_dur', mask_indian, 'indian') for yr in range(0, nyears))

# Put back to original dimensions
spatial_mask_atl_all = xr.concat(avg_duration_mask_atl, dim='years') #~20s
spatial_mask_pac_all = xr.concat(avg_duration_mask_pac, dim='years') #~20s
spatial_mask_ind_all = xr.concat(avg_duration_mask_ind, dim='years') #~7min

ds_avg_duration_atl = spatial_mask_atl_all.duration_sector.mean(dim=['eta_rho', 'xi_rho'])
ds_avg_duration_pac = spatial_mask_pac_all.duration_sector.mean(dim=['eta_rho', 'xi_rho'])
ds_avg_duration_ind = spatial_mask_ind_all.duration_sector.mean(dim=['eta_rho', 'xi_rho'])

# --- PLOT
# Adding trend lines
m, b = np.polyfit(ds_avg_duration_SO.years,ds_avg_duration_SO.values,  1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_clip_on(False)
ax.plot(ds_avg_duration_SO, label="Southern Ocean", color='black', linewidth = 1.5, linestyle="-")
ax.plot(ds_avg_duration_SO.years, m*ds_avg_duration_SO.years + b, color='black',linestyle="--" )
ax.plot(ds_avg_duration_atl, label="Atlantic", color='#778B04', linewidth = 1, linestyle="-")
ax.plot(ds_avg_duration_pac, label="Pacific", color='#BF3100', linewidth = 1, linestyle="-")
ax.plot(ds_avg_duration_ind, label="Indian", color='#E09F3E', linewidth = 1, linestyle="-")
ax.set_title(f'Spatially averaged MHW duration (surface)')
ax.set_xlabel('Years')
ax.set_ylabel('Duration (days)')
ax.legend()
years = list(range(1980, 2020, 5)) + [2019]  # Ensure 2019 is included
years.sort()  # Keep order
ax.set_xticks(np.array(years) - 1980)  # Shift to match index if necessary
ax.set_xticklabels(years)
ax.set_xlim(0, nyears-1)

plt.tight_layout()
plt.show()



# %% Visualisation map - Annual average
eta_rho_to_plot = 200 
xi_rho_to_plot = 1000  
day_to_plot= 98

# Select the slice and load in memory
mhw_plot_data = ds_mhw_duration_stacked.isel(years=3)
mhw_plot_data = ds_mhw_duration_stacked.mean(dim=['days', 'years'])
np.max(mhw_plot_data.mhw_duration)

# Select point once and extract values
point_data = mhw_plot_data.isel(eta_rho=eta_rho_to_plot, xi_rho=xi_rho_to_plot)
lon = point_data.lon_rho.compute().item()
lat = point_data.lat_rho.compute().item()
# value = point_data.compute().item()

# ---------- Plot ----------
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# -------- Plot the map --------
pcolormesh = mhw_plot_data.mhw_duration.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False, 
    # vmin=0, vmax=30,
    cmap='viridis')


# -------- Plot point --------
# sc = ax.scatter(lon, lat, c=[value], cmap='coolwarm', vmin=-5, vmax=5,
#                 transform=ccrs.PlateCarree(), s=100, edgecolor='black', zorder=3, label='Selected Cell')

# -------- Colorbar --------
cbar = plt.colorbar(pcolormesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
cbar.set_label('Duration (days)', fontsize=13)
cbar.ax.tick_params(labelsize=12)

# -------- Map features --------
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgray')
ax.set_facecolor('lightgrey')

# -------- Title --------
ax.set_title(f"SST median climatology (day={day_to_plot}) \nLocation: ({round(lat)}°S, {round(lon)}°E)", fontsize=16, pad=30)

plt.tight_layout()
plt.show()


# %%

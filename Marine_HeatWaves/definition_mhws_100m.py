#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 21 Apr 11:17:33 2025

Calculate MHW durations and thresholds for the first 100m depth 

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



# %% -------------------------------- Detecting Events --------------------------------
def detect_mhw(ieta):
    # ieta =200 #for testing

    start_time = time.time()

    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:,0:365, : ,:] #Extracts daily data : only 40yr + consider 365 days per year
    ds_original_100m = ds_original.isel(z_rho=slice(0,14))

    # Dataset initialization
    nyears, ndays, nz, nxi = ds_original_100m.shape #(40, 365, 14, 1442)
    mhw_rel_threshold = np.full((nyears, ndays, nz, nxi), np.nan, dtype=bool)

    # Deal with NANs values
    ds_original_100m.values[np.isnan(ds_original_100m.values)] = 0 # Set to 0 so that det = False at nans

    # -- Climatology
    # file_clim = os.path.join(output_path_clim, f"clim_{ieta}.nc")
    # climatology_surf = xr.open_dataset(file_clim)['climSST'] #(365, 14, 1442)

    # -- Thresholds
    absolute_thresholds = [1, 2, 3, 4] # Absolute threshold
    file_thresh = os.path.join(output_path_clim, f"thresh_90perc_{ieta}.nc") # Relative threshold
    relative_threshold_surf = xr.open_dataset(file_thresh)['relative_threshold']# shape:(365, 14, 1442)

    # -- MHW events detection
    mhw_events = {}
    for thresh in absolute_thresholds:
        mhw_events[thresh] = np.greater(ds_original_100m.values, thresh)  #Boolean
    
    mhw_rel_threshold[:,:,:,:] = np.greater(ds_original_100m.values, relative_threshold_surf.values) #Boolean

    # -- MHW intensity 
    # mhw_intensity = ds_original_100m.values - climatology_surf.values # Anomaly relative to climatology
    # mhw_intensity[~mhw_events] = np.nan  # Mask non-MHW values

    # Reformating
    mhw_events_ds = xr.Dataset(
        data_vars=dict(
            **{
                f"mhw_abs_threshold_{thresh}_deg": (["years", "days", "z_rho", "xi_rho"], mhw_events[thresh]) for thresh in absolute_thresholds},
            mhw_rel_threshold=(["years", "days", "z_rho", "xi_rho"], mhw_rel_threshold), 
            # mhw_intensity=(["years", "days", "z_rho", "xi_rho"], mhw_intensity), 
            ),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
            ),
        attrs = {
                'mhw_abs_threshold_i': "Detected events where T°C > absolute threshold  (i°C), boolean array",
                'mhw_rel_threshold': "Detected events where T°C > relative threshold  (90th percentile), boolean array"
                # 'mhw_intensity': 'Intensity of events defined as SST - climatology SST (median value obtained from a seasonally varying 11‐day moving window with 30yrs baseline (1980-2009))'
                }                
            ) 
    # np.allclose(xr.open_dataset("/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_200.nc")['mhw_abs_threshold_1_deg'].values, mhw_events_ds['mhw_abs_threshold_1_deg'].isel(z_rho=0).values) #TRUE
    
    # Save output
    output_file_eta = os.path.join(output_path, f"det_{ieta}.nc")
    if not os.path.exists(output_file_eta):
        mhw_events_ds.to_netcdf(output_file_eta, mode='w')  

    elapsed_time = time.time() - start_time
    print(f"Processing time for eta {ieta}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")

     # Free memory
    del mhw_events_ds
    gc.collect()
    
    
# Calling function
process_map(detect_mhw, range(0, neta), max_workers=30, desc="Processing eta")  # detects extremes for each latitude in parallel - computing time ~10min total
# np.unique(xr.open_dataset(f"{output_path}/det_200.nc").isel(years=slice(37,39), z_rho=0).mhw_rel_threshold) #False True

# %% -------------------------------- Combining eta --------------------------------
os.makedirs(os.path.join(output_path, "det_depth"), exist_ok=True)

def agregg(args):
    # Extracting arguments
    ieta, depth_idx = args

    start_time = time.time()
    # ieta = 200  # for testing

    # Read file and close it after processing
    with xr.open_dataset(f"{output_path}/det_{ieta}.nc") as ds:  # Using 'with' ensures the file is closed after use
        ds_depth= ds.isel(z_rho=depth_idx) #extract only 1 depth
        # Extracting data into arrays
        det_abs_eta = {thresh: ds_depth[f"mhw_abs_threshold_{thresh}_deg"].values for thresh in absolute_thresholds}
        det_rel_eta = ds_depth["mhw_rel_threshold"].values
        
    gc.collect() #free unused memory

    elapsed_time = time.time() - start_time
    print(f"Processing time for eta {ieta}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")
   
    return ieta, det_rel_eta, det_abs_eta

# Since way more data -- need another technique than just surface
nyears, ndays, nz, nxi, neta  = (40, 365, 14, 1442, 434)

# Loop through depth -- ~4-5min computing each
for depth_idx in range(nz):

    print(f'---------DEPTH{depth_idx}---------')

    # Initialization -- Move neta to axis 0 for faster computing
    det_rel = np.empty((neta, nyears, ndays, nxi), dtype=np.bool_) 
    det_abs = {thresh: np.empty((neta, nyears, ndays, nxi), dtype=np.bool_) for thresh in absolute_thresholds} 
    # det_intensity = np.empty((neta, nyears, ndays, nz, nxi), dtype=np.float32) 

    start_time = time.time()

    # ==== Combining all eta together ====
    # Calling function to combine eta -- with process_map
    for ieta, det_rel_eta, det_abs_eta in process_map(agregg, [(ieta, depth_idx) for ieta in range(neta)], max_workers=30, desc="Combining eta", chunksize=1):
        det_rel[ieta] = det_rel_eta
        for thresh in absolute_thresholds:
            det_abs[thresh][ieta] = det_abs_eta[thresh]

        # Free memory
        del det_rel_eta, det_abs_eta
        gc.collect()

    # Check
    print('Combine eta -- ', np.unique(det_rel[200, 37:39,:,:])) #should be [False  True]

    # Transpose dimension to have eta on 4th position
    det_abs_transposed = {thresh: det_abs[thresh].transpose(1, 2, 0, 3) for thresh in absolute_thresholds} #shape (40, 365, 434, 1442)
    det_rel_transposed = det_rel.transpose(1, 2, 0, 3)

    # Check
    print('Transposition -- ', np.unique(det_rel_transposed[37:39,:, 200,:])) #should be [False  True]

    # Free memory
    del det_abs, det_rel
    gc.collect() 
    
    # ==== Write per depth ====
    ds = xr.open_dataset(path_mhw + file_var + 'eta200.nc') #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    all_depths= ds['z_rho'].values

    # Select data
    data_vars = {
        **{f"mhw_abs_threshold_{thresh}_deg": (["years", "days", "eta_rho", "xi_rho"], det_abs_transposed[thresh]) for thresh in absolute_thresholds},
        "mhw_rel_threshold": (["years", "days", "eta_rho", "xi_rho"], det_rel_transposed[:, :, :, :])
    }

    # Write to dataset
    ds_depth = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),
        ),
        attrs=dict(
            depth_index=depth_idx,
            description=f"MHW detection data for {all_depths[depth_idx]}m depth, layer {depth_idx}"
        )
    )

    print('To dataset -- ', np.unique(ds_depth.isel(years=slice(37, 39), eta_rho=200).mhw_rel_threshold)) #should be [False  True]

    # Save to file
    output_file_depth = os.path.join(output_path, f"det_depth/det_{-all_depths[depth_idx]}m.nc")
    if not os.path.exists(output_file_depth):
        ds_depth.to_netcdf(output_file_depth, mode='w')  

    # Free memory
    del ds_depth
    gc.collect()

    elapsed_time = time.time() - start_time
    print(f"Processing time for depth {depth_idx}: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")

# xr.open_dataset(os.path.join(output_path, f"det_depth/det_90m.nc"))

# %% -------------------------------- Erase all files for single eta --------------------------------
import glob
files_to_remove = glob.glob(os.path.join(output_path, "det_*.nc"))

for file in files_to_remove:
    os.remove(file)
    print(f"Removed: {file}")


# %% Visualisation map - detection on multiple depths
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Datasets at 3 depths
det_ds_5m = xr.open_dataset(output_path+'det_depth/det_5m.nc')
det_ds_58m = xr.open_dataset(output_path+'det_depth/det_58m.nc')
det_ds_100m = xr.open_dataset(output_path+'det_depth/det_100m.nc')

# Threshold information
threshold_titles = ["T $>$ 1°C", "T $>$ 2°C", "T $>$ 3°C", "T $>$ 4°C", "T $>$ 90th perc"]
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808', '#7F9C96']

# Datasets
datasets = [det_ds_5m, det_ds_58m, det_ds_100m]

# Year and day selection
year_to_plot = 2017 - 1980
day_to_plot = 15 # 15 January

# Create figure and axes
fig, axes = plt.subplots(
    nrows=3, ncols=5,
    figsize=(18, 9),
    subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)},
    gridspec_kw={'wspace': 0.0, 'hspace': 0.05, 'left': 0.02, 'right': 0.98, 'bottom': 0.1, 'top': 0.95}
)

axes = axes.reshape(3, 5)

# Loop over depth (rows)
for irow, det_ds in enumerate(datasets):
    
    ds_to_plot = [
        det_ds.mhw_abs_threshold_1_deg,
        det_ds.mhw_abs_threshold_2_deg,
        det_ds.mhw_abs_threshold_3_deg,
        det_ds.mhw_abs_threshold_4_deg,
        det_ds.mhw_rel_threshold
    ]
    
    # Loop over thresholds (columns)
    for icol, (ax, color, data) in enumerate(zip(axes[irow], threshold_colors, ds_to_plot)):
        
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        
        # Circular boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Custom 2-color colormap
        cmap = ListedColormap(["lightgray", color])
        
        data_sel = data.isel(years=year_to_plot, days=day_to_plot)
        
        data_sel.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            x="lon_rho", y="lat_rho",
            add_colorbar=False,
            cmap=cmap,
            vmin=0, vmax=1
        )
        
        ax.coastlines(color='black', linewidth=1.5, zorder=1)
        ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgray')
        ax.set_facecolor('lightgrey')
        
        # Label rows with depth (on left)
        if icol == 0:
            depth_label = {0: "5 m", 1: "58 m", 2: "100 m"}[irow]
            ax.text(-0.1, 0.5, depth_label, transform=ax.transAxes,
                    ha='right', va='center', fontsize=18, fontweight='bold', rotation=90)

# ======= Add one legend for whole figure =======
# Create legend handles
threshold_patches = [
    mpatches.Patch(color=threshold_colors[0], label=threshold_titles[0]),
    mpatches.Patch(color=threshold_colors[1], label=threshold_titles[1]),
    mpatches.Patch(color=threshold_colors[2], label=threshold_titles[2]),
    mpatches.Patch(color=threshold_colors[3], label=threshold_titles[3]),
    mpatches.Patch(color=threshold_colors[4], label=threshold_titles[4])
]

handles = [
    mpatches.Patch(color="lightgray", label="No Detection"),
] + threshold_patches

# Position of the legend at the bottom of the figure
fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    fontsize=13,
    frameon=False,
    title="Detection Thresholds",
    title_fontsize=14,
    ncol=6  #single row
)

# ======= Add title for the whole figure =======
fig.suptitle('Detection of Marine Heatwaves at Different Depths and Thresholds (15th January 2017)', 
             fontsize=22, y=1.05)


plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
# plt.show()
# Converts all plot elements to raster inside the PDF --> reducing size while keeping the vector type
for ax in plt.gcf().get_axes():
    for artist in ax.get_children():
        if hasattr(artist, 'set_rasterized'):
            artist.set_rasterized(True)

plt.savefig(os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/defining_mhw_100m/example_event_detection_3depths.pdf'),
            format='pdf', dpi=150, bbox_inches='tight')


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

for depth_idx in range(14): # computing time ~10min per depth

    print(f'---------DEPTH{depth_idx}---------')

    start_time = time.time()
    # depth_idx=0
    
    # Read data - only relative threshold
    det_ds_depth= xr.open_dataset(os.path.join(output_path, f"det_depth/det_{-all_depths[depth_idx]}m.nc")) #read each depth
    # det_ds_eta = xr.open_dataset(os.path.join(output_path, f"det_depth/det_{ieta}.nc")) #read each eta
    det_rel_thres = det_ds_depth.mhw_rel_threshold # shape: (40, 365, 434, 1442). Boolean
    det_rel_thres_stacked = det_rel_thres.stack(time=('years', 'days')) #consider time as continuous
    det_rel_thres_stacked = det_rel_thres_stacked.transpose('time', 'eta_rho', 'xi_rho') #time as first dim

    # Initialization
    ntime, neta, nxi = det_rel_thres_stacked.shape
    # mhw_stacked = np.zeros((ntime, neta, nxi), dtype=np.int32)
    # non_mhw_stacked = np.zeros((ntime, neta, nxi), dtype=np.int32)
    det_rel_thres_reshaped = det_rel_thres_stacked.values.reshape(ntime, neta * nxi) #(14600, 625828)
    mhw_durations_all = np.zeros_like(det_rel_thres_reshaped, dtype=np.int32)
    non_mhw_durations_all = np.zeros_like(det_rel_thres_reshaped, dtype=np.int32)

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
    for i in range(det_rel_thres_reshaped.shape[1]): # Loop over grid points (1D)
        mhw_durations_all[:, i], non_mhw_durations_all[:, i] = compute_mhw_durations(det_rel_thres_reshaped[:, i])

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
            'description': 'Durations recalculated using Hobday rules (min 5-day events, 1-2 day gaps allowed)'}
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



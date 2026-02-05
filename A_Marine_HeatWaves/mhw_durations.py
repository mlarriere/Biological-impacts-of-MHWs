# -*- coding: utf-8 -*-
"""
Created on Mon 19 May 09:37:56 2025

Calculate MHW durations for the first 100m depth 

Note: We need to define the duration based on the condition that a mhw event is when T°C > absolute AND relative thresholds 

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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta
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
path_fixed_baseline = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'
path_duration = os.path.join(path_fixed_baseline, 'mhw_durations')

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


def apply_hobday_rules(bool_event):
    # test
    # bool_event = mhw_rel_only.values

    print('Initial: ', bool_event[38*365: 38*365 + 30, 224, 583]) #test

    ntime, neta, nxi = bool_event.shape

    # --- Calculate initial durations
    # Initialization
    reshaped = bool_event.reshape(ntime, neta * nxi) #shape (14600, 625828)
    mhw_dur = np.zeros_like(reshaped, dtype=np.int32) # ~30s
    non_mhw_dur = np.zeros_like(reshaped, dtype=np.int32) #~7min

    for i in range(reshaped.shape[1]):
        mhw_dur[:, i], non_mhw_dur[:, i] = compute_mhw_durations(reshaped[:, i]) # ~30min
    print('First duration calculation: ', mhw_dur[38*365: 38*365 + 30, 224 * nxi + 583]) #test

    # Reshape
    mhw_dur = mhw_dur.reshape(ntime, neta, nxi) 
    non_mhw_dur = non_mhw_dur.reshape(ntime, neta, nxi) 

    # --- Apply Hobday rules
    # MHW last at least 5 days
    mhw_event = mhw_dur >= 5 

    # Detecting gaps of 1 day 
    mhw_prev = np.roll(mhw_event, 1, axis=0)
    mhw_next = np.roll(mhw_event, -1, axis=0)
    mhw_prev[0] = False
    mhw_next[-1] = False

    gap_1 = (non_mhw_dur == 1) & mhw_prev & mhw_next #~1min
    print('Gap 1day: ', gap_1[38*365: 38*365 + 30, 224, 583]) #test

    # Detecting gaps of 2 days 
    mhw_prev2 = np.roll(mhw_event, 2, axis=0)
    mhw_next2 = np.roll(mhw_event, -2, axis=0)
    mhw_prev2[:2] = False
    mhw_next2[-2:] = False

    gap_2 = (non_mhw_dur == 2) & mhw_prev2 & mhw_next2 #~7min
    print('Gap 2days: ', gap_2[38*365: 38*365 + 30, 224, 583]) #test

    # Combine events, i.e. allowing gaps of 1 and 2 days between 2 MHW events lasting more than 5 days
    mhw_combined = mhw_event | gap_1 | gap_2 #~5min

    # --- Recompute final durations
    reshaped = mhw_combined.reshape(ntime, neta * nxi)
    mhw_final = np.zeros_like(reshaped, dtype=np.int32) #~18min

    for i in range(reshaped.shape[1]):
        mhw_final[:, i], _ = compute_mhw_durations(reshaped[:, i]) #~50min
    print('Duration 2nd calculation: ', mhw_final[38*365: 38*365 + 30, 224 * nxi + 583]) #test

    return mhw_final.reshape(ntime, neta, nxi)


ds = xr.open_dataset(path_mhw + file_var + 'eta200.nc') #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
all_depths= ds['z_rho'].values

def mhw_duration(depth_idx):
    depth_idx=0

    print(f'---------DEPTH{depth_idx}---------')

    start_time = time.time()
    
    # Read data - only relative threshold
    det_ds_depth= xr.open_dataset(os.path.join(path_fixed_baseline, f"det_depth/det_{-all_depths[depth_idx]}m.nc")) #read each depth

    # ----- MHW DURATION BASED ON ONLY RELATIVE THRESHOLDS (90th percentile)-----
    output_file_rel = os.path.join(path_duration, f"mhw_duration_90th_{-all_depths[depth_idx]}m.nc")
    if os.path.exists(output_file_rel):
        print(f"File already exists for depth {depth_idx} (90th percentile only).")
        return
    
    # Relative-only MHWs
    det_rel_thres = det_ds_depth.mhw_rel_threshold # shape: (40, 365, 434, 1442). Boolean
    print(det_rel_thres.isel(eta_rho=224, xi_rho=583, years=38, days=slice(0,30)).values)
    mhw_rel_only = det_rel_thres.stack(time=('years', 'days')).transpose('time','eta_rho','xi_rho') #shape (14000, 434, 1442)
    mhw_rel_only_final = apply_hobday_rules(mhw_rel_only.values) #shape (14600, 434, 1442)

    # Reformat with years and days in dim
    mhw_rel_recalc_reshaped = mhw_rel_only_final.reshape((40, 365, neta, nxi))

    # To dataset
    ds_intermediate_rel = xr.Dataset({"duration": (("years", "days", "eta_rho", "xi_rho"), mhw_rel_recalc_reshaped)},
                                     coords={"lon_rho":(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
                                             "lat_rho":(["eta_rho", "xi_rho"], ds_roms.lat_rho.values)})  # (434, 1442)
    # Add attributes
    ds_intermediate_rel.attrs = {"Description": "MHW durations calculated using Hobday et al. (2016) rules (≥5days and 1-2 day gaps allowed). "\
                                                 "Relative threshold (90th percentile) only."}
    
    # Save to file
    ds_intermediate_rel.to_netcdf(output_file_rel, engine="netcdf4")
    print(f"File written (90th percentile only): {depth_idx}")


    # ----- MHW DURATION BASED ON BOTH RELATIVE AND ABSOLUTE THRESHOLDS (1°C)-----
    output_file = os.path.join(path_duration, f"mhw_duration_{-all_depths[depth_idx]}m.nc")
    if os.path.exists(output_file):
        print(f"File already exists for depth {depth_idx} (90th percentile and 1°C).")
        return

    # Intersection between relative and 1°C threshold (min abs thresh) to define mhw duration
    det_abs_1deg = det_ds_depth.mhw_abs_threshold_1_deg # shape: (40, 365, 434, 1442). Boolean
    mhw_both_thresholds = np.logical_and(det_rel_thres, det_abs_1deg) #rel thresh is True if >1°C else False ~1min computing
    print(mhw_both_thresholds.isel(eta_rho=224, xi_rho=583, years=38, days=slice(0,30)).values)
    mhw_rel_abs = mhw_both_thresholds.stack(time=('years', 'days')).transpose('time','eta_rho','xi_rho')
    mhw_rel_abs_final = apply_hobday_rules(mhw_rel_abs.values)

    # Reformat with years and days in dim
    mhw_rel_abs_final = mhw_rel_abs_final.reshape((40, 365, neta, nxi))

    # To dataset
    ds_intermediate = xr.Dataset({ "duration": (("years", "days", "eta_rho", "xi_rho"), mhw_rel_abs_final)},
                                 coords={"lon_rho":(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),  # (434, 1442)
                                         "lat_rho":(["eta_rho", "xi_rho"], ds_roms.lat_rho.values)}),  # (434, 1442)
    
    # Save to file
    ds_intermediate.to_netcdf(output_file, engine="netcdf4")

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
    print(ds_out.mhw_durations.isel(eta_rho=224, xi_rho=583, years=38, days=slice(0,30)).values)

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
    output_file = os.path.join(path_duration, f"mhw_duration_{-all_depths[depth_idx]}m.nc")
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

process_map(mhw_duration, range(3,6), max_workers=6, desc="Processing depth")  # detects extremes for each latitude in parallel - computing time ~10min total



#%% =============== Mean duration over hincast ===============
det_combined_ds= xr.open_dataset( os.path.join(path_det, f"duration_AND_thresh_5mFULL.nc"))
thresholds = ["det_1deg", "det_2deg", "det_3deg", "det_4deg"]

mean_duration_hindcast = xr.Dataset({
    t: det_combined_ds["duration"].where(det_combined_ds[t] > 0).mean(dim="years", skipna=True)
    for t in thresholds
})

# Collapse the days dimension to get one mean duration per grid cell
mean_duration_plot = mean_duration_hindcast.mean(dim="days", skipna=True)

#%% =============== Mean duration Plot ===============
plot = 'slides'  # slides report

# --------- Figure layout ---------
if plot == 'report':
    fig_width = 6.3228348611*0.5
    fig_height = 9.3656988889 #674.33032pt
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(4, 1, hspace=0.4, wspace=0)  # 4 rows, 1 column

    axs = []
    axs.append(fig.add_subplot(gs[0, 0], projection=ccrs.SouthPolarStereo()))  # 1°C
    axs.append(fig.add_subplot(gs[1, 0], projection=ccrs.SouthPolarStereo()))  # 2°C
    axs.append(fig.add_subplot(gs[2, 0], projection=ccrs.SouthPolarStereo()))  # 3°C
    axs.append(fig.add_subplot(gs[3, 0], projection=ccrs.SouthPolarStereo()))  # 4°C

elif plot == 'slides':
    fig_width = 6.3228348611
    fig_height = fig_width
    fig = plt.figure(figsize=(fig_width * 5, fig_height))  # 5 columns wide
    gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)
    axs = [fig.add_subplot(gs[0, j], projection=ccrs.SouthPolarStereo()) for j in range(4)]

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}

plot_data = [(mean_duration_plot['det_1deg'], r"MHWs $\ge$ 1$^\circ$C"),
            (mean_duration_plot['det_2deg'], r"MHWs $\ge$ 2$^\circ$C"),
            (mean_duration_plot['det_3deg'], r"MHWs $\ge$ 3$^\circ$C"),
            (mean_duration_plot['det_4deg'], r"MHWs $\ge$ 4$^\circ$C")]

# --- Color Setup ---
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm

# Define bin
bins = [0, 5, 10, 15, 20, 25, 30]  
n_colors = len(bins) - 1 

colors = ["#F4E9CD", "#9DBEBB", "#5F9796", "#468189", "#254D58", "#031926"]  
# cmap = LinearSegmentedColormap.from_list("duration", colors, N=256)
cmap = LinearSegmentedColormap.from_list("duration", colors, N=n_colors)
norm = BoundaryNorm(boundaries=bins, ncolors=n_colors)

# norm = mcolors.Normalize(vmin=0, vmax=100)


# --------- Plot ---------
for i, (data, title) in enumerate(plot_data):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map features
    lw = 1 if plot == 'slides' else 0.5
    ax.coastlines(color='black', linewidth=lw, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=lw, zorder=5)

    # Gridlines
    lw_grid = 0.7 if plot == 'slides' else 0.3
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=lw_grid, zorder=3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    # Font size settings for gridline labels
    gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # CHLA data
    im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                       transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, 
                       shading='auto', zorder=1, rasterized=True)
    
    # ax.set_title(title, **title_kwargs)
    if plot == 'report':
        ax.text(1.5, 0.5,  title,
            rotation=-90, va='center', ha='center',
            transform=ax.transAxes, **subtitle_kwargs)
    else:
        ax.set_title(title, **subtitle_kwargs)
# --------- Colorbar ---------
if plot == 'report':
    cbar_kwargs = {'fraction': 0.02, 'pad': 0.06, 'aspect': 50}
else:
    cbar_kwargs = {'fraction': 0.05, 'pad': 0.07, 'aspect': 40}

if plot == 'report':
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', extend='max', location='bottom', fraction=0.025, pad=0.04, shrink=0.9)
else: 
    cbar = fig.colorbar(im, ax=axs, extend='max', orientation='vertical', fraction=0.025, pad=0.04, shrink=0.9)


# cbar = fig.colorbar(im, ax=axs, orientation='horizontal', extend='max', **cbar_kwargs)
cbar.set_label("Mean Duration [days]", **label_kwargs)
cbar.ax.tick_params(**tick_kwargs)

# --------- Title and subtitle ---------
if plot == 'report':
    suptitle_y = 0.98
    fig.suptitle(f'Mean MHW duration', y=suptitle_y, **maintitle_kwargs)
    # fig.text(0.5, suptitle_y - 0.05, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', **title_kwargs, style='italic')
else:
    suptitle_y = 1.05
    fig.suptitle(f'Mean MHW duration', y=suptitle_y, **maintitle_kwargs)
    # fig.text(0.5, suptitle_y - 0.08, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', fontsize=17, style='italic')


# --------- Output handling ---------
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/MHWs_metrics/')
    outfile = f"mhw_duration_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/MHWs_metrics/mhw_duration_{plot}.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    # plt.show()
# %%


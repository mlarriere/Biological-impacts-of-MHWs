#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 17 March 08:55:12 2025

Detect MHW in ROMS-SO using absolute threshold

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
    "text.latex.preamble": r"\usepackage{mathptmx} \usepackage[x11names, dvipsnames, table]{xcolor}",
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
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'
path_det_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer'
os.makedirs(os.path.join(path_det_summer, "combined_thresholds"), exist_ok=True)
path_combined_thesh= '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/combined_thresholds'

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
doy_list = list(range(305, 365)) + list(range(0, 121))
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)


# %% Combined relative and absolute thresholds
det_files = glob.glob(os.path.join(path_det_summer, "det_*.nc"))

def combine_thresh(file):
    start_time = time.time()

    file =  '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/det_depth5m.nc'

    # Retrieve depth of file as string
    basename = os.path.basename(file)
    depth_str = basename.split('_')[1].replace('m.nc', '')  # gets '5', '11', etc.
    print(f'Depth being processed: {depth_str}\n')

    # Load data
    det_ds = xr.open_dataset(file) #-- No NaNs value in the dataset

    absolute_thresh_extended=True

    if absolute_thresh_extended:
        file =  '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/det5m_extended.nc'

        # Load data
        det_ds_extended = xr.open_dataset(os.path.join(path_det, 'det5m_extended.nc')) # -- No NaNs value in the dataset

        # Check of inconsistencies
        # print(det_ds_extended.det_1deg_extended.sum()>=det_ds_extended.det_2deg_extended.sum()>=det_ds_extended.det_3deg_extended.sum()>= det_ds_extended.det_4deg_extended.sum()) #True

        # Combine thresholds
        det_1deg = np.where(det_ds.mhw_rel_threshold &  det_ds_extended.det_1deg_extended, True, False)
        det_2deg = np.where(det_ds.mhw_rel_threshold & det_ds_extended.det_2deg_extended, True, False)
        det_3deg = np.where(det_ds.mhw_rel_threshold & det_ds_extended.det_3deg_extended, True, False)
        det_4deg = np.where(det_ds.mhw_rel_threshold & det_ds_extended.det_4deg_extended, True, False)

    else:
            
        # Check of inconsistencies
        # print(det_ds.mhw_abs_threshold_1_deg.sum()>=det_ds.mhw_abs_threshold_2_deg.sum()>=det_ds.mhw_abs_threshold_3_deg.sum()>= det_ds.mhw_abs_threshold_4_deg.sum()) #True

        # Combine thresholds
        det_1deg = np.where(det_ds.mhw_rel_threshold &  det_ds.mhw_abs_threshold_1_deg, True, False)
        det_2deg = np.where(det_ds.mhw_rel_threshold & det_ds.mhw_abs_threshold_2_deg, True, False)
        det_3deg = np.where(det_ds.mhw_rel_threshold & det_ds.mhw_abs_threshold_3_deg, True, False)
        det_4deg = np.where(det_ds.mhw_rel_threshold & det_ds.mhw_abs_threshold_4_deg, True, False)

    # Check of inconsistencies
    print('Detection 1°C > 2°C > 3°C > 4°C -- ', det_1deg.sum()>=det_2deg.sum()>=det_3deg.sum()>=det_4deg.sum()) #True

    # Reformating
    det_combined_ds= xr.Dataset(
        data_vars=dict(
            det_1deg = (["years", "days", "eta_rho" ,"xi_rho"], det_1deg), #shape (40, 181, 434, 1442)
            det_2deg = (["years", "days", "eta_rho" ,"xi_rho"], det_2deg),
            det_3deg = (["years", "days", "eta_rho" ,"xi_rho"], det_3deg),
            det_4deg = (["years", "days", "eta_rho" ,"xi_rho"], det_4deg)
            ),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
            original_days=(['days'], det_ds.coords['days'].values), # Keeping information on day 
            ),
        attrs = {
                'det_ideg': "Detected events where SST > (absolute threshold (i°C) AND 90th percentile) , boolean array"
                }                
            ) 

    # Save file
    if absolute_thresh_extended:
        output_file = os.path.join(path_combined_thesh, f"det_{depth_str}m_extended.nc")
    else:
        output_file = os.path.join(path_combined_thesh, f"det_{depth_str}m.nc")
    if not os.path.exists(output_file):
        try:
            det_combined_ds.to_netcdf(output_file, engine="netcdf4")
            print(f"File written: {depth_str}")
        except Exception as e:
            print(f"Error writing {depth_str}: {e}")    
    
    elapsed_time = time.time() - start_time
    print(f"Processing time for {depth_str}m: {elapsed_time:.2f} secs, Memory used: {psutil.virtual_memory().percent}%")


# Calling function in parallel
process_map(combine_thresh, det_files, max_workers=30, desc="Processing file")  #computing time ~1min per file

# det_combined_ds= xr.open_dataset( os.path.join(path_combined_thesh, f"det_depth5m_extended.nc"))
# %% Visualization/ Explanation of threshold combination
# Interesting locations
choice_eta = 211 #200 #200, 220, 190
choice_xi =  989 #1000 #1000, 950, 600
day_to_plot = 305 # 305, 98, 67
year_to_plot = 37
choice_year = slice(36,40)
depth_to_plot= 0

all_depths = xr.open_dataset(path_temp + file_var + 'eta200.nc')['z_rho'].values 

# --- NOTE: we select the full year  --- to illustrate!

# Surface temperature for 1 location (eta, xi)
selected_temp_surf = xr.open_dataset(path_temp + file_var + 'eta' + str(choice_eta) + '.nc')[var][:, 0:365, depth_to_plot, :].sel(xi_rho=choice_xi)  # 30yrs - 365days per year
selected_temp_surf = selected_temp_surf.isel(year=choice_year)
selected_temp_surf = selected_temp_surf.stack(time=['year', 'day'])

# 90th percentile threshold for 1 location (eta, xi)
selected_rel_thresh_surf = xr.open_dataset(path_clim + 'thresh_90perc_' + str(choice_eta) + '.nc').sel(xi_rho=choice_xi, z_rho=depth_to_plot).relative_threshold  
selected_rel_thresh_surf=np.tile(selected_rel_thresh_surf, choice_year.stop-choice_year.start) #repeat the relative threshold 

# Plot settings
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']


# %% ===== PLOT1 - time series =====
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width / 2  # Or set manually
# fig, ax = plt.subplots(figsize=(fig_width, fig_height))
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(selected_temp_surf, color='black', linewidth=1, label='SST')
ax.plot(selected_rel_thresh_surf, color='#7832AE', linewidth=1, label='90th perc')

ax.hlines(y=1, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[0], linewidth=1)
ax.hlines(y=2, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[1], linewidth=1)
ax.hlines(y=3, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[2], linewidth=1)
ax.hlines(y=4, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[3], linewidth=1)

# Condition: temp>90th perc and i°C 
sst_values = selected_temp_surf.values  # to np array - shape (1825,)
threshold_values = selected_rel_thresh_surf  # to np array - shape (1825,)

for temp, color, alpha in zip([1, 2, 3, 4], threshold_colors[:], [0.7, 0.7, 0.9, 0.9]):
    condition = (sst_values >= threshold_values) & (sst_values >= temp)
    sst_masked = np.where(condition, sst_values, np.nan)
    threshold_masked = np.where(condition, threshold_values, np.nan)
    ax.fill_between(np.arange(len(sst_values)), threshold_masked, sst_masked, color=color, alpha=alpha)

# Highlight specific day (plot after)
time_values = np.array([f"{year}-{day:03d}" for year, day in selected_temp_surf.time.values])

day_map = f"2017-{day_to_plot:03d}" 
idx = np.where(time_values == day_map)[0]

ticks_years = np.arange(0, 365 * (choice_year.stop + 1 - choice_year.start), 365)
tick_labels = np.append(np.unique([int(time_values[i].split("-")[0]) for i in range(time_values.shape[0])]), 2019)

if idx.size > 0:
    day_index = idx[0]
    new_ticks = np.append(ticks_years, day_index)  # Add custom tick
    ax.set_xticks(new_ticks)
    
    # Use existing labels but leave space for custom tick
    new_labels = list(map(str, tick_labels)) + [""]
    ax.set_xticklabels(new_labels, fontsize=10)
    
    ax.text(day_index, ax.get_ylim()[0] - 0.35, date_dict[day_to_plot], color='red', ha='center') # Tick in red
    ax.tick_params(axis='x', direction='in', length=5) # Small tick visible


ax.set_xlim(0, 365 * (choice_year.stop - choice_year.start))
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Temperature (°C)',  fontsize=12)
ax.set_title(
    f'Combining absolute and relative thresholds' "\n"
    rf'Location: ({round(selected_temp_surf.lat_rho.item())}°S, {round(selected_temp_surf.lon_rho.item())}°E) at $\mathbf{{{-all_depths[depth_to_plot]}\ \mathrm{{m\ depth}}}}$',
    fontsize=20,
    y=1
)

ax.legend(loc='upper right', bbox_to_anchor = (1, 1), fontsize=12)

plt.tight_layout()
plt.show()
# Converts all plot elements to raster inside the PDF --> reducing size while keeping the vector type
# for ax in plt.gcf().get_axes():
#     for artist in ax.get_children():
#         if hasattr(artist, 'set_rasterized'):
#             artist.set_rasterized(True)

# plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/combined_thresholds/comb_thresh_illustr_{-round(selected_temp_surf.lat_rho.item())}S_{round(selected_temp_surf.lon_rho.item())}E_{-all_depths[depth_to_plot]}m.pdf'),
#             format='pdf', bbox_inches='tight')

# ===== PLOT2 - map COMBINED thresholds =====
from matplotlib.lines import Line2D      
import matplotlib.patches as mpatches

det_combined_ds = xr.open_dataset(os.path.join(path_combined_thesh, f"det_depth{-all_depths[depth_to_plot]}m.nc"))
ds_to_plot = [det_combined_ds.det_1deg, det_combined_ds.det_2deg, det_combined_ds.det_3deg, det_combined_ds.det_4deg]

titles = [
    r"T $> 1^\circ\mathrm{C} \,\&\, 90^\mathrm{th}$ perc",
    r"T $> 2^\circ\mathrm{C} \,\&\, 90^\mathrm{th}$ perc",
    r"T $> 3^\circ\mathrm{C} \,\&\, 90^\mathrm{th}$ perc",
    r"T $> 4^\circ\mathrm{C} \,\&\, 90^\mathrm{th}$ perc",
]

threshold_patches = [mpatches.Patch(color=col, label=title) for title, col in zip(titles, threshold_colors)]
no_detection_patch = mpatches.Patch(color="lightgray", label="No Detection")

# --- PLOT
fig, axes = plt.subplots(
    nrows=1, ncols=4,
    figsize=(18, 9),
    # figsize=(fig_width, fig_height), 
    subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)},
    gridspec_kw={'wspace': 0.3, 'hspace': 0.05, 'left': 0.02, 'right': 0.98, 'bottom': -0.1, 'top': 1.05}
)

axs = axes.flatten()

# Loop through each subplot and dataset
for i, (ax, dataset, title, col) in enumerate(zip(axs, ds_to_plot, titles, threshold_colors)):
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular map boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Plot the data for the current day
    pcolormesh = dataset.isel(years=year_to_plot, days= dataset.coords['original_days'].values.tolist().index(day_to_plot)).plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x="lon_rho", y="lat_rho",
        add_colorbar=False,
        cmap=plt.matplotlib.colors.ListedColormap(['lightgray', col])
    )

    point_data = dataset.isel(eta_rho=choice_eta, xi_rho=choice_xi, years=year_to_plot, days=dataset.coords['original_days'].values.tolist().index(day_to_plot))
    point_color = col if point_data.item() else 'lightgray'  # Adjust the color condition as needed
    sc = ax.scatter(point_data.lon_rho.item(), point_data.lat_rho.item(), c=[point_color], cmap=plt.matplotlib.colors.ListedColormap([point_color]),
                    transform=ccrs.PlateCarree(), s=30, edgecolor='black', zorder=3, label='Selected Cell')

    # Map features
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Legend
    fig.legend(
        handles=[no_detection_patch] + threshold_patches,
        loc='lower center',
        # bbox_to_anchor=(0.5, 0.09),
        bbox_to_anchor=(0.5, 0.15),
        fontsize=13,
        frameon=False,
        title="Detection Thresholds",
        title_fontsize=14,
        ncol=6,  #single row
        handlelength=1.0,     # shorter handle bar
        handleheight=0.8,     # smaller box height
        handletextpad=0.4,    # less space between color and text
        columnspacing=0.8,    # tighter columns
    )

    # Atlantic-Pacific boundary (near Drake Passage)
    ax.plot([-85, -85], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Atlantic sector

    # Pacific-Indian boundary
    ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Pacific sector

    # Indian-Atlantic boundary
    ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Indian sector

    ax.set_title("")

# Title for whole figure
plt.suptitle(
    rf"Temperature above relative and absolute thresholds" "\n"
    rf"Snapshot: {date_dict[day_to_plot]}, {1980 + year_to_plot}, at $\mathbf{{{abs(all_depths[depth_to_plot])}}}\ \mathrm{{m\ depth}}$",
    fontsize=16,
    y=0.8

)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/combined_thresholds/comb_thresh_map_{-round(selected_temp_surf.lat_rho.item())}S_{round(selected_temp_surf.lon_rho.item())}E_{-all_depths[depth_to_plot]}m.pdf'),
#             format='pdf',bbox_inches='tight')

# %% Visualization - map ABSOLUTE thresholds
# titles = ["SST > 1°C", "SST > 2°C",  "SST > 3°C", "SST > 4°C"]
# threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
# ds_to_plot = [det_ds.mhw_abs_threshold_1_deg, det_ds.mhw_abs_threshold_2_deg, det_ds.mhw_abs_threshold_3_deg, det_ds.mhw_abs_threshold_4_deg]

# # --- PLOT
# fig, axs = plt.subplots(1, 4, figsize=(15, 10), subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)})
# axs = axs.flatten()

# # Loop through each subplot and dataset
# for i, (ax, dataset, title, col) in enumerate(zip(axs, ds_to_plot, titles, threshold_colors)):
#     ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

#     # Circular map boundary
#     theta = np.linspace(0, 2 * np.pi, 100)
#     center, radius = [0.5, 0.5], 0.5
#     verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#     circle = mpath.Path(verts * radius + center)
#     ax.set_boundary(circle, transform=ax.transAxes)

#     # Plot the data for the current day
#     pcolormesh = dataset.isel(years=year_to_plot, days=day_to_plot).plot.pcolormesh(
#         ax=ax, transform=ccrs.PlateCarree(),
#         x="lon_rho", y="lat_rho",
#         add_colorbar=False,
#         cmap=plt.matplotlib.colors.ListedColormap(['lightgray', col])
#     )

#     point_data = dataset.isel(eta_rho=choice_eta, xi_rho=choice_xi, years=year_to_plot, days=day_to_plot)
#     point_color = col if point_data.item() else 'lightgray'  # Adjust the color condition as needed
#     sc = ax.scatter(point_data.lon_rho.item(), point_data.lat_rho.item(), c=[point_color], cmap=plt.matplotlib.colors.ListedColormap([point_color]),
#                     transform=ccrs.PlateCarree(), s=50, edgecolor='black', zorder=3, label='Selected Cell')

#     # Map features
#     ax.coastlines(color='black', linewidth=1.5, zorder=1)
#     ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
#     ax.set_facecolor('lightgrey')

#     # Legend
#     threshold = title.split(" ")[2]

#     # Create a binary legend for the current threshold
#     legend_elements = [
#         Line2D([0], [0], marker='s', color='w', label=f'T°C < {threshold}',
#                markerfacecolor='lightgray', markersize=10),
#         Line2D([0], [0], marker='s', color='w', label=f'T°C ≥ {threshold}',
#                markerfacecolor=col, markersize=10)
#     ]
#     ax.legend(handles=legend_elements, loc='center', fontsize=14,
#               borderpad=0.8, frameon=True, bbox_to_anchor=(0.5, -0.15))
    
#     # Atlantic-Pacific boundary (near Drake Passage)
#     ax.plot([-70, -70], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Atlantic sector

#     # Pacific-Indian boundary
#     ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Pacific sector

#     # Indian-Atlantic boundary
#     ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Indian sector

# # Title
# # plt.suptitle(f"Relative threshold above absolute ($d_{{{day_to_plot}}}$) \n- not observed (theory) -", fontsize=16)
# plt.suptitle(f"SST above absolute thresholds ($y_{{{1980 + year_to_plot}}}, d_{{{day_to_plot}}}$)", fontsize=20, y=0.7)

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()


# # %%

# %%

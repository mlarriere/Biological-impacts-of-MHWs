"""
Created on Tue 05 Aug 16:04:36 2025

Krill evolution in subareas

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
from matplotlib.patches import Patch

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
roms_bathymetry = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc').h
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


# %% ====================== Load data ======================
def subset_spatial_domain(ds, lat_range=(-80, -60), lon_range=(270, 360)): #, (0, 30)
    lat_min, lat_max = lat_range
    lon_range1, lon_range2 = lon_range

    lat_mask = (ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max)
    lon_mask = ((ds['lon_rho'] >= lon_range1) & (ds['lon_rho'] <= lon_range2)) #| ((ds['lon_rho'] >= lon_range2[0]) & (ds['lon_rho'] <= lon_range2[1]))

    return ds.where(lat_mask & lon_mask, drop=True)

# --- Drivers
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (39, 181, 231, 1442)
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 

# --- MHW events
mhw_duration_seasonal = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))) #shape (39, 181, 231, 1442)


roms_bathymetry_box= subset_spatial_domain(roms_bathymetry, lat_range=(-90, -60), lon_range=(250, 360))


# %% Finding extnet
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker

# Central point of Box 3
lat_center = -61.0
lon_center = 286.0

# Updated expansion list with smaller and mid
expansions = [0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5]
selected_years = [1989, 2000, 2016]
full_palette = ['red', 'purple', 'orange', 'green', 'blue', 'brown', 'cyan']  # extended to match expansions

for year in selected_years:
    box_sizes = []
    temp_dists = []
    box_outlines = []
    labels = []
    all_temp_boxes = []
    valid_colors = []

    for i, expand in enumerate(expansions):
        lat_min = lat_center - expand
        lat_max = lat_center + expand
        lon_min = lon_center - expand
        lon_max = lon_center + expand

        if lat_max > -60:
            continue  # keep everything south of 60°S

        temp_box = subset_spatial_domain(
            temp_avg_100m_SO_allyrs.sel(years=[year]),
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max)
        )

        if temp_box.sizes['eta_rho'] == 0 or temp_box.sizes['xi_rho'] == 0:
            continue

        temp_data = temp_box['avg_temp'].isel(years=0)  # [days, eta_rho, xi_rho]
        flat_temp = temp_data.values.flatten()
        flat_temp = flat_temp[~np.isnan(flat_temp)]

        if len(flat_temp) == 0:
            continue

        lat_size = lat_max - lat_min
        lon_size = lon_max - lon_min
        box_area = lat_size * lon_size

        box_sizes.append(box_area)
        temp_dists.append(flat_temp)
        box_outlines.append(((lon_min, lon_max), (lat_min, lat_max)))
        labels.append(f"{lat_size:.2f}° x {lon_size:.2f}°")
        all_temp_boxes.append(temp_data)
        valid_colors.append(full_palette[i])  # match color to valid box

    if not all_temp_boxes:
        continue  # skip if no valid boxes

    # --- Determine dynamic map extent ---
    all_lons = []
    all_lats = []
    for (lon_bounds, lat_bounds) in box_outlines:
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds
        all_lons.extend([lon_min, lon_max])
        all_lats.extend([lat_min, lat_max])

    # More horizontal (longitude) padding, less vertical (latitude)
    lon_min_map = min(all_lons) - 3.0  # wider longitude padding
    lon_max_map = max(all_lons) + 3.0
    lat_min_map = max(min(all_lats) - 0.2, -63)
    lat_max_map = min(max(all_lats) + 0.2, -60)

    # === Map field (from largest valid box) ===
    temp_data_largest = all_temp_boxes[-1]  # last one is largest extent
    temp_mean_field = temp_data_largest.mean(dim='days')
    temp_std_field = temp_data_largest.std(dim='days')

    # === Plotting ===
    fig = plt.figure(figsize=(13, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 2], wspace=0.3)

    # --- Plot 1: KDE temperature distribution ---
    ax1 = fig.add_subplot(gs[0])
    for i, temp_vals in enumerate(temp_dists):
        sns.kdeplot(temp_vals, label=labels[i], ax=ax1, color=valid_colors[i], linewidth=2)

    ax1.set_title(f"Temperature Distribution ({year})")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Density")
    ax1.legend(title="Box Extent", fontsize=8, title_fontsize=9)
    ax1.grid(True)

    # --- Plot 2: Polar map ---
    ax2 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
    ax2.set_extent([lon_min_map, lon_max_map, lat_min_map, lat_max_map], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
    ax2.coastlines(color='black', linewidth=0.4, zorder=3)
    ax2.set_facecolor('#DEE2E6')

    # Add gridlines with labels
    gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = ccrs.cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ccrs.cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlocator = mticker.MultipleLocator(1)
    gl.ylocator = mticker.MultipleLocator(0.5)

    # Mean temp field as base (with white at 0°C)
    im = ax2.pcolormesh(temp_data_largest.lon_rho, temp_data_largest.lat_rho, temp_mean_field,
                        transform=ccrs.PlateCarree(), cmap='coolwarm', shading='auto',
                        zorder=1)

    # Add STD as contours
    std_levels = np.linspace(np.nanmin(temp_std_field), np.nanmax(temp_std_field), 6)
    cs = ax2.contour(temp_data_largest.lon_rho, temp_data_largest.lat_rho, temp_std_field,
                     levels=std_levels, colors='black', linewidths=0.6,
                     transform=ccrs.PlateCarree(), zorder=4)
    ax2.clabel(cs, fmt='%.2f', fontsize=6)

    # Latitude circle at -60°S
    lons = np.linspace(-180, 180, 1000)
    lats = np.full_like(lons, -60)
    ax2.plot(lons, lats, transform=ccrs.PlateCarree(),
             color='black', linestyle='--', linewidth=0.5, zorder=10)

    # Draw all valid box outlines
    for i, ((lon_bounds, lat_bounds), label) in enumerate(zip(box_outlines, labels)):
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds
        box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
        box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
        ax2.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
                 linewidth=1.2, linestyle='-', color=valid_colors[i], alpha=0.8, zorder=5)

    cbar = plt.colorbar(im, ax=ax2, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label("Mean Temperature (°C)")

    ax2.set_title("Box Extents + Mean and STD")
    plt.tight_layout()
    plt.show()



# %% ====================== Candidating Areas ======================
from scipy.ndimage import label
def count_events_longer_than_30days(da):
    """Count how many MHW events (consecutive 1s) last >30 days."""
    events = []
    for y in range(da.sizes['years']):
        binary_series = da.isel(years=y).mean(dim=['eta_rho', 'xi_rho']).values  # shape (181,)
        labeled_array, num = label(binary_series)
        long_events = [np.sum(labeled_array == i) for i in range(1, num + 1) if np.sum(labeled_array == i) > 30]
        events.append(len(long_events))
    return sum(events)

lat_candidates = np.arange(-66, -60, 1.0)
lon_candidates = np.arange(270, 290, 1.0)
threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
selected_years = [1989, 2000, 2016]
cv_threshold = 0.5

results = []

for lat_min in lat_candidates:
    lat_max = min(lat_min + 3, -60)
    for lon_min in lon_candidates:
        lon_max = lon_min + 4

        temp_box = subset_spatial_domain(temp_avg_100m_SO_allyrs.sel(years=selected_years),
                                        lat_range=(lat_min, lat_max),
                                        lon_range=(lon_min, lon_max))

        cv_all_good = True
        temp_stats_str = ""
        for year_idx in range(len(selected_years)):
            temp_year = temp_box.isel(years=year_idx)
            temp_data = temp_year['avg_temp']

            mean_val = temp_data.mean(dim=['days', 'eta_rho', 'xi_rho'])
            std_val = temp_data.std(dim=['days', 'eta_rho', 'xi_rho'])

            mean_val_item = float(mean_val.values)
            std_val_item = float(std_val.values)

            cv = std_val_item / abs(mean_val_item) if mean_val_item != 0 else np.inf

            if cv > cv_threshold:
                cv_all_good = False
                break

            temp_stats_str += (f"Year {selected_years[year_idx]}: Mean={mean_val_item:.3f}, "
                              f"Std={std_val_item:.3f}, CV={cv:.3f} | ")

        if not cv_all_good:
            continue

        # Print boundaries + temp stats only if CV condition passed
        print(f"Box lat: ({lat_min}, {lat_max}), lon: ({lon_min}, {lon_max}) - {temp_stats_str}")

        # Now check MHW events presence in the box for thresholds
        mhw_box = subset_spatial_domain(mhw_duration_seasonal,
                                       lat_range=(lat_min, lat_max),
                                       lon_range=(lon_min, lon_max))
        if mhw_box.sizes['eta_rho'] == 0 or mhw_box.sizes['xi_rho'] == 0:
            continue

        thresholds_with_events = []
        for var in threshold_vars:
            n_events = count_events_longer_than_30days(mhw_box[var])
            if n_events > 0:
                thresholds_with_events.append(var)

        if thresholds_with_events:
            results.append({
                'lat_range': (lat_min, lat_max),
                'lon_range': (lon_min, lon_max),
                'thresholds': thresholds_with_events
            })

# %% ================================================================================================================================================================================
#                                                                                           Northern Extent 
#    ================================================================================================================================================================================
candidate_boxes = {
    'box0': {'lat_range': (-64.0, -61.0), 'lon_range': (272.0, 276.0)}, 
    'box1': {'lat_range': (-64.0, -61.0), 'lon_range': (273.0, 277.0)},
    'box2': {'lat_range': (-64.0, -61.0), 'lon_range': (274.0, 278.0)},
    'box3': {'lat_range': (-61.0, -60), 'lon_range': (271.0, 275.0)}, 
}

selected_box_key = 'box3'  # box0 box1 box2 box3

lat_range = candidate_boxes[selected_box_key]['lat_range']
lon_range = candidate_boxes[selected_box_key]['lon_range']

# ==== Drivers
temp_north = subset_spatial_domain(temp_avg_100m_SO_allyrs, lat_range=lat_range, lon_range=lon_range)
chla_north = subset_spatial_domain(chla_surf_SO_allyrs, lat_range=lat_range, lon_range=lon_range)

# ==== MHWs
mhw_north_box = subset_spatial_domain(mhw_duration_seasonal, lat_range=lat_range, lon_range=lon_range)

# %% ==================== Mean Temperature =====================
selected_years = [1989, 2000, 2016]
selected_years_idx = np.array(selected_years) - 1980  # [9, 20, 36]
year_idx = selected_years_idx[2]

# --- North extent
temp_north_avg = temp_north.isel(years=year_idx).mean(dim=['eta_rho', 'xi_rho']) #shape (181,)
temp_north_std = temp_north.isel(years=year_idx).std(dim=['eta_rho', 'xi_rho'])

mean_val = temp_north_avg.avg_temp.mean().item()  # scalar mean value
std_val = temp_north_std.avg_temp.mean().item()  # scalar std value

# %% ================ Plot MHWs ================
# Threshold info
threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['$\geq$ 1°C and 90th perc', '$\geq$ 2°C and 90th perc', '$\geq$ 3°C and 90th perc', '$\geq$ 4°C and 90th perc']

# Layout config
plot = 'report'
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width/2
else:
    fig_width = 16
    fig_height = 8

# Selected days and labels
selected_days = [0, 30, 61, 92, 120, 180]
day_labels = ['Nov 1st', 'Dec 1st', 'Jan 1st', 'Feb 1st', 'Mar 1st', 'Apr 30th']

# === Custom font sizes ===
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}


fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.3, wspace=0.05)

axes = []
for i in range(6):
    ax = fig.add_subplot(gs[i // 3, i % 3], projection=ccrs.SouthPolarStereo())
    axes.append(ax)

for ax, day_idx, day_label in zip(axes, selected_days, day_labels):
    data = mhw_north_box  

    # No event mask for that day averaged over years
    no_event_mask = ((data['det_1deg'].isel(years= year_idx, days=day_idx) == 0) &
                     (data['det_2deg'].isel(years= year_idx, days=day_idx) == 0) &
                     (data['det_3deg'].isel(years= year_idx, days=day_idx) == 0) &
                     (data['det_4deg'].isel(years= year_idx, days=day_idx) == 0)).fillna(True)
    
    ax.contourf(data.lon_rho, data.lat_rho, no_event_mask, levels=[0.5, 1], colors=['white'], transform=ccrs.PlateCarree(), zorder=1)
    
    for var, color in zip(threshold_vars, threshold_colors):
        event_mask = data[var].isel(years= year_idx, days=day_idx).fillna(0)
        binary_mask = (event_mask >= 0.166).astype(int)
        ax.contourf(data.lon_rho, data.lat_rho, binary_mask, levels=[0.5, 1], colors=[color], transform=ccrs.PlateCarree(), alpha=0.8, zorder=2)
    
    lw = 0.7 if plot == 'slides' else 0.4
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=3)
    ax.coastlines(color='black', linewidth=lw, zorder=4)
    ax.set_facecolor('#DEE2E6')

    ax.set_extent([268, 288, -70, -57], crs=ccrs.PlateCarree())
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=lw, color='gray', alpha=0.3, linestyle='--', zorder=9)
    gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs  

    # Sector boundaries
    for lon in [-90, 120, 0]:
            ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=0.5)

    # Draw 60°S latitude circle accurately
    lons = np.linspace(-180, 180, 1000)  # or 0 to 360 if your data is in that range
    lats = np.full_like(lons, -60)       # constant latitude at -60°S
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=lw, zorder=10)

    # --- Add bathymetry contours ---
    # bathy_lon = roms_bathymetry['lon_rho']
    # bathy_lat = roms_bathymetry['lat_rho']
    # bathy = roms_bathymetry.where((bathy_lon >= 260) & (bathy_lon <= 300) & (bathy_lat >= -70) & (bathy_lat <= -60))
    # contour_levels = [200, 2000]
    # cs = ax.contour(
    #     bathy_lon, bathy_lat, bathy,
    #     levels=contour_levels,
    #     colors='black',
    #     linewidths=0.6,
    #     linestyles='--',
    #     transform=ccrs.PlateCarree(),
    #     zorder=2
    # )

    # # Label the contours with depth values
    # ax.clabel(cs, fmt='%d m', inline=True, fontsize=5, colors='black')

    ax.set_title(day_label, **title_kwargs)

# Legend
from matplotlib.patches import Patch

# --- Legend handles ---
legend_handles = [Patch(facecolor='white', edgecolor='black', label='No MHW event', linewidth=0.5)]
legend_handles += [
    Patch(facecolor=c, edgecolor='black', label=l, linewidth=0.5)
    for c, l in zip(threshold_colors, threshold_labels)
]

legend_kwargs = {
    "frameon": True,
    "ncol": 3,  
    "loc": "lower center",
    "bbox_to_anchor": (0.52, -0.1)
}

fig.legend(handles=legend_handles, **legend_kwargs)

# Title
if plot=='slides':
    fig.suptitle(f"MHW Detection in {year_idx+1980} - {year_idx+1980+1}", **suptitle_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"mhws_subarea_North_{year_idx+1980}_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %% ================== Identify events ==================
from scipy.ndimage import label
data = mhw_north_box  

threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_labels = ['$\\geq$ 90th perc and 1°C', '$\\geq$ 90th perc and 2°C', '$\\geq$ 90th perc and 3°C', '$\\geq$ 90th perc and 4°C']
threshold_events = {}

for var, label_name in zip(threshold_vars, threshold_labels):
    # Binary 3D array: (days, eta_rho, xi_rho)
    det_filtered = data[var].isel(years=year_idx).fillna(0).astype(bool)

    # Spatial fraction per day
    total_valid = det_filtered.notnull().sum(dim=['eta_rho', 'xi_rho'])
    daily_detected = det_filtered.sum(dim=['eta_rho', 'xi_rho'])
    spatial_fraction = daily_detected / total_valid

    # Daily series of True where >5% of region is active
    daily_series = (spatial_fraction > 0.05).values  # shape (181,)
    
    # daily_series = det_filtered.any(dim=['eta_rho', 'xi_rho'])

    # Label connected time events
    labeled_array, num_events = label(daily_series)

    # Filter only events with duration > 30 days
    event_lengths = [np.sum(labeled_array == i) for i in range(1, num_events + 1)]
    long_events = [l for l in event_lengths if l > 30]

    # Save
    threshold_events[label_name] = {
        'n_events': len(long_events),
        'lengths': long_events,
        'days': labeled_array  # useful for plotting
    }

# %% ================ LENGTH ================
from Growth_Model.growth_model import length_Atkison2006  
# == Climatological Drivers -> Mean Chla and T°C  (days, eta, xi)
temp_clim_atl = temp_north.isel(years=slice(0,30)) #shape: (30, 181, 231, 360)
temp_clim_atl_mean = temp_clim_atl.mean(dim=['years']) #shape: (181, 231, 360)
chla_clim_atl = chla_north.isel(years=slice(0,30))
chla_clim_atl_mean = chla_clim_atl.mean(dim=['years'])

# == Climatological Length
climatological_length_north = length_Atkison2006(chla=chla_clim_atl_mean.chla, temp=temp_clim_atl_mean.avg_temp, initial_length=35, intermoult_period=10)
climatological_length_north_mean = climatological_length_north.mean(dim=['eta_rho', 'xi_rho'])

# == Length for north area
length_north = length_Atkison2006(chla=chla_north.chla.isel(years=year_idx), 
                                   temp=temp_north.avg_temp.isel(years=year_idx), 
                                   initial_length=35, intermoult_period=10)
length_north_mean = length_north.mean(dim=['eta_rho', 'xi_rho'])

# %% ========== Plot area with final lentgh ==========
from matplotlib.colors import LinearSegmentedColormap
# === Layout config ===
plot = 'report'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611 / 2  # half-column width in inches
    fig_height = fig_width/1.5
else:
    fig_width = 8
    fig_height = 6

# === Custom font sizes ===
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}

# === Color map ===
colors = ["#511c5b", "#762a83", "#c2a5cf", "#f7f7f7", "#fdae61", "#d73027", "#7d1c17"]
cmap_len = LinearSegmentedColormap.from_list("length", colors, N=256)
plot_kwargs = dict(cmap=cmap_len, vmin=33, vmax=37, rasterized=True)
# plot_kwargs = dict(cmap=cmap_len, vmin=-1, vmax=5, rasterized=True)

# === Figure and GridSpec for plot + colorbar ===
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[20, 1], wspace=0.05)

# === Main map ===
ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
theta = np.linspace(np.pi / 2, np.pi, 100)
center, radius = [0.5, 0.51], 0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

# Plot data
data = length_north.isel(days=-1)
# data= temp_north.avg_temp.isel(years=year_idx).mean(dim='days')
im = ax.pcolormesh(data.lon_rho, data.lat_rho, data, transform=ccrs.PlateCarree(), **plot_kwargs, zorder=1)

# Coastlines and land
# Map extent and features
ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
lw = 0.7 if plot == 'slides' else 0.4
ax.coastlines(color='black', linewidth=lw, zorder=3)
ax.set_facecolor('#DEE2E6')

# Zoom region
# ax.set_extent([270, 290, -70, -57], crs=ccrs.PlateCarree())
ax.set_extent([268, 288, -70, -57], crs=ccrs.PlateCarree())
    
# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=lw, color='gray', alpha=0.3, linestyle='--', zorder=4)
# gl.top_labels = False
# gl.right_labels = False
gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs

# Draw 60°S latitude circle accurately
lons = np.linspace(-180, 180, 1000)  # or 0 to 360 if your data is in that range
lats = np.full_like(lons, -60)       # constant latitude at -60°S
# Sector boundaries
for lon in [-90, 120, 0]:
        ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=0.5)

ax.plot(lons, lats, transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=lw, zorder=10)

# === Colorbar ===
cax = fig.add_subplot(gs[1])
cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend='both')
cbar.set_ticks([33, 35, 37])               # Set exact tick locations
cbar.set_ticklabels(['33', '35', '37'])    # Set custom labels (optional)
cbar.set_label("Length [mm]", ** label_kwargs)
cbar.ax.tick_params(**tick_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"maplength_subarea_North_{year_idx+1980}_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %% ================= Times Series =================
from datetime import datetime, timedelta

# --- Setup days axis ---
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)  # Start from Nov 1
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)

tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15)
tick_labels = [date_dict.get(day, '') for day in tick_positions]

# --- Colors for threshold fills ---
threshold_colors = {
    '$\geq$ 90th perc and 1°C': '#5A7854',
    '$\geq$ 90th perc and 2°C': '#8780C6',
    '$\geq$ 90th perc and 3°C': '#E07800',
    '$\geq$ 90th perc and 4°C': '#9B2808'
}

# === Layout config ===
plot = 'report'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611 / 1.5  # half-column width in inches
    fig_height = fig_width
else:
    fig_width = 8
    fig_height = 6

# === Custom font sizes ===
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}
lw= 0.5 if 'report' else 2

# --- Plot setup ---
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(fig_width, fig_height),
    sharex=True,
    gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.15}
)

# --- Temperature Plot ---
mean = temp_north_avg.avg_temp
std= temp_north_std.avg_temp
line = ax1.plot(days_xaxis, mean, color='black', linewidth=lw, label='Mean')[0]
std_fill = ax1.fill_between(days_xaxis, mean - std, mean + std,
                            color='black', alpha=0.3, label='±1 Std Dev')

used_labels = set()
for label, info in threshold_events.items():
    color = threshold_colors.get(label, 'grey')
    active_days = info['days']
    if len(active_days) == 0:
        continue
    for event_id in np.unique(active_days[active_days > 0]):
        idx = np.where(active_days == event_id)[0]
        if len(idx) == 0:
            continue
        start_day = days_xaxis[idx[0]]
        end_day = days_xaxis[idx[-1]] + 1
        if label not in used_labels:
            ax1.axvspan(start_day, end_day, color=color, alpha=0.6, label=label)
            used_labels.add(label)
        else:
            ax1.axvspan(start_day, end_day, color=color, alpha=0.6)

ax1.set_ylabel("Temperature [°C]", **label_kwargs)

# Legend with only mean and std
leg=ax1.legend(handles=[line, Patch(facecolor='black', alpha=0.3, label='±1 Std Dev')], 
           loc='upper right', frameon=True, bbox_to_anchor =(0.3, 0.99), handlelength=0.7, fontsize=9)
leg.get_frame().set_linewidth(0.5)  # Thinner frame, default is usually 1.0 or more

ymin = np.floor((mean - std).min().item() * 2) / 2
ymax = np.ceil((mean + std).max().item() * 2) / 2
yticks = np.arange(ymin, ymax + 0.1, 1)
ax1.set_yticks(yticks)


# --- Krill Length Plot ---
line_1 = ax2.plot(days_xaxis, length_north_mean, color='black', linewidth=lw, label='Krill Length')[0]
line_2 = ax2.plot(days_xaxis, climatological_length_north_mean, color='#263E69', linewidth=lw, linestyle='--', label='Climatology')[0]

for label, info in threshold_events.items():
    color = threshold_colors.get(label, 'grey')
    active_days = info['days']
    if len(active_days) == 0:
        continue
    for event_id in np.unique(active_days[active_days > 0]):
        idx = np.where(active_days == event_id)[0]
        if len(idx) == 0:
            continue
        start_day = days_xaxis[idx[0]]
        end_day = days_xaxis[idx[-1]] + 1
        ax2.axvspan(start_day, end_day, color=color, alpha=0.6)

ax2.set_ylabel("Length [mm]", **label_kwargs)
# ax2.grid(True, linestyle='--', alpha=0.4)
leg=ax2.legend([line_2], ['Climatology'], loc='upper left', frameon=True, bbox_to_anchor =(0.01, 0.49), handlelength=0.7, fontsize=9)
leg.get_frame().set_linewidth(0.5)  # Thinner frame, default is usually 1.0 or more

ymin = np.floor(length_north_mean.min().item() * 4) / 4
ymax = np.ceil(length_north_mean.max().item() * 4) / 4
yticks = np.arange(ymin, ymax + 0.01, 0.5)
ax2.set_yticks(yticks)



# --- MHW Area Coverage Plot ---
data = mhw_north_box
eta_dim = data.dims['eta_rho']
xi_dim = data.dims['xi_rho']
total_area = eta_dim * xi_dim

for var, label in zip(['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], threshold_colors.keys()):
    detection = data[var].isel(years=year_idx).fillna(0).astype(bool)  # [days, y, x]
    active_days_mask = threshold_events[label]['days']  # [days,] array where 1,2,... = valid long events
    
    # Make 3D mask where only valid event days are retained
    mask_valid = np.isin(active_days_mask, np.unique(active_days_mask[active_days_mask > 0]))  # shape: [days]
    detection_filtered = detection.copy()
    detection_filtered[~mask_valid, :, :] = False

    # Calculate area per day
    daily_area_covered = detection_filtered.sum(dim=['eta_rho', 'xi_rho']).values
    daily_area_percent = 100 * daily_area_covered / total_area

    ax3.plot(days_xaxis, daily_area_percent, color=threshold_colors[label], linewidth=lw)


# for var, label in zip(['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], threshold_colors.keys()):
#     mask = data[var].isel(years=year_idx)
#     daily_area_covered = mask.fillna(0).astype(bool).sum(dim=['eta_rho', 'xi_rho']).values
#     daily_area_percent = 100 * daily_area_covered / total_area
#     ax3.plot(days_xaxis, daily_area_percent, color=threshold_colors[label], linewidth=lw)

ax3.set_ylabel("Area [$\%$]", **label_kwargs)
ax3.set_xlabel("Date", **label_kwargs)
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha='right', **tick_kwargs)
# ax3.grid(True, linestyle='--', alpha=0.4)

# --- Bottom legend: only threshold colors ---
bottom_handles = [
    Patch(facecolor=color, edgecolor='black', lw=0.5)
    for label, color in threshold_colors.items()
]
bottom_labels = list(threshold_colors.keys())

fig.legend(
    bottom_handles, bottom_labels,
    loc='lower center',
    ncol=2,
    frameon=True,
    bbox_to_anchor=(0.5, -0.15),
    handlelength=1.5,
    fontsize=9
)
if plot=='slides':
    fig.suptitle(f"Growth Season {year_idx+1980}-{year_idx+1980+1}", **suptitle_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"timeseries_subarea_North_{year_idx+1980}_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()


# %% ================================================================================================================================================================================
#                                                                                           Peninsula Extent 
#    ================================================================================================================================================================================
from shapely.geometry import Polygon, Point

def tilted_rectangle_mask(ds, point1, point2, point3, point4):
    polygon = Polygon([point1, point2, point3, point4, point1])

    mask = np.zeros(ds.lat_rho.shape, dtype=bool)
    for i in range(ds.dims['eta_rho']):
        for j in range(ds.dims['xi_rho']):
            lon = ds.lon_rho.values[i, j]
            lat = ds.lat_rho.values[i, j]
            if polygon.contains(Point(lon, lat)):
                mask[i, j] = True

    mask_da = xr.DataArray(mask, dims=['eta_rho', 'xi_rho'],
                           coords={'eta_rho': ds.eta_rho, 'xi_rho': ds.xi_rho})
    return ds.where(mask_da, drop=True)

# -- Peninsula big extent
# Subset broad rectangle around peninsula
temp_peninsula_box = subset_spatial_domain(temp_avg_100m_SO_allyrs, lat_range=(-90, -60), lon_range=(270, 360))
chla_peninsula_box = subset_spatial_domain(chla_surf_SO_allyrs, lat_range=(-90, -60), lon_range=(270, 360))
mhw_peninsula_box = subset_spatial_domain(mhw_duration_seasonal, lat_range=(-90, -60), lon_range=(270, 360))


# -- Peninsula smaller extent
temp_peninsula_tilted = tilted_rectangle_mask(
    ds=temp_peninsula_box,
    point1=(360-65, -64),   # SW corner (more west/south)
    point2=(360-64, -65),   # SE corner
    point3=(360-62, -64),   # NE corner (more east/north)
    point4=(360-63, -63)    # NW corner
)

chla_peninsula_tilted = tilted_rectangle_mask(
    ds=chla_peninsula_box,
    point1=(360-65, -64),   # SW corner (more west/south)
    point2=(360-64, -65),   # SE corner
    point3=(360-62, -64),   # NE corner (more east/north)
    point4=(360-63, -63)    # NW corner
)

mhw_peninsula_tilted = tilted_rectangle_mask(
    ds=mhw_peninsula_box,
    point1=(360-65, -64),   # SW corner (more west/south)
    point2=(360-64, -65),   # SE corner
    point3=(360-62, -64),   # NE corner (more east/north)
    point4=(360-63, -63)    # NW corner
)


# %% ==================== Mean Temperature =====================
selected_years = [1989, 2000, 2016]
selected_years_idx = np.array(selected_years) - 1980  # [9, 20, 36]
year_idx = selected_years_idx[2]

# --- Peninsula extent
temp_peninsula_avg = temp_peninsula_tilted.isel(years=year_idx).mean(dim=['eta_rho', 'xi_rho']) #shape (181,)
temp_peninsula_std = temp_peninsula_tilted.isel(years=year_idx).std(dim=['eta_rho', 'xi_rho']) 

mean_val = temp_peninsula_avg.avg_temp.mean().item()  # scalar mean value
std_val = temp_peninsula_std.avg_temp.mean().item()  # scalar std value


# %% ================ Plot MHWs ================
# Threshold info
threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['$\geq$ 1°C and 90th perc', '$\geq$ 2°C and 90th perc', '$\geq$ 3°C and 90th perc', '$\geq$ 4°C and 90th perc']

# Layout config
plot = 'report'
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width/2
else:
    fig_width = 16
    fig_height = 8

# Selected days and labels
selected_days = [0, 30, 61, 92, 120, 180]
day_labels = ['Nov 1st', 'Dec 1st', 'Jan 1st', 'Feb 1st', 'Mar 1st', 'Apr 30th']

# === Custom font sizes ===
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}


fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.3, wspace=0.05)

axes = []
for i in range(6):
    ax = fig.add_subplot(gs[i // 3, i % 3], projection=ccrs.SouthPolarStereo())
    axes.append(ax)

for ax, day_idx, day_label in zip(axes, selected_days, day_labels):
    data = mhw_peninsula_tilted

    # No event mask for that day averaged over years
    no_event_mask = ((data['det_1deg'].isel(years= year_idx, days=day_idx) == 0) &
                     (data['det_2deg'].isel(years= year_idx, days=day_idx) == 0) &
                     (data['det_3deg'].isel(years= year_idx, days=day_idx) == 0) &
                     (data['det_4deg'].isel(years= year_idx, days=day_idx) == 0)).fillna(True)
    
    ax.contourf(data.lon_rho, data.lat_rho, no_event_mask, levels=[0.5, 1], colors=['white'], transform=ccrs.PlateCarree(), zorder=1)
    
    for var, color in zip(threshold_vars, threshold_colors):
        event_mask = data[var].isel(years= year_idx, days=day_idx).fillna(0)
        binary_mask = (event_mask >= 0.166).astype(int)
        ax.contourf(data.lon_rho, data.lat_rho, binary_mask, levels=[0.5, 1], colors=[color], transform=ccrs.PlateCarree(), alpha=0.8, zorder=2)
    
    lw = 0.7 if plot == 'slides' else 0.4
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=3)
    ax.coastlines(color='black', linewidth=lw, zorder=4)
    ax.set_facecolor('#DEE2E6')

    ax.set_extent([285, 305, -70, -60], crs=ccrs.PlateCarree())
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=lw, color='gray', alpha=0.3, linestyle='--', zorder=9)
    gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs  

    # Draw 60°S latitude circle accurately
    lons = np.linspace(-180, 180, 1000)  # or 0 to 360 if your data is in that range
    lats = np.full_like(lons, -60)       # constant latitude at -60°S
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=lw, zorder=10)

    ax.set_title(day_label, **title_kwargs)

    # --- Add bathymetry contours ---
    bathy_lon = roms_bathymetry_box['lon_rho']
    bathy_lat = roms_bathymetry_box['lat_rho']
    bathy = roms_bathymetry_box.where((bathy_lon >= 285) & (bathy_lon <= 305) & (bathy_lat >= -70) & (bathy_lat <= -60))

    # Choose contour levels (e.g., shelf and slope breaks)
    contour_levels = [200, 2000]

    # Plot contour lines
    cs = ax.contour(
        bathy_lon, bathy_lat, bathy,
        levels=contour_levels,
        colors='black',
        linewidths=0.6,
        linestyles='--',
        transform=ccrs.PlateCarree(),
        zorder=2
    )

    # Label the contours with depth values
    ax.clabel(cs, fmt='%d m', inline=True, fontsize=5, colors='black')


# Legend
from matplotlib.patches import Patch

# --- Legend handles ---
legend_handles = [Patch(facecolor='white', edgecolor='black', label='No MHW event', linewidth=0.5)]
legend_handles += [
    Patch(facecolor=c, edgecolor='black', label=l, linewidth=0.5)
    for c, l in zip(threshold_colors, threshold_labels)
]

legend_kwargs = {
    "frameon": True,
    "ncol": 3,  
    "loc": "lower center",
    "bbox_to_anchor": (0.52, -0.1)
}

fig.legend(handles=legend_handles, **legend_kwargs)

# Title
if plot=='slides':
    fig.suptitle(f"MHW Detection in {year_idx+1980} - {year_idx+1980+1}", **suptitle_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"mhws_subarea_Peninsula_{year_idx+1980}_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %% ================== Identify events ==================
from scipy.ndimage import label
data = mhw_peninsula_tilted  

threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_labels = ['$\\geq$ 90th perc and 1°C', '$\\geq$ 90th perc and 2°C', '$\\geq$ 90th perc and 3°C', '$\\geq$ 90th perc and 4°C']
threshold_events = {}

for var, label_name in zip(threshold_vars, threshold_labels):
    # Binary 3D array: (days, eta_rho, xi_rho)
    det_filtered = data[var].isel(years=year_idx).fillna(0).astype(bool)

    # # Spatial fraction per day
    # total_valid = det_filtered.notnull().sum(dim=['eta_rho', 'xi_rho'])
    # daily_detected = det_filtered.sum(dim=['eta_rho', 'xi_rho'])
    # spatial_fraction = daily_detected / total_valid

    # # Daily series of True where >5% of region is active
    # daily_series = (spatial_fraction > 0.05).values  # shape (181,)
    
    daily_series = det_filtered.any(dim=['eta_rho', 'xi_rho'])

    # Label connected time events
    labeled_array, num_events = label(daily_series)

    # Filter only events with duration > 30 days
    event_lengths = [np.sum(labeled_array == i) for i in range(1, num_events + 1)]
    long_events = [l for l in event_lengths if l > 30]

    # Save
    threshold_events[label_name] = {
        'n_events': len(long_events),
        'lengths': long_events,
        'days': labeled_array  # useful for plotting
    }



# %% ================ LENGTH ================
from Growth_Model.growth_model import length_Atkison2006  

# == Climatological Drivers
temp_clim_peninsula = temp_peninsula_tilted.isel(years=slice(0,30)) #shape: (30, 181, 231, 360)
temp_clim_peninsula_mean = temp_clim_peninsula.mean(dim=['years']) #shape: (181, 231, 360)
chla_clim_peninsula = chla_peninsula_tilted.isel(years=slice(0,30))
chla_clim_peninsula_mean = chla_clim_peninsula.mean(dim=['years'])

# == Climatological Length
climatological_length_peninsula = length_Atkison2006(chla=chla_clim_peninsula_mean.chla, 
                                                     temp=temp_clim_peninsula_mean.avg_temp, 
                                                     initial_length=35, intermoult_period=10)
climatological_length_peninsula_mean = climatological_length_peninsula.mean(dim=['eta_rho', 'xi_rho'])

# == Length for Peninsula area
length_peninsula = length_Atkison2006(chla=chla_peninsula_tilted.chla.isel(years=year_idx),
                                      temp=temp_peninsula_tilted.avg_temp.isel(years=year_idx),
                                      initial_length=35, intermoult_period=10)
length_peninsula_mean = length_peninsula.mean(dim=['eta_rho', 'xi_rho'])

# %% ========== Plot area with final lentgh ==========
from matplotlib.colors import LinearSegmentedColormap
# === Layout config ===
plot = 'report'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611 / 2  # half-column width in inches
    fig_height = fig_width/1.5
else:
    fig_width = 8
    fig_height = 6

# === Custom font sizes ===
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}

# === Color map ===
colors = [
    (0.00, "#f7f7f7"),  # 35 mm
    (0.25, "#F27A02"),  # 36 mm
    (0.50, "#D73027"),  # 37 mm
    (0.75, "#7D1C17"),  # 38 mm
    (1.00, "#450F0D")   # 39 mm
]
cmap_len = LinearSegmentedColormap.from_list("length_custom", colors, N=256)
plot_kwargs = dict(cmap=cmap_len, vmin=35, vmax=39, rasterized=True)

# === Figure and GridSpec for plot + colorbar ===
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[20, 1], wspace=0.05)

# === Main map ===
ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
theta = np.linspace(np.pi / 2, np.pi, 100)
center, radius = [0.5, 0.51], 0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

# Plot data
mask = ~np.isnan(mhw_peninsula_tilted.duration.isel(years=year_idx, days=-1))
length_data_masked = length_peninsula.isel(days=-1).where(mask)
im = ax.pcolormesh(length_data_masked.lon_rho, length_data_masked.lat_rho,
                   length_data_masked, transform=ccrs.PlateCarree(), **plot_kwargs, zorder=1)
# data = length_peninsula.isel(days=-1)
# im = ax.pcolormesh(data.lon_rho, data.lat_rho, data, transform=ccrs.PlateCarree(), **plot_kwargs, zorder=1)

# Map extent and features
ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
lw = 0.7 if plot == 'slides' else 0.4
ax.coastlines(color='black', linewidth=lw, zorder=3)
ax.set_facecolor('#DEE2E6')

# Zoom region
ax.set_extent([285, 305, -70, -60], crs=ccrs.PlateCarree())

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=lw, color='gray', alpha=0.3, linestyle='--', zorder=4)
gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs

# Draw 60°S latitude circle accurately
lons = np.linspace(-180, 180, 1000)  # or 0 to 360 if your data is in that range
lats = np.full_like(lons, -60)       # constant latitude at -60°S
ax.plot(lons, lats, transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=lw, zorder=10)

# --- Add bathymetry contours ---
bathy = roms_bathymetry_box.where((bathy_lon >= 285) & (bathy_lon <= 305) & (bathy_lat >= -70) & (bathy_lat <= -60))
bathy_lon = roms_bathymetry_box['lon_rho']
bathy_lat = roms_bathymetry_box['lat_rho']

# Choose contour levels (e.g., shelf and slope breaks)
contour_levels = [200, 2000]

# Plot contour lines
cs = ax.contour(
    bathy_lon, bathy_lat, bathy,
    levels=contour_levels,
    colors='black',
    linewidths=0.6,
    linestyles='--',
    transform=ccrs.PlateCarree(),
    zorder=2
)

# Label the contours with depth values
ax.clabel(cs, fmt='%d m', inline=True, fontsize=5, colors='black')

# === Colorbar ===
cax = fig.add_subplot(gs[1])
cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend='max')
cbar.set_ticks([35, 36, 37, 38, 39])
cbar.set_ticklabels(['35', '36', '37', '38', '39'])
cbar.set_label("Length [mm]", ** label_kwargs)
cbar.ax.tick_params(**tick_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"maplength_subarea_Peninsula_{year_idx+1980}_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %% ================= Times Series =================
from datetime import datetime, timedelta

# --- Setup days axis ---
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)  # Start from Nov 1
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)

tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15)
tick_labels = [date_dict.get(day, '') for day in tick_positions]

# --- Colors for threshold fills ---
threshold_colors = {
    '$\geq$ 90th perc and 1°C': '#5A7854',
    '$\geq$ 90th perc and 2°C': '#8780C6',
    '$\geq$ 90th perc and 3°C': '#E07800',
    '$\geq$ 90th perc and 4°C': '#9B2808'
}

# === Layout config ===
plot = 'report'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611 / 1.5  # half-column width in inches
    fig_height = fig_width
else:
    fig_width = 8
    fig_height = 6

# === Custom font sizes ===
title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}
lw= 0.5 if 'report' else 2

# --- Plot setup ---
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(fig_width, fig_height),
    sharex=True,
    gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.15}
)

# --- Temperature Plot ---
mean = temp_peninsula_avg.avg_temp
std= temp_peninsula_std.avg_temp
line = ax1.plot(days_xaxis, mean, color='black', linewidth=lw, label='Mean')[0]
std_fill = ax1.fill_between(days_xaxis, mean - std, mean + std,
                            color='black', alpha=0.3, label='±1 Std Dev')

used_labels = set()
for label, info in threshold_events.items():
    color = threshold_colors.get(label, 'grey')
    active_days = info['days']
    if len(active_days) == 0:
        continue
    for event_id in np.unique(active_days[active_days > 0]):
        idx = np.where(active_days == event_id)[0]
        if len(idx) == 0:
            continue
        start_day = days_xaxis[idx[0]]
        end_day = days_xaxis[idx[-1]] + 1
        if label not in used_labels:
            ax1.axvspan(start_day, end_day, color=color, alpha=0.6, label=label)
            used_labels.add(label)
        else:
            ax1.axvspan(start_day, end_day, color=color, alpha=0.6)

ax1.set_ylabel("Temperature [°C]", **label_kwargs)

# Legend with only mean and std
leg=ax1.legend(handles=[line, Patch(facecolor='black', alpha=0.3, label='±1 Std Dev')], 
           loc='upper right', frameon=True, bbox_to_anchor =(0.3, 0.99), handlelength=0.7, fontsize=9)
leg.get_frame().set_linewidth(0.5)  # Thinner frame, default is usually 1.0 or more

ymin = np.floor((mean - std).min().item() * 2) / 2
ymax = np.ceil((mean + std).max().item() * 2) / 2
yticks = np.arange(ymin, ymax + 0.1, 1)
ax1.set_yticks(yticks)


# --- Krill Length Plot ---
line_1 = ax2.plot(days_xaxis, length_peninsula_mean, color='black', linewidth=lw, label='Krill Length')[0]
line_2 = ax2.plot(days_xaxis, climatological_length_peninsula_mean, color='#263E69', linewidth=lw, linestyle='--', label='Climatology')[0]

for label, info in threshold_events.items():
    color = threshold_colors.get(label, 'grey')
    active_days = info['days']
    if len(active_days) == 0:
        continue
    for event_id in np.unique(active_days[active_days > 0]):
        idx = np.where(active_days == event_id)[0]
        if len(idx) == 0:
            continue
        start_day = days_xaxis[idx[0]]
        end_day = days_xaxis[idx[-1]] + 1
        ax2.axvspan(start_day, end_day, color=color, alpha=0.6)

ax2.set_ylabel("Length [mm]", **label_kwargs)
# ax2.grid(True, linestyle='--', alpha=0.4)
leg=ax2.legend([line_2], ['Climatology'], loc='upper left', frameon=True, bbox_to_anchor =(0.01, 0.49), handlelength=0.7, fontsize=9)
leg.get_frame().set_linewidth(0.5)  # Thinner frame, default is usually 1.0 or more

ymin = np.floor(length_peninsula_mean.min().item() * 4) / 4
ymax = np.ceil(length_peninsula_mean.max().item() * 4) / 4
yticks = np.arange(ymin, ymax + 0.01, 0.5)
ax2.set_yticks(yticks)


# --- MHW Area Coverage Plot ---
data = mhw_peninsula_tilted.where(mhw_peninsula_tilted.duration > 30)
eta_dim = data.dims['eta_rho']
xi_dim = data.dims['xi_rho']
total_area = eta_dim * xi_dim #256 cells

for var, label in zip(['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], threshold_colors.keys()):
    mask = data[var].isel(years=year_idx)
    daily_area_covered = mask.fillna(0).astype(bool).sum(dim=['eta_rho', 'xi_rho']).values
    daily_area_percent = 100 * daily_area_covered / total_area
    ax3.plot(days_xaxis, daily_area_percent, color=threshold_colors[label], linewidth=lw)

ax3.set_ylabel("Area [$\%$]", **label_kwargs)
ax3.set_xlabel("Date", **label_kwargs)
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha='right', **tick_kwargs)
# ax3.grid(True, linestyle='--', alpha=0.4)

# --- Bottom legend: only threshold colors ---
bottom_handles = [
    Patch(facecolor=color, edgecolor='black', lw=0.5)
    for label, color in threshold_colors.items()
]
bottom_labels = list(threshold_colors.keys())

fig.legend(
    bottom_handles, bottom_labels,
    loc='lower center',
    ncol=2,
    frameon=True,
    bbox_to_anchor=(0.5, -0.15),
    handlelength=1.5,
    fontsize=9
)
if plot=='slides':
    fig.suptitle(f"Growth Season {year_idx+1980}-{year_idx+1980+1}", **suptitle_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"timeseries_subarea_Peninsula_{year_idx+1980}_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()


# %% # %%  ================================================ Plot BOTH areas ================================================
# === Layout config ===
plot = 'report'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611 
    fig_height = fig_width 
else:
    fig_width = 8
    fig_height = 6

title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}
lw = 0.7 if plot == 'slides' else 0.4

# === Data slices ===
# year_idx = 36 
# day_idx = 10
# north = temp_north.avg_temp.isel(years=year_idx, days=day_idx)
# pen = temp_peninsula_tilted.avg_temp.isel(years=year_idx, days=day_idx)
north = temp_clim_atl_mean.avg_temp.mean(dim='days')
pen = temp_clim_peninsula_mean.avg_temp.mean(dim='days')

# === Setup Figure and Axes ===
fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

# === Plot Data ===
from matplotlib.colors import LinearSegmentedColormap

colors1 = [
    (0.00, "#D73027"),  # 35 mm
    (0.25, "#D73027"),  # 36 mm
    (0.50, "#D73027"),  # 37 mm
    (0.75, "#D73027"),  # 38 mm
    (1.00, "#D73027")   # 39 mm
]

colors2 = [
    (0.00, "#7B2CBF"),  # 35 mm
    (0.25, "#7B2CBF"),  # 36 mm
    (0.50, "#7B2CBF"),  # 37 mm
    (0.75, "#7B2CBF"),  # 38 mm
    (1.00, "#7B2CBF")   # 39 mm
]
cmap_len1 = LinearSegmentedColormap.from_list("length_custom", colors1, N=256)
cmap_len2 = LinearSegmentedColormap.from_list("length_custom", colors2, N=256)
plot_kwargs1 = dict(cmap=cmap_len1, vmin=0, vmax=2, rasterized=True)
plot_kwargs2 = dict(cmap=cmap_len2, vmin=0, vmax=2, rasterized=True)

pcm1 = ax.pcolormesh(
    north.lon_rho, north.lat_rho, north,
    transform=ccrs.PlateCarree(),
    cmap=cmap_len1, zorder=1
)
pcm2 = ax.pcolormesh(
    pen.lon_rho, pen.lat_rho, pen,
    transform=ccrs.PlateCarree(),
    cmap=cmap_len2, zorder=1
)

# === Features ===
ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
ax.coastlines(resolution='10m', color='black', linewidth=lw, zorder=3)
ax.set_facecolor('#DEE2E6')

# === Extent and Grid ===
ax.set_extent([265, 0, -90, -55], crs=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True, linewidth=lw, color='gray', alpha=0.3, linestyle='--', zorder=4)
gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs
# Sector boundaries
for lon in [-90, 120, 0]:
        ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=lw)

# === Circle for 60°S Latitude ===
lons = np.linspace(-180, 180, 1000)
lats = np.full_like(lons, -60)
ax.plot(lons, lats, transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=lw, zorder=10)

# === Optional: Clipping Circle for Circular Look ===
theta = np.linspace(np.pi / 2, np.pi+np.pi/10, 100)
center, radius = [0.5, 0.5], 0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)
ax.set_boundary(circle, transform=ax.transAxes)

# === Colorbar ===
cbar = fig.colorbar(pcm2, ax=ax, orientation='vertical', extend='both', pad=0.05, shrink=0.8)
cbar.set_label('Temperature [°C]', **label_kwargs)

# === Title or Label ===
if plot == 'slides':
    ax.set_title(f'Temperature at 100m – Day {day_idx} of Year {1980 + year_idx}', **title_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"both_areas_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %% # %%  ================================================ Plot Mean Temperature ================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# === Layout config ===
plot = 'report'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611  # full-column width in inches
    fig_height = fig_width / 2
else:
    fig_width = 10
    fig_height = 5

# === Font sizes ===
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}

# === Figure and GridSpec ===
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(nrows=1, ncols=4, width_ratios=[20, 1, 20, 1], wspace=0.3)

# === Normalization shared for both plots ===
norm = Normalize(vmin=0, vmax=4)

# === Subplot 1: Atlantic Sector ===
ax1 = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
ax1.set_extent([268, 288, -70, -57], crs=ccrs.PlateCarree())

temp1 = temp_clim_atl_mean.avg_temp.isel(days=-1)
lon1 = temp_clim_atl_mean.lon_rho
lat1 = temp_clim_atl_mean.lat_rho

im1 = ax1.pcolormesh(
    lon1, lat1, temp1,
    cmap='Reds',
    norm=norm,
    transform=ccrs.PlateCarree(),
    rasterized=True,
    zorder=1
)

ax1.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
ax1.coastlines(color='black', linewidth=0.4, zorder=3)
ax1.set_facecolor('#DEE2E6')

gl1 = ax1.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.3, linestyle='--', zorder=4)
gl1.xlabel_style = {'size': 6}
gl1.ylabel_style = {'size': 6}

# Latitude circle
lons = np.linspace(-180, 180, 1000)
lats = np.full_like(lons, -60)
ax1.plot(lons, lats, transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=0.4, zorder=10)

# Colorbar 1
cax1 = fig.add_subplot(gs[1])
cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical', extend='both', pad=0.01, aspect=30)
cbar1.set_label("T [°C]", **label_kwargs)
cbar1.set_ticks([0, 1, 2, 3, 4])
cbar1.ax.tick_params(**tick_kwargs)

# === Subplot 2: Peninsula Sector ===
ax2 = fig.add_subplot(gs[2], projection=ccrs.SouthPolarStereo())
ax2.set_extent([285, 305, -70, -60], crs=ccrs.PlateCarree())

temp2 = temp_clim_peninsula_mean.avg_temp.isel(days=-1)
lon2 = temp_clim_peninsula_mean.lon_rho
lat2 = temp_clim_peninsula_mean.lat_rho

im2 = ax2.pcolormesh(
    lon2, lat2, temp2,
    cmap='Reds',     # change here from 'coolwarm' to 'Reds'
    norm=norm,       # use same normalization
    transform=ccrs.PlateCarree(),
    rasterized=True,
    zorder=1
)

ax2.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
ax2.coastlines(color='black', linewidth=0.4, zorder=3)
ax2.set_facecolor('#DEE2E6')

gl2 = ax2.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.3, linestyle='--', zorder=4)
gl2.xlabel_style = {'size': 6}
gl2.ylabel_style = {'size': 6}

ax2.plot(lons, lats, transform=ccrs.PlateCarree(), color='black', linestyle='--', linewidth=0.4, zorder=10)

# Colorbar 2
cax2 = fig.add_subplot(gs[3])
cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical', extend='both', pad=0.05, aspect=30)
cbar2.set_label("T [°C]", **label_kwargs)
cbar2.set_ticks([0, 1, 2, 3, 4])
cbar2.ax.tick_params(**tick_kwargs)

# === Title ===
if plot == 'slides':
    fig.suptitle("Climatological Mean Temperature (Last Day)", **suptitle_kwargs)

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/SubAreas/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"both_areas_temp_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:
    plt.show()

# %%

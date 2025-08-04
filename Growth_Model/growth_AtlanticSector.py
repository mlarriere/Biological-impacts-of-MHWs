"""
Created on Tue 03 June 16:04:36 2025

How a krill grow during 1 season under MHWs in Atlantic Sector

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

# %% ======================== Load data ========================
# MHW durations
mhw_duration_5m = xr.open_dataset(os.path.join(path_duration, "mhw_duration_5m.nc")).mhw_durations #dataset - shape (40, 365, 434, 1442)
det_combined_ds = xr.open_dataset(os.path.join(path_combined_thesh, 'det_depth5m.nc')) #boolean shape (40, 181, 434, 1442)
mhw_duration_seasonal = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc')))

#%% ======================== Select extent (Atlantic Sector) ========================
# According to Atkinson 2009: 70 % of the entire circumpolar population is concentrated within the Southwest Atlantic (0–90W)
growth_seasons = xr.open_dataset(os.path.join(path_growth, "growth_Atkison2006_seasonal.nc")) #from growth_Atkinson.py
growth_seasons = growth_seasons.rename_vars({'days': 'days_of_yr'})

def subset_spatial_domain(ds, lat_range=(-80, -60), lon_range=(270, 360)): #, (0, 30)
    lat_min, lat_max = lat_range
    lon_range1, lon_range2 = lon_range

    lat_mask = (ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max)
    lon_mask = ((ds['lon_rho'] >= lon_range1) & (ds['lon_rho'] <= lon_range2)) #| ((ds['lon_rho'] >= lon_range2[0]) & (ds['lon_rho'] <= lon_range2[1]))

    return ds.where(lat_mask & lon_mask, drop=True)

# -- Write or load data
growth_file = os.path.join(os.path.join(path_growth, 'atlantic_sector/growth_study_area.nc'))
duration_file = os.path.join(os.path.join(path_growth, 'atlantic_sector/mhw_duration_study_area.nc'))

if not os.path.exists(growth_file):
    growth_study_area = subset_spatial_domain(growth_seasons) #shape (years, eta_rho, xi_rho, days): (39, 360, 385, 181) -- 2019 excluded
    growth_study_area.to_netcdf(os.path.join(path_growth, 'atlantic_sector/growth_study_area.nc')) # Write to file

if not os.path.exists(duration_file):
    mhw_duration_study_area = subset_spatial_domain(mhw_duration_seasonal) #shape (years, days, eta_rho, xi_rho) :(40, 181, 231, 360)
    mhw_duration_study_area.to_netcdf(os.path.join(path_growth, 'atlantic_sector/mhw_duration_study_area.nc')) # Write to file

else: 
    # Load data
    growth_study_area = xr.open_dataset(os.path.join(path_growth, 'atlantic_sector/growth_study_area.nc'))
    mhw_duration_study_area = xr.open_dataset(os.path.join(path_growth, 'atlantic_sector/mhw_duration_study_area.nc'))

#%% ======================== Compute mean growth for two periods ========================
growth_early = growth_study_area.growth.sel(years=slice(1980, 2009)).mean(dim=['years', 'days'], skipna=True)
growth_late = growth_study_area.growth.sel(years=slice(2010, 2018)).mean(dim=['years', 'days'], skipna=True)
growth_diff = growth_late - growth_early

#%% ======================== Plot Study Area climatology and recent periods ========================
# ---- Figure layout ----
plot = 'slides'  # slides report
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = 9.3656988889
    fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    fig.subplots_adjust(hspace=-0.2) 
else:
    fig_width = 16
    fig_height = 8
    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})

title_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 11}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 10}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 13, 'fontweight': 'bold'}

# Circular boundary
theta = np.linspace(np.pi / 2, np.pi, 100)
center, radius = [0.5, 0.51], 0.5 # centered at 0.5,0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

cmap_diff = 'coolwarm'
cmap_var = 'PuOr_r'
if plot == 'report':
    font_size_macro = r'\small'
else:  # slides
    font_size_macro = r'\normalsize'

plot_data = [
    (growth_early,  f"Climatological Growth\n{font_size_macro}{{(1980\\,\\textendash\\,2009)}}", cmap_var, mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.2)),
    (growth_late,   f"Recent Growth\n{font_size_macro}{{(2010\\,\\textendash\\,2018)}}", cmap_var, mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.2)),
    (growth_diff,   f"Difference\n{font_size_macro}{{(recent\\,\\textendash\\,climatological)}}", cmap_diff, mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.1))
]

ims = []
for ax, (data, title, cmap, norm_to_use) in zip(axs, plot_data):
    # Plot data
    im = ax.pcolormesh(growth_study_area.lon_rho, growth_study_area.lat_rho, data,
                       transform=ccrs.PlateCarree(), cmap=cmap, norm=norm_to_use,
                       shading='auto', rasterized=True, zorder=1)
    ims.append(im)

    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map extent and features
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
    ax.coastlines(color='black', linewidth=1)
    ax.set_facecolor('#F6F6F3')
        
    # Sector boundaries
    for lon in [-90, 120, 0]:
            ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=0.5)

    # Gridlines
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
    gl.xlocator = mticker.FixedLocator(np.arange(-80, 1, 20))  # -90, -60, -30, 0
    gl.xformatter = LongitudeFormatter(degree_symbol='°', number_format='.0f', dateline_direction_label=False)
    gl.yformatter = LatitudeFormatter()
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.ylabels_right = False
    gl.xlabels_left = True
    gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    
    if plot == 'report':
        ax.set_title(title, x=0.35, y=1.1, ha='center', **title_kwargs)
    else:
        ax.set_title(title, y=1.1, **title_kwargs)
    
# -----------------------------
# Colorbars
# -----------------------------
tick_positions_growth = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
tick_positions_diff = [-0.1, -0.05, 0.0, 0.05, 0.1]
if plot == 'report':
    x_shift=0.15
    y_shift=0.071

    pos1 = axs[0].get_position()
    pos2 = axs[1].get_position()
    height = (pos1.y1 - pos2.y0) * 0.73  # 80% of the original height
    y0 = pos2.y0 + ((pos1.y1 - pos2.y0) - height) / 2  # center vertically
    cbar_ax1 = fig.add_axes([pos2.x1 - x_shift, y0 + y_shift, 0.015, height])
    cbar1 = fig.colorbar(ims[0], cax=cbar_ax1, cmap=cmap_var, ticks=tick_positions_growth, extend='both')
    cbar1.set_label('Growth [mm/d]', **label_kwargs)
    cbar1.ax.tick_params(**tick_kwargs)

    pos3 = axs[2].get_position()
    cbar_width = 0.015
    cbar_height = pos3.height * 0.5  # reduce height to 70%
    cbar_x = pos3.x1 - x_shift  # move left by 0.07 (from 0.05 to 0.02)
    cbar_y = pos3.y0 + (pos3.height - cbar_height) / 2  # vertically center the shorter colorbar
    cbar_ax2 = fig.add_axes([cbar_x, cbar_y+y_shift, cbar_width, cbar_height])
    cbar2 = fig.colorbar(ims[2], cax=cbar_ax2, cmap=cmap_diff, ticks=tick_positions_diff, extend='both')
    cbar2.set_label("$\Delta$ Growth [mm/d]", **label_kwargs)
    cbar2.ax.tick_params(**tick_kwargs)


else:
    # Shared horizontal colorbar for axs[0] and axs[1]
    pos0 = axs[0].get_position()
    pos1 = axs[1].get_position()
    x_center = (pos0.x0 + pos1.x1) / 2
    width = 0.35
    x_shift = 0.05
    x0 = x_center - width / 2 - x_shift
    y = min(pos0.y0, pos1.y0) + 0.15  # vertical position below both plots

    cbar_ax1 = fig.add_axes([x0, y, width, 0.015])
    cbar1 = fig.colorbar(ims[0], cax=cbar_ax1, cmap=cmap_var, ticks=tick_positions_growth, orientation='horizontal', extend='both')
    cbar1.set_label('Growth [mm/d]', **label_kwargs)
    cbar1.ax.tick_params(**tick_kwargs)

    # Horizontal colorbar for axs[2]
    pos2 = axs[2].get_position()
    cbar_width = 0.008
    cbar_height = pos2.height * 0.5
    cbar_x = pos2.x1 - 0.1  # moved left from 0.01 to 0.03
    cbar_y = pos2.y0 + 0.113 + (pos2.height - cbar_height) / 2
    cbar_ax2 = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar2 = fig.colorbar(ims[2], cax=cbar_ax2, cmap=cmap_diff, ticks=tick_positions_diff, orientation='vertical', extend='both')
    cbar2.set_label("$\Delta$ Growth [mm/d]", **label_kwargs)
    cbar2.ax.tick_params(**tick_kwargs)


# --- Axis labels and title ---
if plot == 'report':
    suptitle_y = 1
    fig.suptitle(f'Antarctic Krill Growth - Atlantic Sector', y=suptitle_y, **suptitle_kwargs)
    fig.text(0.5, suptitle_y - 0.03, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', **label_kwargs, style='italic')

else:
    suptitle_y = 0.9
    fig.suptitle(f'Antarctic Krill Growth - Atlantic Sector', y=suptitle_y, **suptitle_kwargs)
    fig.text(0.5, suptitle_y - 0.055, 'Growth season (1Nov–30Apr), 1980–2018', ha='center', **label_kwargs, style='italic')

# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/case_study_AtlanticSector')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"growth_diff_atlantic_sect_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/growth_diff_atlantic_sect_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()


# %% ======================== Find years with maximum occurence of MHWs events ========================
# Set threshold
duration_thresh = 30  # Minimum duration of MHW event
duration_mask = mhw_duration_study_area['duration'] > duration_thresh

# Function to find years with maximum MHWs events for each threshold
def process_threshold(threshold_deg):
    # threshold_deg = 1
    print(f'\n ---------------- Processing {threshold_deg}°C threshold ---------------- ')
    det_key = f'det_{threshold_deg}deg'
    intensity_mask = mhw_duration_study_area[det_key].astype(bool)
    
    # Filter duration & threshold
    combined_mask = duration_mask & intensity_mask

    # Events respecting mask
    valid_events = mhw_duration_study_area['duration'].where(combined_mask,  drop=True)
    valid_mask = ~np.isnan(valid_events.values)

    # Get years where valid events occur
    valid_indices = np.array(np.nonzero(valid_mask)).T
    years = valid_events['years'].values
    years_list = [years[idx[0]] for idx in valid_indices]

    # Count events per year
    year_counts = collections.Counter(years_list)
    top3_years = year_counts.most_common(5)
    top3_actual_years = [(1980 + y, count) for y, count in top3_years]
    print(f'{threshold_deg}°C done \n')

    return (threshold_deg, top3_actual_years)

# Run in parallel
from tqdm.contrib.concurrent import thread_map
thresholds = [1, 2, 3, 4]
results = thread_map(process_threshold, thresholds, max_workers=4)

for threshold_deg, top3_years in results:
    print(f"\nTop 5 years with most {threshold_deg}°C MHW events:")
    for year, count in top3_years:
        print(f"  Year: {year}, Events: {count}")

# Top 5 years with most 1°C MHW events: 2016, 1989, 1988, 2018, 2000
# Top 5 years with most 2°C MHW events: 1989, 2016, 2000, 1984, 2009
# Top 5 years with most 3°C MHW events: 1989, 2000, 1984, 2002, 1987
# Top 5 years with most 4°C MHW events: 1989, 2000, 1984, 2002, 1987

# %% ======================== Find years with maximum extent of MHWs events ========================
def process_threshold(threshold_deg):
    print(f'\n ---------------- Processing {threshold_deg}°C threshold ---------------- ')
    
    det_key = f'det_{threshold_deg}deg'
    intensity_mask = mhw_duration_study_area[det_key].astype(bool)
    combined_mask = duration_mask & intensity_mask

    valid_events = mhw_duration_study_area['duration'].where(combined_mask, drop=True)
    valid_mask = ~np.isnan(valid_events.values)  # (years, days, eta, xi)

    # Identify grid points with MHW any day per year
    mhw_presence = np.any(valid_mask, axis=1)  # shape (years, eta, xi)

    # Count grid points per year
    spatial_extent = np.sum(mhw_presence, axis=(1, 2))  # shape (years,)

    years = np.arange(1980, 1980 + spatial_extent.shape[0])

    # Find top 5 years by spatial extent
    top5_idx = np.argsort(spatial_extent)[::-1][:5]
    top5_years = years[top5_idx]
    top5_extent = spatial_extent[top5_idx]

    print(f'{threshold_deg}°C done')
    return (threshold_deg, top5_years, top5_extent)


from tqdm.contrib.concurrent import thread_map
thresholds = [1, 2, 3, 4]
results = thread_map(process_threshold, thresholds, max_workers=4)

for threshold_deg, top5_years, top5_extent in results:
    print(f"\nTop 5 years with biggest {threshold_deg}°C MHW extent:")
    for year, extent in zip(top5_years, top5_extent):
        # print(f"{year}")
        print(f"Year: {year}, MHW spatial extent: {extent} grid points")

# Top 5 years with biggest 1°C MHW extent: 2016, 1989, 1988, 2018, 2000
# Top 5 years with biggest 2°C MHW extent: 2016, 1989, 1988, 2018, 2000
# Top 5 years with biggest 3°C MHW extent:1989, 2016, 2000, 1984, 2002
# Top 5 years with biggest 4°C MHW extent: 1989, 1984, 2000, 2002, 2009

# %% ======================== Selected years ========================
selected_years = [1989, 2000, 2016]
selected_years_idx = np.array(selected_years) - 1980  # [9, 20, 36]

#%% ======================== Plot Study Area for 1 year ========================
# Year of interest
yr_chosen=2
year_index = selected_years_idx[yr_chosen] 

# Threshold info
threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']

# === Layout config ===
plot = 'report' #report slides
if plot == 'report':
    fig_width = 6.3228348611
    fig_height =fig_width/2
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    plt.subplots_adjust(wspace=0.45)
else:
    fig_width = 16
    fig_height = 8
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    plt.subplots_adjust(wspace=0.05)


title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 8}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 7}
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {'fontsize': 8}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 12, 'fontweight': 'bold'}

# === Subplot 1: Detected Events ===
# Base: where no MHW was detected at any threshold
no_event_mask = ((mhw_duration_study_area['det_1deg'].isel(years=year_index).mean(dim='days') == 0) &
                 (mhw_duration_study_area['det_2deg'].isel(years=year_index).mean(dim='days') == 0) &
                 (mhw_duration_study_area['det_3deg'].isel(years=year_index).mean(dim='days') == 0) &
                 (mhw_duration_study_area['det_4deg'].isel(years=year_index).mean(dim='days') == 0)).fillna(True)

# Plot no detected MHW in white
ax0.contourf(mhw_duration_study_area.lon_rho, mhw_duration_study_area.lat_rho, no_event_mask, 
             levels=[0.5, 1], colors=['white'], transform=ccrs.PlateCarree(), zorder=1)

# Plot detected MHW events for each threshold
for var, color in zip(threshold_vars, threshold_colors):
    event_mask = mhw_duration_study_area[var].isel(years=year_index).mean(dim='days').fillna(0)
    binary_mask = (event_mask >= 0.166).astype(int)  # 1 if event ≥ 30 days else 0
    ax0.contourf(mhw_duration_study_area.lon_rho, mhw_duration_study_area.lat_rho, binary_mask, 
                 levels=[0.5, 1], colors=[color], 
                 transform=ccrs.PlateCarree(), 
                 alpha=0.8, zorder=2)
    
# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='white', edgecolor='black', label='No MHW event', linewidth=0.5)]
legend_handles += [Patch(facecolor=c, edgecolor='black', label=l, linewidth=0.5) for c, l in zip(threshold_colors, threshold_labels)]
if plot == 'report':
    legend_box = (0.85, 0.95)
else:
    legend_box = (0.67, 0.73)
legend = ax0.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=legend_box, ncol=1, frameon=True, **legend_kwargs)
legend.get_frame().set_linewidth(0.5)  # thinner box line

# === Subplot 2: Growth ===
growth_data = growth_study_area.growth.isel(years=year_index).mean(dim='days')
growth_plot = growth_data.plot.pcolormesh(ax=ax1, transform=ccrs.PlateCarree(), x='lon_rho', y='lat_rho', 
                                          cmap='PuOr_r', add_colorbar=False, norm=mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2), rasterized=True)

# === Colorbar
pos = ax1.get_position()
if plot == 'report':
    cbar_width = 0.012
    cbar_height = pos.height * 0.5
    cbar_x = pos.x1 - 0.13
    cbar_y = pos.y0 + (pos.height - cbar_height) / 2 + 0.155 # vertically centered

else:
    cbar_width = 0.012
    cbar_height = pos.height * 0.5
    cbar_x = pos.x1 - 0.185
    cbar_y = pos.y0 + (pos.height - cbar_height) / 2 + 0.19 # vertically centered

cbar_ax1 = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
cbar1 = fig.colorbar(growth_plot, cax=cbar_ax1, orientation='vertical', extend='both')
cbar1.set_label("Growth [mm/d]", **label_kwargs)
cbar1.ax.tick_params(**tick_kwargs)

# Maps Features
theta = np.linspace(np.pi/2, np.pi , 100)  # from 0° to -90° clockwise - Quarter-circle sector boundary
center, radius = [0.5, 0.51], 0.5 # centered at 0.5,0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

for ax in [ax0, ax1]:
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map extent and features
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)  # Land should be drawn above the plot
    lw = 0.7 if plot == 'slides' else 0.5
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.set_facecolor('#F6F6F3')
        
    # Sector boundaries
    for lon in [-90, 120, 0]:
            ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=0.5)

    # Gridlines
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7, zorder=7)
    gl.xlocator = mticker.FixedLocator(np.arange(-80, 1, 20))  # -90, -60, -30, 0
    gl.xformatter = LongitudeFormatter(degree_symbol='°', number_format='.0f', dateline_direction_label=False)
    gl.yformatter = LatitudeFormatter()
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.ylabels_right = False
    gl.xlabels_left = True
    gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs

# === Titles ===
if plot == 'report':
    ax0.set_title(f"Long-lasting MHW events\n$\\small{{{selected_years[yr_chosen]}\\textendash {selected_years[yr_chosen]+1}}}$", x=0.35, y=1.05, ha='center', **title_kwargs)
    ax1.set_title(f"Mean growth\n$\\small{{{selected_years[yr_chosen]}\\textendash {selected_years[yr_chosen]+1}}}$", x=0.35, y=1.05, ha='center', **title_kwargs)

else:
    ax0.set_title(f"Long-lasting MHW events\n$\\small{{{selected_years[yr_chosen]}\\textendash {selected_years[yr_chosen]+1}}}$", x=0.35, y=1.05, **title_kwargs)
    ax1.set_title(f"Mean growth\n$\\small{{{selected_years[yr_chosen]}\\textendash {selected_years[yr_chosen]+1}}}$", x=0.35,  y=1.05, **title_kwargs)

# --- Axis labels and title ---
if plot == 'report':
    suptitle_y = 1.05
    fig.suptitle(f'Growth and MHW events from 1st November {selected_years[yr_chosen]} until 30th April {selected_years[yr_chosen]+1}', y=suptitle_y, **suptitle_kwargs)
else:
    suptitle_y = 1.05
    fig.suptitle(f'Growth and MHW events from 1st November {selected_years[yr_chosen]} until 30th April {selected_years[yr_chosen]+1}', y=suptitle_y, **suptitle_kwargs)
# fig.text(0.5, 0.1, f"Note that on panel2, frequent corresponds to events lasting $\geq${round(181*0.166)} days during the growing season", ha='center')


# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/case_study_AtlanticSector')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"atlantic_sector{selected_years[yr_chosen]}_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

# %%  ======================== INPUTS growth calculation ========================
# Year of interest
yr_chosen=2
year_index = selected_years_idx[yr_chosen] 

# ==== Temperature [°C] 
# Weighted averaged temperature of the first 100m - Austral summer - 60S - years = seasonal (i.e. ranging from 1980 to 2018 with days 304-119)
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (39, 181, 231, 1442)
temp_avg_100m_study_area_allyrs = subset_spatial_domain(temp_avg_100m_SO_allyrs) #select spatial extent -- shape (39, 181, 231, 360)
temp_avg_100m_study_area_1season = temp_avg_100m_study_area_allyrs.isel(years=year_index) #select temporal extent for 1 year of interest -- shape (181, 231, 360)

# ==== Chla [mh Chla/m3] 
# Weighted averaged chla of the first 100m - Austral summer - 60S - years = seasonal (i.e. ranging from 1980 to 2018 with days 304-119)
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 
chla_surf_study_area_allyrs = subset_spatial_domain(chla_surf_SO_allyrs) #select spatial extent
chla_surf_study_area_1season = chla_surf_study_area_allyrs.isel(years=year_index) #select temporal extent for 1 year of interest -- shape (181, 231, 360)

#%% ============== Equation decomposition ==============
def decompose_growth(chla, temp, length=35):
    """Returns the breakdown of growth components."""
    # Constants
    a = np.mean([-0.196, -0.216])
    b, c = 0.00674, -0.000101
    d, e = 0.377, 0.321
    f, g = 0.013, -0.0115

    const_value =  b * length + c * length**2
    const_term = xr.full_like(chla, const_value)
    chla_term = a + (d * chla) / (e + chla)
    temp_term = f * temp + g * temp**2
    total_growth = const_term + chla_term + temp_term

    return xr.Dataset({
        "const_term": const_term,
        "chla_term": chla_term,
        "temp_term": temp_term,
        "total_growth": total_growth
    })

# Region-mean timeseries
# chl_series = chla_surf_study_area_1season['chla'].mean(dim=['eta_rho', 'xi_rho'])
# temp_series = temp_avg_100m_study_area_1season['avg_temp'].mean(dim=['eta_rho', 'xi_rho'])
chl_series = chla_surf_study_area_allyrs['chla'].mean(dim=['years', 'eta_rho', 'xi_rho'])
temp_series = temp_avg_100m_study_area_allyrs['avg_temp'].mean(dim=['years', 'eta_rho', 'xi_rho'])

# Decompose
growth_decomp = decompose_growth(chl_series, temp_series)

# ======== PLOT
plot='slides' #report slides

if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width *1.5
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)

else:  # 'slides'
    fig_width = 16 
    fig_height = 9 
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

from datetime import datetime, timedelta
base_date = datetime(2021, 11, 1)  # Nov 1 (season start)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
days_xaxis = np.arange(181)  # Just 0 to 180 (1 per day)

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}

# Format x-axis
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
selected_ticks = [(day, label) for day, label in date_dict.items() if label in wanted_labels]
selected_ticks.sort()  # Sort by day to ensure chronological order
tick_positions, tick_labels = zip(*selected_ticks)

# --- Subplot 1: Drivers only ---
axes[0].plot(days_xaxis, growth_decomp['chla_term'].values, label='Chla term', color='#385129')
axes[0].plot(days_xaxis, growth_decomp['temp_term'].values, label='T°C term', color='#AD460B')
axes[0].plot(days_xaxis, growth_decomp['const_term'].values, label='Length term', color='#942624')
axes[0].set_ylabel("Growth [mm/d]", **label_kwargs)
axes[0].set_ylim(-0.1, 0.25)
if plot == 'slides':
    axes[0].set_xlabel("Date", **label_kwargs)
axes[0].legend(loc='upper left', **legend_kwargs)
axes[0].grid(False)
axes[0].tick_params(**tick_kwargs)

# --- Subplot 2: Sums and total with drivers faint ---
alpha_driver = 0.6
axes[1].plot(days_xaxis, growth_decomp['chla_term'].values, color='#385129', alpha=alpha_driver)
axes[1].plot(days_xaxis, growth_decomp['temp_term'].values, color='#AD460B', alpha=alpha_driver)
axes[1].plot(days_xaxis, growth_decomp['const_term'].values, color='#942624', alpha=alpha_driver)

axes[1].plot(days_xaxis, (growth_decomp['chla_term'] + growth_decomp['temp_term']).values, label='(Chla + T°C) terms', linestyle='--', color='#385129')
axes[1].plot(days_xaxis, (growth_decomp['chla_term'] + growth_decomp['const_term']).values, label='(Chla + Length) terms', linestyle='--', color='#942624')
axes[1].plot(days_xaxis, growth_decomp['total_growth'].values, label='Total growth', color='black', linewidth=2)
if plot == 'report':
    axes[1].set_ylabel("Growth [mm/d]", **label_kwargs)
axes[1].set_xlabel("Date", **label_kwargs)
axes[1].set_ylim(-0.1, 0.25)
axes[1].legend(loc='upper left', **legend_kwargs)
axes[1].grid(False)
axes[1].tick_params(**tick_kwargs)

for ax in axes:
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

if plot == 'report':
    suptitle_y = 0.92
    title_text = ("Mean Krill Growth in the Atlantic Sector\n"
                  r"\small{Component-wise Contribution from the Atkinson Equation}""\n"r"\small{Growth season (1980-2018)}")
else:  # slides
    suptitle_y = 0.93
    title_text = ("Mean Krill Growth in the Atlantic Sector\n"
                  "Component-wise Contribution from the Atkinson Equation\nGrowth season (1980-2018)")

fig.suptitle(title_text, **maintitle_kwargs, y=suptitle_y)

# --- Output handling ---    
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave space at bottom for colorbar
if plot == 'report':    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/eq_decomposition/Atkison_decomp_{plot}.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/eq_decomposition/Atkison_decomp_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()

#%% ============== Plot drivers ============== 
temp_mean_ts = temp_avg_100m_study_area_1season['avg_temp'].mean(dim=['eta_rho', 'xi_rho'])
chla_mean_ts = chla_surf_study_area_1season['chla'].mean(dim=['eta_rho', 'xi_rho'])
days = temp_avg_100m_study_area_1season['avg_temp'].days.values
days_xaxis = np.where(days < 304, days+ 365, days).astype(int)
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0+365, 121+365)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

plot='report' #slides report
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width/1.5
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

else:  # 'slides'
    fig_width = 9 
    fig_height = 6 
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 9}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}

# --- Axis 1: temperature ---
ax1.plot(days_xaxis, temp_mean_ts, color='#F3722C', label='Mean Temperature (100m)')
ax1.set_xlabel("Date", **label_kwargs)
ax1.set_ylabel("Temperature [°C]", color='#F3722C', **label_kwargs)
ax1.tick_params(axis='y', labelcolor='#F3722C', **tick_kwargs)

# --- Axis 2: chlorophyll ---
ax2 = ax1.twinx()
ax2.plot(days_xaxis, chla_mean_ts, color='green', label='Mean Surface Chla')
ax2.set_ylabel("Chlorophyll [mg/m³]", color='green', **label_kwargs)
ax2.tick_params(axis='y', labelcolor='green', **tick_kwargs)

# Format x-axis
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []
for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)

ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, fontsize=tick_kwargs.get("labelsize", 10))

# --- Title
suptitle_y = 0.98 if plot == 'slides' else 0.95
fig.suptitle("Time Series of Temperature and Chlorophyll", y=suptitle_y, **maintitle_kwargs)
ax1.set_title(f"Atlantic Sector, Growth Season {selected_years[yr_chosen]}", **subtitle_kwargs)

# --- Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
all_lines = lines_1 + lines_2
all_labels = labels_1 + labels_2

if plot == 'report':
    ax1.legend( all_lines, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.3),
        ncol=2, frameon=True, **legend_kwargs)
else:
    ax1.legend(all_lines, all_labels, loc='upper left',bbox_to_anchor=(0.15, -0.3),
               ncol=2, frameon=True, **legend_kwargs)


# --- Output handling ---
plt.tight_layout()
outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/case_study_AtlanticSector')
os.makedirs(outdir, exist_ok=True)
if plot == 'report':
    outfile = f"drivers_{selected_years[yr_chosen]}_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    outfile = f"drivers_{selected_years[yr_chosen]}_{plot}.png"
    # plt.savefig(os.path.join(outdir, outfile), dpi=500, format='png', bbox_inches='tight')
    plt.show()    


# %% ---- Valid Chla data
valid_fraction = (~np.isnan(chla_surf_study_area_1season['chla'])).mean(dim=['eta_rho', 'xi_rho'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(days_xaxis, valid_fraction, color='blue', label='Valid Chla Fraction')
ax.set_title(f"Fraction of Valid (Non-NaN) Chlorophyll Values Over Time \nNov 1st {selected_years[yr_chosen]} - Apr. 30 {selected_years[yr_chosen]+1}", fontsize=16)
ax.set_ylabel("Fraction of Valid Grid Cells", fontsize=14)
ax.set_xlabel("Date", fontsize=14)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=12)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# ----- CHLA all cells in the region
fig_all, ax_all = plt.subplots(figsize=(10, 6))
ntime, nlat, nlon = chla_surf_study_area_1season['chla'].shape
for i in range(nlat):
    for j in range(nlon):
        chla_ts = chla_surf_study_area_1season['chla'][:, i, j].values
        ax_all.plot(days_xaxis, chla_ts, color='green', alpha=0.1, linewidth=0.5)
ax_all.plot(days_xaxis, chla_mean_ts, color='darkgreen', linewidth=2, label='Mean Surface Chla')
ax_all.set_xticks(tick_positions)
ax_all.set_xticklabels(tick_labels, rotation=45, fontsize=12)
ax_all.set_xlabel("Date", fontsize=14)
ax_all.set_ylabel("Chlorophyll (mg/m³)", fontsize=14)
ax_all.grid(True)
ax_all.set_title(f"All Chlorophyll Time Series + Mean\nAustral Summer\nNov 1st {selected_years[yr_chosen]} - Apr. 30 {selected_years[yr_chosen]+1}", fontsize=16)
ax_all.legend(fontsize=12, loc='upper left')
plt.show()

#%% ============== Chla and Temperature under MHWs and NON MHWs ==============
# Output file paths
temp_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/temp_avg100m_daily_mhw.nc")
chla_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/chla_surf_daily_mhw.nc")
temp_non_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/temp_avg100m_daily_nomhw.nc")
chla_non_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/chla_surf_daily_nomhw.nc")

if not (os.path.exists(temp_mhw_file) and os.path.exists(temp_non_mhw_file) and
        os.path.exists(chla_mhw_file) and os.path.exists(chla_non_mhw_file)):

    ds_duration_thresh_SEASONyear = xr.open_dataset(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))
    ds_duration_thresh_SEASONyear['years'] = ds_duration_thresh_SEASONyear['years'] + 1980

    # ------ MHWs ------
    temp_mhw = xr.Dataset()
    chla_mhw = xr.Dataset()

    duration_mask = ds_duration_thresh_SEASONyear['duration'] >= 30

    for var in ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']:
        print(f'Variable --- {var}')
        suffix_var = var.replace('det_', '')
        temp_var = f"temp_{suffix_var}"
        chla_var = f"chla_{suffix_var}"

        det_mask = ds_duration_thresh_SEASONyear[var] == 1
        mhw_mask = duration_mask & det_mask
        mhw_mask_study_area = subset_spatial_domain(mhw_mask)

        # print(f"{var} coverage (sum of True pixels):", mhw_mask_study_area.isel(years=year_index).sum().values)

        mhw_mask_year20 = mhw_mask_study_area.isel(years=year_index)
        mhw_days_sum = mhw_mask_year20.sum(dim=['eta_rho', 'xi_rho'])
        mhw_days = mhw_days_sum.where(mhw_days_sum > 0, drop=True)
        # print("Days in Atlantic Sector with MHW in year 2000:", mhw_days['days'].values)

        mhw_mask_clean = mhw_mask_study_area.drop_vars('days').rename({'days_of_yr': 'days'})

        masked_temp = temp_avg_100m_study_area_allyrs['avg_temp'].where(mhw_mask_clean.values.astype(bool))
        masked_chla = chla_surf_study_area_allyrs['chla'].where(mhw_mask_clean.values.astype(bool))

        temp_mhw[temp_var] = masked_temp
        chla_mhw[chla_var] = masked_chla

    # Metadata
    temp_mhw.attrs = {
        "description": "Averaged temperature (100m) during MHW events.",
        "mhw_criteria": "≥30 days duration + absolute thresholds (1–4°C)",
        "temporal_extent": "seasonal",
        "spatial_extent": "Atlantic Sector (90°W–0°)",
        "creation_date": "2025-07-02",
    }
    chla_mhw.attrs = {
        "description": "Surface chlorophyll during MHW events.",
        "mhw_criteria": "≥30 days duration + absolute thresholds (1–4°C)",
        "temporal_extent": "seasonal",
        "spatial_extent": "Atlantic Sector (90°W–0°)",
        "creation_date": "2025-07-02",
    }

    temp_mhw.to_netcdf(temp_mhw_file)
    chla_mhw.to_netcdf(chla_mhw_file)

    # ------ Non MHWs ------
    temp_non_mhw = xr.Dataset()
    chla_non_mhw = xr.Dataset()

    nonmhw_mask = ds_duration_thresh_SEASONyear['duration'] == 0
    nonmhw_mask_study_area = subset_spatial_domain(nonmhw_mask)
    nonmhw_mask_clean = nonmhw_mask_study_area.drop_vars('days').rename({'days_of_yr': 'days'})

    masked_temp = temp_avg_100m_study_area_allyrs['avg_temp'].where(nonmhw_mask_clean.values.astype(bool))
    masked_chla = chla_surf_study_area_allyrs['chla'].where(nonmhw_mask_clean.values.astype(bool))
    temp_non_mhw['temp_nonmhw'] = masked_temp
    chla_non_mhw['chla_nonmhw'] = masked_chla

    # Metadata
    temp_non_mhw.attrs = {
        "description": "Averaged temperature (100m) during non-MHW periods (duration == 0).",
        "temporal_extent": "seasonal",
        "spatial_extent": "Atlantic Sector (90°W–0°)",
        "creation_date": "2025-07-02",
    }
    chla_non_mhw.attrs = {
        "description": "Surface chlorophyll during non-MHW periods (duration == 0).",
        "temporal_extent": "seasonal",
        "spatial_extent": "Atlantic Sector (90°W–0°)",
        "creation_date": "2025-07-02",
    }

    temp_non_mhw.to_netcdf(temp_non_mhw_file)
    chla_non_mhw.to_netcdf(chla_non_mhw_file)

else:
    # Load existing datasets
    temp_mhw = xr.open_dataset(temp_mhw_file)
    chla_mhw = xr.open_dataset(chla_mhw_file)
    temp_non_mhw = xr.open_dataset(temp_non_mhw_file)
    chla_non_mhw = xr.open_dataset(chla_non_mhw_file)

# Selecting temporal extent for 1 year of interest -- shape (181, 231, 360)
temp_mhw_study_area_1season = temp_mhw.isel(years=year_index) 
chla_mhw_study_area_1season = chla_mhw.isel(years=year_index) 
temp_non_mhw_study_area_1season = temp_non_mhw.isel(years=year_index)
chla_non_mhw_study_area_1season = chla_non_mhw.isel(years=year_index)


# %% ============== Calculating length ==============
from Growth_Model.growth_model import length_Atkison2006  

# == Length Southern Ocean
# simulated_length_full_SO = length_Atkison2006(chla=chla_surf_1season_SO.raw_chla, temp=temp_avg_100m_1season_SO.avg_temp, initial_length= 35, intermoult_period=10)
# mean_length_area_SO = simulated_length_full_SO.mean(dim=["eta_rho", "xi_rho"])

# == Length Atlantic Sector for all years
lengths_allyears = []

for yr in range(39):
    print(f' -- Processing {1980+yr}')
    chla = chla_surf_study_area_allyrs.chla.isel(years=yr)
    temp = temp_avg_100m_study_area_allyrs.avg_temp.isel(years=yr)

    simulated_length = length_Atkison2006(chla=chla, temp=temp, initial_length=35, intermoult_period=10)

    lengths_allyears.append(simulated_length)

# Combine and assign years
simulated_length_study_area_1980_2019 = xr.concat(lengths_allyears, dim='years')
simulated_length_study_area_1980_2019 = simulated_length_study_area_1980_2019.assign_coords(years=chla_surf_study_area_allyrs['years'])

# == Length Atlantic Sector for 1 season of interest
# simulated_length_study_area_1season = length_Atkison2006(chla=chla_surf_study_area_1season.chla, temp=temp_avg_100m_study_area_1season.avg_temp, 
                                                        #  initial_length= 35, intermoult_period=10)

# == Length Non MHWS Atlantic Sector for 1 season of interest
# simulated_length_study_area_non_MHWs = length_Atkison2006(chla=chla_non_mhw_study_area_1season.chla_nonmhw, temp=temp_non_mhw_study_area_1season.temp_nonmhw, initial_length= 35, intermoult_period=10)

# # == Length MHWS Atlantic Sector for 1 season of interest
# simulated_length_study_area_MHWs_1deg = length_Atkison2006(chla=chla_mhw_study_area_1season.chla_1deg, temp=temp_mhw_study_area_1season.temp_1deg, initial_length= 35, intermoult_period=10)
# simulated_length_study_area_MHWs_2deg = length_Atkison2006(chla=chla_mhw_study_area_1season.chla_2deg, temp=temp_mhw_study_area_1season.temp_2deg, initial_length= 35, intermoult_period=10)
# simulated_length_study_area_MHWs_3deg = length_Atkison2006(chla=chla_mhw_study_area_1season.chla_3deg, temp=temp_mhw_study_area_1season.temp_3deg, initial_length= 35, intermoult_period=10)
# simulated_length_study_area_MHWs_4deg = length_Atkison2006(chla=chla_mhw_study_area_1season.chla_4deg, temp=temp_mhw_study_area_1season.temp_4deg, initial_length= 35, intermoult_period=10)


# %% ======================== Extracting the growth rate ========================
# Calculate length on a daily basis
simulated_length_1season_daily = length_Atkison2006(chla=chla_surf_study_area_1season.chla, temp=temp_avg_100m_study_area_1season.avg_temp, 
                                                         initial_length= 35, intermoult_period=1)

print(simulated_length_1season_daily.isel(eta_rho=98, xi_rho=136).values)

# One step back - extracting growth rate (mm/d) between each days
daily_growth = simulated_length_1season_daily.diff(dim='days') #shape (180, 231, 360)
print(daily_growth.isel(eta_rho=98, xi_rho=136).values)

# -- Create masks for the different MHW scenarios (Use Temp Mask - Boolean: where MHW occurred)
mhw_1 = xr.where(~np.isnan(temp_mhw.temp_1deg.isel(years=year_index)), 1, 0)
mhw_2 = xr.where(~np.isnan(temp_mhw.temp_2deg.isel(years=year_index)), 1, 0)
mhw_3 = xr.where(~np.isnan(temp_mhw.temp_3deg.isel(years=year_index)), 1, 0)
mhw_4 = xr.where(~np.isnan(temp_mhw.temp_4deg.isel(years=year_index)), 1, 0)

# Give priority to MHW -- assign highest MHW level if overlapping -- to avoid counting cell twice
# mhw_mask = xr.where(mhw_4 == 1, 4,
#             xr.where(mhw_3 == 1, 3,
#             xr.where(mhw_2 == 1, 2,
#             xr.where(mhw_1 == 1, 1, 0))))  # 0 = no MHW
# valid_mask = (~np.isnan(daily_growth)) & (mhw_mask.notnull())
# growth_by_mhw_level = {
#     level: daily_growth.where((mhw_mask == level) & valid_mask)
#     for level in range(5)
# }

# Non non-exclusive masks, i.e. a 4°C MHW also counted as a 1°C
growth_by_mhw_level = {
    1: daily_growth.where(mhw_1 == 1),
    2: daily_growth.where(mhw_2 == 1),
    3: daily_growth.where(mhw_3 == 1),
    4: daily_growth.where(mhw_4 == 1),
    0: daily_growth.where((mhw_1 + mhw_2 + mhw_3 + mhw_4) == 0)  # explicitly outside any MHW
}

# -- Compute length under the different scenarios
initial_length = 35
intermoult_period = 10
n_days = daily_growth.sizes["days"]+1

length_by_mhw_level = {}
for level in range(5):
    growth = growth_by_mhw_level[level]  # shape: (days, eta_rho, xi_rho)

    # 1. Mean growth over space for each day (daily)
    daily_mean_growth = growth.mean(dim=["eta_rho", "xi_rho"], skipna=True)

    # 2. Calculate length under the different scenarios -- with IMP=10days
    # Average over 10-day blocks 
    growth_blocks = []
    for i in range(0, n_days, intermoult_period):
        block = daily_mean_growth.isel(days=slice(i, min(i + intermoult_period, n_days)))
        avg_block_growth = block.mean().item()  # scalar growth per day in this block
        growth_blocks.extend([avg_block_growth] * len(block))  # same growth each day in block
    
    # Length
    length_series = [initial_length]
    current_length = initial_length
    for i in range(1, n_days):  # Start from 1 because day 0 is initial
        if i % intermoult_period == 0:
            current_length += growth_blocks[i - 1]  # Apply growth from previous full day
        length_series.append(current_length)

    # Convert to DataArray
    length_by_mhw_level[level] = xr.DataArray(
        data=length_series,
        dims=["days"],
        coords={"days": simulated_length_1season_daily.days}
    )

# -- Do the same but mean avg over full period - disregarding mhws 
# Mean daily growth per day
daily_mean_growth_full_extent = daily_growth.where((~np.isnan(daily_growth))).mean(dim=["eta_rho", "xi_rho"], skipna=True)

# Average over 10-day blocks
growth_blocks_full_extent = []
for i in range(0, n_days, intermoult_period):
    block = daily_mean_growth_full_extent.isel(days=slice(i, min(i + intermoult_period, n_days)))
    avg_block_growth = block.mean().item()
    growth_blocks_full_extent.extend([avg_block_growth] * len(block))

# Length
length_full_extent = [initial_length]
current_length_full_extent = initial_length
for i in range(1, n_days):  # Start from 1 because day 0 is initial
    if i % intermoult_period == 0:
        current_length_full_extent += growth_blocks_full_extent[i - 1]  # Apply growth from previous full day
    length_full_extent.append(current_length_full_extent)

length = np.array(length_full_extent)


print(f"Final krill length (no MHW distinction): {length[-1]:.2f} mm")
print("Final krill length at day 180 under each MHW category:")
for level in range(5):
    final_length = length_by_mhw_level[level].isel(days=-1).item()
    print(f"  MHW Level {level}: {final_length:.2f} mm")


# Rename
weighted_mean_length_1season = xr.DataArray(data=length, dims=["days"], coords={"days": simulated_length_1season_daily.days}, name="weighted_mean_length_1season")
mean_length_study_area_non_MHWs = xr.DataArray(data=length_by_mhw_level[0], dims=["days"], coords={"days": simulated_length_1season_daily.days}, name="mean_length_study_area_non_MHWs")
average_length_ts_1deg = xr.DataArray(data=length_by_mhw_level[1], dims=["days"], coords={"days": simulated_length_1season_daily.days}, name="average_length_ts_1deg")
average_length_ts_2deg = xr.DataArray(data=length_by_mhw_level[2], dims=["days"], coords={"days": simulated_length_1season_daily.days}, name="average_length_ts_2deg")
average_length_ts_3deg = xr.DataArray(data=length_by_mhw_level[3], dims=["days"], coords={"days": simulated_length_1season_daily.days}, name="average_length_ts_3deg")
average_length_ts_4deg = xr.DataArray(data=length_by_mhw_level[4], dims=["days"], coords={"days": simulated_length_1season_daily.days}, name="average_length_ts_4deg")

print(f"Full seasonal mean length (weighted): {weighted_mean_length_1season[-1].values:.2f} mm")
print(f"Non MHW mean length:                  {mean_length_study_area_non_MHWs[-1].values:.2f} mm")
print(f"MHW 1°C mean length:                  {average_length_ts_1deg[-1].values:.2f} mm")
print(f"MHW 2°C mean length:                  {average_length_ts_2deg[-1].values:.2f} mm")
print(f"MHW 3°C mean length:                  {average_length_ts_3deg[-1].values:.2f} mm")
print(f"MHW 4°C mean length:                  {average_length_ts_4deg[-1].values:.2f} mm")


# %% ======================== Spatial Average - weighted by exposure time ========================
# # -- Number of days under MHWs per cell
# mhws_exposure_days = xr.open_dataset(os.path.join(path_det, 'nb_days_underMHWs_5mFULL.nc')) #shape: (39, 231, 1442)
# mhws_exposure_days_Atl = subset_spatial_domain(mhws_exposure_days) #shape: (39, 231, 360)
# mhws_exposure_days_Atl_1yr = mhws_exposure_days_Atl.isel(years=year_index)

# # -- Assigning cells to only 1 scenarios 
# # i.e. cell that have been exposed in a 4°C mhw are only conisdered in a 4°C and not in 1 2 3 and non_mhw. Avoiding counting cells multiple times
# deg1_mask = mhws_exposure_days_Atl_1yr.nb_days_1deg > 0
# deg2_mask = mhws_exposure_days_Atl_1yr.nb_days_2deg > 0
# deg3_mask = mhws_exposure_days_Atl_1yr.nb_days_3deg > 0
# deg4_mask = mhws_exposure_days_Atl_1yr.nb_days_4deg > 0
# any_mhw_mask = (deg1_mask | deg2_mask | deg3_mask | deg4_mask)
# non_mhw_mask = ~any_mhw_mask

# # Priority assignment of scenarios to avoid multiple counting
# scenario_mask = np.full(deg1_mask.shape, np.nan) #shape: (eta, xi)
# scenario_mask[deg4_mask.values] = 4
# scenario_mask[np.isnan(scenario_mask) & deg3_mask.values] = 3
# scenario_mask[np.isnan(scenario_mask) & deg2_mask.values] = 2
# scenario_mask[np.isnan(scenario_mask) & deg1_mask.values] = 1
# scenario_mask[np.isnan(scenario_mask) & non_mhw_mask.values] = 0
# # Convert to DataArray
# scenario_mask = xr.DataArray(scenario_mask, coords=deg1_mask.coords, dims=deg1_mask.dims)

# # --- Valid environmental cells (cells where both chla and temperature data are valid)
# def get_valid_environmental_mask(chla, temp):
#     return chla.notnull() & temp.notnull()

# # Apply mask (valid if any day has valid data) - valid over time
# env_mask = get_valid_environmental_mask(chla_surf_study_area_1season.chla, temp_avg_100m_study_area_1season.avg_temp)
# env_static_mask = env_mask.any(dim="days") # valid cells - shape (231, 360)

# # Apply the environmental validity mask to scenario (keep only valid cells for scenarios)
# scenario_mask = scenario_mask.where(env_static_mask) # valid cells for scenarios - shape (231, 360)

# # Count valid environmental cells per scenario (drivers!=Nans) -- should be equal to the sum of the valid cells at each scenarios
# scenario_counts = {}
# total_scenario_cells = 0
# for cat in range(5):
#     count = (scenario_mask == cat).sum().item()
#     scenario_counts[cat] = count
#     total_scenario_cells += count
#     # print(f"Category {cat}: {count} valid environmental cells")

# print(f"\nSum of all scenario-assigned cells: {total_scenario_cells}")
# print(f"Total valid environmental cells:     {env_static_mask.sum().item()}")

# def valid_cells(length_ds, exposure_ds, valid_env_mask, scenario_mask, scenario_value):
#     # Mask length and exposure by:
#     # - Valid environmental mask (valid_env_mask)
#     # - Scenario mask equal to scenario_value (e.g. 0 for non-MHW, 1 for 1deg, etc)
#     scenario_specific_mask = (scenario_mask == scenario_value)
    
#     # Combine masks
#     combined_mask = valid_env_mask & scenario_specific_mask
    
#     # Apply combined mask to length and exposure
#     length_masked = length_ds.where(combined_mask)
#     exposure_masked = exposure_ds.where(combined_mask) if exposure_ds is not None else None
    
#     if exposure_masked is not None:
#         # Number of valid cells per day
#         n_valid_cells_per_day = (~length_masked.isnull()).sum(dim=["eta_rho", "xi_rho"])
#         normalized_exposure = exposure_masked / n_valid_cells_per_day
#         return length_masked, normalized_exposure
#     else:
#         return length_masked, None

# # -- Mean time series -- Spatial average with weight on mhw exposure -- 1 season
# # Non MHWs
# length_masked_nonmhw, exposure_masked_nonmhw = valid_cells(simulated_length_study_area_non_MHWs, mhws_exposure_days_Atl_1yr.nb_days_non_mhw,
#                                                            env_static_mask, scenario_mask, scenario_value=0)
# mean_length_study_area_non_MHWs = ((length_masked_nonmhw * exposure_masked_nonmhw).sum(dim=['eta_rho', 'xi_rho']) / exposure_masked_nonmhw.sum(dim=['eta_rho', 'xi_rho']))

# # MHWs Scenarios - 1 season
# length_masked_1deg, exposure_masked_1deg = valid_cells(simulated_length_study_area_MHWs_1deg, mhws_exposure_days_Atl_1yr.nb_days_1deg,
#                                                            env_static_mask, scenario_mask, scenario_value=1)
# average_length_ts_1deg = ((length_masked_1deg * exposure_masked_1deg).sum(dim=['eta_rho', 'xi_rho']) / exposure_masked_1deg.sum(dim=['eta_rho', 'xi_rho']))

# length_masked_2deg, exposure_masked_2deg = valid_cells(simulated_length_study_area_MHWs_2deg, mhws_exposure_days_Atl_1yr.nb_days_2deg,
#                                                            env_static_mask, scenario_mask, scenario_value=2)
# average_length_ts_2deg = ((length_masked_2deg * exposure_masked_2deg).sum(dim=['eta_rho', 'xi_rho']) / exposure_masked_2deg.sum(dim=['eta_rho', 'xi_rho']))

# length_masked_3deg, exposure_masked_3deg = valid_cells(simulated_length_study_area_MHWs_3deg, mhws_exposure_days_Atl_1yr.nb_days_3deg,
#                                                            env_static_mask, scenario_mask, scenario_value=3)
# average_length_ts_3deg = ((length_masked_3deg * exposure_masked_3deg).sum(dim=['eta_rho', 'xi_rho']) / exposure_masked_3deg.sum(dim=['eta_rho', 'xi_rho']))

# length_masked_4deg, exposure_masked_4deg = valid_cells(simulated_length_study_area_MHWs_4deg, mhws_exposure_days_Atl_1yr.nb_days_4deg,
#                                                            env_static_mask, scenario_mask, scenario_value=4)
# average_length_ts_4deg = ((length_masked_4deg * exposure_masked_4deg).sum(dim=['eta_rho', 'xi_rho']) / exposure_masked_4deg.sum(dim=['eta_rho', 'xi_rho']))


# -- Mean time series for the full period, i.e. diseagrding the mhws scenarios
# length_masked_yr, _ = valid_cells(simulated_length_study_area_1season, None, env_static_mask, None, None) #shape: (181, 231, 360)
# mean_length_study_area_yr = length_masked_yr.mean(dim=["eta_rho", "xi_rho"], skipna=True)
# std_length_mean_length_study_area_yr = length_masked_yr.std(dim=["eta_rho", "xi_rho"], skipna=True)

# Total exposuree days (weights) in 1 season
# w_nonmhw = non_mhw_mask.sum(dim=["eta_rho", "xi_rho"])
# w_1deg = deg1_mask.sum(dim=["eta_rho", "xi_rho"])
# w_2deg = deg2_mask.sum(dim=["eta_rho", "xi_rho"])
# w_3deg = deg3_mask.sum(dim=["eta_rho", "xi_rho"])
# w_4deg = deg4_mask.sum(dim=["eta_rho", "xi_rho"])

# # Weighted mean
# weighted_sum = mean_length_study_area_non_MHWs * w_nonmhw + average_length_ts_1deg * w_1deg + average_length_ts_2deg * w_2deg + average_length_ts_3deg * w_3deg + average_length_ts_4deg * w_4deg
# total_weight = w_nonmhw + w_1deg + w_2deg + w_3deg + w_4deg
# weighted_mean_length_1season = weighted_sum / total_weight
# weighted_std_length_1season = np.sqrt((w_nonmhw * (mean_length_study_area_non_MHWs - weighted_mean_length_1season)**2 +
#                                       w_1deg * (average_length_ts_1deg - weighted_mean_length_1season)**2 +
#                                       w_2deg * (average_length_ts_2deg - weighted_mean_length_1season)**2 +
#                                       w_3deg * (average_length_ts_3deg - weighted_mean_length_1season)**2 +
#                                       w_4deg * (average_length_ts_4deg - weighted_mean_length_1season)**2) / total_weight)

# print(f"Full seasonal mean length (weighted): {weighted_mean_length_1season[-1].values:.2f} mm")
# print(f"Non MHW mean length:                  {mean_length_study_area_non_MHWs[-1].values:.2f} mm")
# print(f"MHW 1°C mean length:                  {average_length_ts_1deg[-1].values:.2f} mm")
# print(f"MHW 2°C mean length:                  {average_length_ts_2deg[-1].values:.2f} mm")
# print(f"MHW 3°C mean length:                  {average_lngth_ts_3deg[-1].values:.2f} mm")
# print(f"MHW 4°C mean length:                  {average_length_ts_4deg[-1].values:.2f} mm")

# %% --- soft assignment

# # Total exposure days per cell across all categories
# # Prepare an empty array to hold the total days per cell according to assigned scenario
# total_days_corrected = xr.full_like(mhws_exposure_days_Atl_1yr.nb_days_non_mhw, fill_value=0)

# # Assign days based on scenario_mask
# total_days_corrected = total_days_corrected.where(scenario_mask != 0, mhws_exposure_days_Atl_1yr.nb_days_non_mhw)
# total_days_corrected = total_days_corrected.where(scenario_mask != 1, mhws_exposure_days_Atl_1yr.nb_days_1deg)
# total_days_corrected = total_days_corrected.where(scenario_mask != 2, mhws_exposure_days_Atl_1yr.nb_days_2deg)
# total_days_corrected = total_days_corrected.where(scenario_mask != 3, mhws_exposure_days_Atl_1yr.nb_days_3deg)
# total_days_corrected = total_days_corrected.where(scenario_mask != 4, mhws_exposure_days_Atl_1yr.nb_days_4deg)
# # print(f"Max number of days under MHWs: {total_days_corrected.max().values}") #181

# # Normalize -- fractions per cell
# frac_non_mhw = mhws_exposure_days_Atl_1yr.nb_days_non_mhw / total_days_corrected
# frac_1deg = mhws_exposure_days_Atl_1yr.nb_days_1deg / total_days_corrected
# frac_2deg = mhws_exposure_days_Atl_1yr.nb_days_2deg / total_days_corrected
# frac_3deg = mhws_exposure_days_Atl_1yr.nb_days_3deg / total_days_corrected
# frac_4deg = mhws_exposure_days_Atl_1yr.nb_days_4deg / total_days_corrected

# print(f"Max fraction for cells under 4°C MHWs: {frac_4deg.max().values}") #0.24
# print(f"Max fraction for cells under 1°C MHWs: {frac_1deg.max().values}") #1.0

# def weighted_mean_length(length_ds, frac_weights):
#     # Multiply length by fractional weights, sum spatially, divide by sum of weights
#     numerator = (length_ds * frac_weights).sum(dim=["eta_rho", "xi_rho"])
#     denominator = frac_weights.sum(dim=["eta_rho", "xi_rho"])
#     return numerator / denominator

# mean_length_nonmhw = weighted_mean_length(simulated_length_study_area_1season, frac_non_mhw)
# mean_length_1deg = weighted_mean_length(simulated_length_study_area_1season, frac_1deg)
# mean_length_2deg = weighted_mean_length(simulated_length_study_area_1season, frac_2deg)
# mean_length_3deg = weighted_mean_length(simulated_length_study_area_1season, frac_3deg)
# mean_length_4deg = weighted_mean_length(simulated_length_study_area_1season, frac_4deg)

# # Total weighted mean (all scenarios combined) should match full spatial mean
# total_weighted_mean = (
#     mean_length_nonmhw * frac_non_mhw.sum(["eta_rho", "xi_rho"]) +
#     mean_length_1deg * frac_1deg.sum(["eta_rho", "xi_rho"]) +
#     mean_length_2deg * frac_2deg.sum(["eta_rho", "xi_rho"]) +
#     mean_length_3deg * frac_3deg.sum(["eta_rho", "xi_rho"]) +
#     mean_length_4deg * frac_4deg.sum(["eta_rho", "xi_rho"])
# ) / total_days_corrected.sum(["eta_rho", "xi_rho"])

# print(f"Full spatial mean length: {simulated_length_study_area_1season.mean(['eta_rho', 'xi_rho'])[-1].values:.2f} mm")


# %% ============== Plotting length over 1 season ==============
# Define colors and labels
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']

# Deal with day to have continuous x-axis
days_xaxis = np.where(simulated_length_study_area_1980_2019.days.values < 304, 
                      simulated_length_study_area_1980_2019.days.values + 365, 
                      simulated_length_study_area_1980_2019.days.values).astype(int)
from datetime import datetime, timedelta
base_date = datetime(2021, 11, 1)  # Nov 1 (season start)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
days_xaxis = np.arange(181)  # Just 0 to 180 (1 per day)

plot = 'report'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611  # text width in inches
    fig_height = fig_width / 2
else:  # 'slides'
    fig_width = 15
    fig_height = 6

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 13}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 10}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}

# --- Length time series ---
lw=1.5 if 'report' else 2
ax.plot(days_xaxis, average_length_ts_1deg, color='#5A7854', linewidth=lw, label='1°C and 90th perc')
ax.plot(days_xaxis, average_length_ts_2deg, color='#8780C6', linewidth=lw, label='2°C and 90th perc') 
ax.plot(days_xaxis, average_length_ts_3deg, color='#E07800', linewidth=lw, label='3°C and 90th perc') 
ax.plot(days_xaxis, average_length_ts_4deg, color='#9B2808', linewidth=lw, label='4°C and 90th perc')
ax.plot(days_xaxis, mean_length_study_area_non_MHWs, label=f"Non-MHWs", color="black", linestyle='-', linewidth=lw)

# mean_values_1980_2018 = mean_length_study_area_1980_2019.values
# std_values = std_length_mean_length_study_area_1980_2019.values
# mean_values_season = weighted_mean_length_1season.values
# std_values_season  = weighted_std_length_1season.values
# ax.plot(days_xaxis, mean_values_season, label="Mean (1980-2018)", color="grey", linestyle='--', linewidth=lw)
# ax.fill_between(days_xaxis,
#                 mean_values_season - std_values_season,
#                 mean_values_season + std_values_season,
#                 color="gray", alpha=0.2, label="±1 $\sigma _{1980-2018}$")

# Define labels to keep
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []
for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)
ax.set_xticks(tick_positions)
ax.tick_params(axis='both', **tick_kwargs)

# ax.set_xticks(tick_positions, **tick_kwargs)
ax.set_xticklabels(tick_labels, rotation=45)
ax.set_xlabel("Date", **label_kwargs)
ax.set_ylabel("Length (mm)", **label_kwargs)
# ax.set_ylim(32,37.5)
# ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', **tick_kwargs)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
          ncol=3, frameon=True, **legend_kwargs)

if plot == 'report':
    suptitle_y = 1.07
else:  # slides
    suptitle_y = 1.01
    
# --- Title and subtitle ---
fig.suptitle(
    "Evolution of Krill Length \nunder MHW and Non-MHW conditions in the Atlantic Sector",
    **maintitle_kwargs,
    y=suptitle_y
)

ax.set_title(f"Growth Season {selected_years[yr_chosen]}–{selected_years[yr_chosen]+1}", **subtitle_kwargs)


# --- Output handling ---    
if plot == 'report':    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/length_mhw_{selected_years[yr_chosen]}_{plot}.pdf'), dpi =200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/length_mhw_{selected_years[yr_chosen]}_{plot}.png'), dpi =500, format='png', bbox_inches='tight')
    plt.show()

# %% ============== Plot evolution of the area ==============
from datetime import datetime, timedelta

# Create days axis: 0 to 180
days_xaxis = np.arange(181)
base_date = datetime(2021, 11, 1)  # Start from Nov 1
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)

# Set up color map from red (1980) to green (2019)
colors = [(0, "#278E2F"), (0.6, "#EE9B00"), (1, "#AE2012")]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreenYellowRed", colors)

norm = mpl.colors.Normalize(vmin=1980, vmax=2019)
sm = mpl.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

plot = 'slides'  # 'report' or 'slides'

if plot == 'report':
    fig_width = 6.3228348611  # text width in inches
    fig_height = fig_width/2
else:  # 'slides'
    fig_width = 10
    fig_height = 5

fig, ax_len = plt.subplots(figsize=(fig_width, fig_height))

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 13}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 10}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}


# --- Loop over years ---
for yr in range(38):
    annual_length = length_Atkison2006(chla=chla_surf_study_area_allyrs.chla.isel(years=yr),
                                       temp=temp_avg_100m_study_area_allyrs.avg_temp.isel(years=yr),
                                       initial_length= 35, intermoult_period=10)
    annual_length_mean = annual_length.mean(dim=('eta_rho', 'xi_rho'))
    ax_len.plot(days_xaxis, annual_length_mean, color=custom_cmap(norm(yr+1980)), linewidth=1)

# Plot mean 
ax_len.plot(days_xaxis, mean_length_study_area_1980_2019.values, linestyle='--', color='black',
            label="Mean 1980–2018")
    
# Format x-axis ticks
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions, tick_labels = zip(*[(day, label) for day, label in date_dict.items() if label in wanted_labels])
ax_len.set_xticks(tick_positions)
ax_len.set_xticklabels(tick_labels, rotation=45, fontsize=tick_kwargs.get('labelsize', None))

# Labels and title
ax_len.set_xlabel("Date", **label_kwargs)
ax_len.set_ylabel("Length (mm)", **label_kwargs)
ax_len.tick_params(axis='both', **tick_kwargs)
ax_len.legend(loc='upper left', handlelength=2.5, **legend_kwargs)

fig.subplots_adjust(top=0.88 if plot == 'slides' else 0.90)

suptitle_y = 1.05 if plot == 'slides' else 0.99
fig.suptitle("Interannual Variability of Krill Length", y=suptitle_y, **maintitle_kwargs)

subtitle_y = 0.95 if plot == 'slides' else 0.89
fig.text(0.5, subtitle_y, "Spatial average over the Atlantic Sector (0-90°W) during growth season", ha='center', **subtitle_kwargs)


# Colorbar
cbar = fig.colorbar(sm, ax=ax_len, orientation='vertical', label='Year')
cbar.set_ticks(np.linspace(1980, 2018, 5).astype(int))
cbar.ax.tick_params(**tick_kwargs)


# --- Output handling ---    
plt.tight_layout()
if plot == 'report':    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/length_all_years_atl_sector_{plot}.pdf'), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/length_all_years_atl_sector_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()
    

# %% Likelihood

mean_length = mean_length_study_area_1980_2019  # shape: (181,)

all_lengths = []
for yr in range(38):
    annual_length = length_Atkison2006(
        chla=chla_surf_study_area_allyrs.chla.isel(years=yr),
        temp=temp_avg_100m_study_area_allyrs.avg_temp.isel(years=yr),
        initial_length=35, intermoult_period=10
    )
    annual_length_mean = annual_length.mean(dim=('eta_rho', 'xi_rho'))  # shape: (181,)
    all_lengths.append(annual_length_mean.values)

all_lengths = np.stack(all_lengths)  # shape: (38, 181)


probability_above_mean = (all_lengths > mean_length.values).sum(axis=0) / all_lengths.shape[0]
overall_likelihood = (all_lengths.mean(axis=1) > mean_length.values.mean()).mean()

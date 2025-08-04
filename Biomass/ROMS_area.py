"""
Created on Tue 30 July 15:30:36 2025

ROMS - area calculation

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
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
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





# %% ================================= Area of ROMS grid cells =================================
# --- Load data
ROMS_area = xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc')

# --- 1. Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = ROMS_area['area'].isel(z_t=0)

# Mask latitudes south of 60°S (lat_rho <= -60)
area_SO = area_SO_surf.where(ROMS_area['lat_rho'] <= -60, drop=True)

# Sum grid cell areas
total_area_SO_km2 = area_SO.sum().item()
print(f'Total Southern Ocean area south of 60°S: {total_area_SO_km2:.2f} km²')

# --- 2. Atlantic Sector
def subset_atlantic_sector(ds, lat_range=(-80, -60), lon_range=(270, 360)):
    """
    Subset dataset ds to given latitude and longitude ranges.
    lon_range: tuple with (min_lon, max_lon) in degrees [0, 360]
    """
    lat_mask = (ds['lat_rho'] >= lat_range[0]) & (ds['lat_rho'] <= lat_range[1])
    lon_mask = (ds['lon_rho'] >= lon_range[0]) & (ds['lon_rho'] <= lon_range[1])

    combined_mask = lat_mask & lon_mask
    return ds.where(combined_mask, drop=True)


# Apply subset
area_Atl_Sect = subset_atlantic_sector(ROMS_area)

# Select surface layer
area_Atl_surf = area_Atl_Sect['area'].isel(z_t=0)

# Sum area
total_area_Atl_km2 = area_Atl_surf.sum().item()
print(f'Total Atlantic Sector area south of 60°S: {total_area_Atl_km2:.2f} km²')


# %% ============================================ Areas Values  ============================================
# ---- Plot ----
# --- Output format ---
plot = 'report'  # report 'slides'

# --- Figure dimensions ---
fig_width = 6.32 if plot == 'report' else 10
fig_height = 5.5 if plot == 'report' else 8

# --- Font and style kwargs ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 13}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 10}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}

# --- Setup figure and axes ---
fig = plt.figure(figsize=(fig_width, fig_height))
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# --- Map extent and features ---
ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
ax.coastlines(color='black', linewidth=1)
ax.set_facecolor("#D8D8D8")

# Sector lines
lw = 1 if plot == 'report' else 2
for lon_line in [-90, 0, 120]:
    ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
            color="#080808", linestyle='--', linewidth=lw, zorder=5)

# --- Gridlines with labels ---
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7, zorder=3)
# gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Plot data
im = ax.pcolormesh(area_SO.lon_rho, area_SO.lat_rho, area_SO,
                    transform=ccrs.PlateCarree(), cmap='Blues',
                    shading='auto', zorder=1, rasterized=True)


# --- Colorbar ---
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, pad=0.05, extend='both')
cbar.ax.minorticks_off()
cbar.set_label('Area [km²]', **label_kwargs)

# --- Highlight Atlantic sector ---
lat, lat60 = np.linspace(-90, -60, 200), -60
ax.plot([0]*200, lat, color='red', transform=ccrs.PlateCarree(), lw=lw, zorder=6)         # 0°E
ax.plot([270]*200, lat, color='red', transform=ccrs.PlateCarree(), lw=lw, zorder=6)      # 90°W
ax.plot(np.linspace(0, -90, 200), [lat60]*200, color='red', transform=ccrs.PlateCarree(), lw=lw, zorder=6)  # -60°S arc

# --- Title ---
var_bigtitle = 'Area for each ROMS cell'
suptitle_y = 1.02 if plot == 'slides' else 1.01
# fig.suptitle(var_bigtitle, y=suptitle_y, x=0.5, ha='center', **maintitle_kwargs)

plt.tight_layout()

# --- Output handling ---
outdir = os.path.join(os.getcwd(), 'Biomass/figures_outputs/')
os.makedirs(outdir, exist_ok=True)

if plot == 'report':
    outfile = "area_ROMS_report.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    outfile = "area_ROMS_slides.png"
    # plt.savefig(os.path.join(outdir, outfile), dpi=500, format='png', bbox_inches='tight')
    plt.show()
# %%

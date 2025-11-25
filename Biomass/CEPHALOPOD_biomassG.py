"""
Created on Tue 25 Nov 10:51:30 2025

CEPHALOPOD model outputs for biomass

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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


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
# --- Biomass
biomass_data = xr.open_dataset('/net/meso/work/aschickele/CEPHALOPOD/output/Marguerite_krill_SO_2025-11-21 16:44:03.790757/236217/euphausia_biomass.nc') #shape (bootstrap=10, northing=64800, time=12)

# --- Reformatting
nlat, nlon = 180, 360
lat = np.linspace(-89.5, 89.5, nlat)
lon = np.linspace(-179.5, 179.5, nlon)

# Raw data
arr = biomass_data.euphausia_biomass.values  # shape: (bootstrap=10, northing=64800, time=12)

# Transpose to (months, bootstrap, northing)
arr = arr.transpose(2, 0, 1)  # (12, 10, 64800)

# Reshape northing into (lat, lon)
arr = arr.reshape(12, 10, 180, 360)  # (months, bootstrap, lat, lon)

# Flip latitude (before it was going from -90 to 90, i.e. from south pole to north pole)
arr_global = arr[:, :, ::-1, :]  # flip along lat axis
lat_flipped = lat[::-1]

# Create new Dataset
biomass_cephalopod = xr.Dataset(
    data_vars=dict(euphausia_biomass=(["months", "bootstrap", "lat", "lon"], arr_global)), 
    coords=dict(months=np.arange(1, 13),
                bootstrap=np.arange(1, 11),
                lat=lat_flipped, lon=lon),
    attrs=biomass_data.attrs)

# Add info in attributed
biomass_cephalopod.attrs.update({
    "model_name": "Cephalopod",
    "model_resolution": "1 degree",
    "model_extent": "global",
    "model_inputs": "WOA",
    "ensemble_member_selection": "RF and SVM",
    "note": "Under assumption that krill spend most of their time in the 0-100m, all observations are integrated (median concentration on depth)."
})

# -- Select only the Southern Ocean (south of 60°S)
arr = biomass_data.euphausia_biomass.values.transpose(2, 0, 1).reshape(12, 10, 180, 360)
lat = np.linspace(89.5, -89.5, 180)  # north → south

lat_mask = lat <= -60
arr_60S = arr[:, :, lat_mask, :]
lat_60S = lat[lat_mask]  # lat_60S: -60.5 → -89.5 (north → south)

arr_60S_flipped = arr_60S[:, :, ::-1, :]
lat_60S_flipped = lat_60S[::-1]  # now first row = south pole

biomass_cephalopod_60S = xr.Dataset(
    data_vars=dict(euphausia_biomass=(["months", "bootstrap", "lat", "lon"], arr_60S_flipped)),
    coords=dict(
        months=np.arange(1, 13),
        bootstrap=np.arange(1, 11),
        lat=lat_60S_flipped,
        lon=np.linspace(-179.5, 179.5, 360)
    )
)

# %% Plot test
ds_1d = biomass_cephalopod_60S.euphausia_biomass.isel(months=10, bootstrap=6)
data = ds_1d.values
lat = ds_1d.lat.values
lon = ds_1d.lon.values

# Meshgrid
lon2d, lat2d = np.meshgrid(lon, lat)

# Colorbar (using quantile)
vmin, vmax = np.nanquantile(data, [0.05, 0.95])  # 5th and 95th percentiles

# Figure
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Plot data
pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(),
                    cmap='coolwarm', vmin= vmin, vmax = vmax, shading='auto')
cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
cbar.set_label('Krill biomass')

# Gridlines
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.7, linestyle='--', linewidth=0.4, zorder=7)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'rotation': 0}
gl.ylabel_style = {'rotation': 0}

ax.set_title(f"Euphausia superba biomass\nMonth: {int(ds_1d.months.values)}, Bootstrap: {int(ds_1d.bootstrap.values)}", fontsize=14)

plt.tight_layout()
plt.show()


# %% ====================== Initial Biomass ======================
# Biomass in November = Initial Biomass (growth season)
biomass_nov = biomass_cephalopod_60S.euphausia_biomass.isel(months=10)

# Mean over all bootstraps
mean_biomass_nov = biomass_nov.mean(dim='bootstrap')
std_biomass_nov = biomass_nov.std(dim='bootstrap')
# %%

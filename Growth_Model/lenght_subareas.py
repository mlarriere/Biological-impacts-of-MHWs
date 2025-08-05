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

# # %% ====================== Defining subareas ======================
# import geopandas as gpd
# # Areas defined using geojson.io

# box_peninsula = {
#   "type": "FeatureCollection",
#   "features": [
#     {
#       "type": "Feature",
#       "properties": {},
#       "geometry": {
#         "coordinates": [
#           [
#              [
#               -60,
#               -64
#             ],
#             [
#               -61,
#               -63
#             ],
            
#             [
#               -65,
#               -63
#             ],
#             [
#               -69,
#               -65
#             ],
#             [
#               -73.5,
#               -67.5
#             ],
#             [
#               -76,
#               -69
#             ],
#             [
#               -76,
#               -70
#             ],
#             [
#               -73.5,
#               -71
#             ],
#             [
#               -69,
#               -70
#             ],
#             [
#               -60.5,
#               -64
#             ]
#           ]
#         ],
#         "type": "Polygon"
#       }
#     }
#   ]
# }

# gdf_peninsula = gpd.GeoDataFrame.from_features(box_peninsula["features"])
# gdf_peninsula.crs = "EPSG:4326"

# box_north = {
#   "type": "FeatureCollection",
#   "features": [
#     {
#       "type": "Feature",
#       "properties": {},
#       "geometry": {
#         "coordinates": [
#           [
#             [
#               -70,
#               -60
#             ],
#             [
#               -75,
#               -60
#             ],
#             [
#               -80,
#               -60
#             ],
#             [
#               -90,
#               -60
#             ],
#             [
#               -90,
#               -63
#             ],
#             [
#               -80,
#               -63
#             ],
#             [
#               -75,
#               -63
#             ],
#             [
#               -70,
#               -63
#             ],
#             [
#               -70,
#               -60
#             ]
#           ]
#         ],
#         "type": "Polygon"
#       }
#     }
#   ]
# }
# gdf_north = gpd.GeoDataFrame.from_features(box_north["features"])
# gdf_north.crs = "EPSG:4326"


# # %% ======================== Plot subareas ========================
# # MHW durations
# mhw_duration_5m = xr.open_dataset(os.path.join(path_duration, "mhw_duration_5m.nc")).mhw_durations #dataset - shape (40, 365, 434, 1442)
# det_combined_ds = xr.open_dataset(os.path.join(path_combined_thesh, 'det_depth5m.nc')) #boolean shape (40, 181, 434, 1442)
# mhw_duration_seasonal = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc')))

# # Load data
# growth_study_area = xr.open_dataset(os.path.join(path_growth, 'atlantic_sector/growth_study_area.nc'))
# mhw_duration_study_area = xr.open_dataset(os.path.join(path_growth, 'atlantic_sector/mhw_duration_study_area.nc'))


# lon = mhw_duration_study_area.lon_rho.values     # shape: (eta_rho, xi_rho)
# lat = mhw_duration_study_area.lat_rho.values     # shape: (eta_rho, xi_rho)
# data = mhw_duration_study_area.duration.isel(years=36, days=10).values

# # 5. === Setup plot ===
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})

# theta = np.linspace(np.pi / 2, np.pi, 100)
# center, radius = [0.5, 0.51], 0.5
# arc = np.vstack([np.cos(theta), np.sin(theta)]).T
# verts = np.concatenate([[center], arc * radius + center, [center]])
# circle = mpath.Path(verts)

# ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
# ax.set_boundary(circle, transform=ax.transAxes)

# # 6. === Plot duration background ===
# cmap = 'plasma'  # or any preferred colormap
# im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
#                    cmap=cmap, shading='auto', zorder=1, vmin=0, vmax=30)

# # 7. === Map features ===
# ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
# ax.coastlines(color='black', linewidth=0.5, zorder=3)

# # 8. === Polygon overlay ===
# gdf_peninsula.plot(ax=ax, facecolor='none', edgecolor='orange', linewidth=2,
#          transform=ccrs.PlateCarree(), zorder=4)
# gdf_north.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2,
#          transform=ccrs.PlateCarree(), zorder=4)

# # 9. === Colorbar ===
# cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
# cbar.set_label('Mean MHW Duration (days)', fontsize=10)


# plt.tight_layout()
# plt.show()


# %% ====================== Drivers in the sub areas ======================
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (39, 181, 231, 1442)

def subset_spatial_domain(ds, lat_range=(-80, -60), lon_range=(270, 360)): #, (0, 30)
    lat_min, lat_max = lat_range
    lon_range1, lon_range2 = lon_range

    lat_mask = (ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max)
    lon_mask = ((ds['lon_rho'] >= lon_range1) & (ds['lon_rho'] <= lon_range2)) #| ((ds['lon_rho'] >= lon_range2[0]) & (ds['lon_rho'] <= lon_range2[1]))

    return ds.where(lat_mask & lon_mask, drop=True)

temp_north = subset_spatial_domain(temp_avg_100m_SO_allyrs, 
                                       lat_range=(-63, -60),
                                       lon_range=(270,290))



# Subset broad rectangle first
temp_peninsula_box = subset_spatial_domain(temp_avg_100m_SO_allyrs,
                                           lat_range=(-90, -60),
                                           lon_range=(270, 360))

from shapely.geometry import Polygon, Point

def tilted_rectangle_mask(ds, point1, point2, point3, point4):
    """
    Applies a polygonal (tilted rectangle) mask using 4 corner points (lon, lat).
    
    Parameters:
    - ds: xarray.Dataset with lat_rho and lon_rho
    - point1 → point2 → point3 → point4 → point1: corners of the polygon in (lon, lat)
    
    Returns:
    - Masked xarray.Dataset within polygon
    """
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

temp_peninsula_tilted = tilted_rectangle_mask(
    ds=temp_peninsula_box,
    point1=(360-68, -64),   # SW corner (more west/south)
    point2=(360-64, -65),   # SE corner
    point3=(360-54, -62),   # NE corner (more east/north)
    point4=(360-61, -61)    # NW corner
)


# %%  ======================== PLot area ========================
# Year and day slice
year_idx = 36  # 2015
day_idx = 10

# Extract slices
north_data = temp_north.avg_temp.isel(years=year_idx, days=day_idx)
peninsula_data = temp_peninsula_tilted.avg_temp.isel(years=year_idx, days=day_idx)

# Define circular boundary path for polar view
def circular_boundary():
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.cos(theta), np.sin(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return circle

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})

# Shared plotting settings
def plot_region(ax, data, title):
    im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                       transform=ccrs.PlateCarree(),
                       cmap='coolwarm', vmin=-2, vmax=4)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=100)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circular_boundary(), transform=ax.transAxes)
    ax.set_title(title)
    # Sector boundaries
    for lon in [-90, 120, 0]:
            ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(), color='#080808', linestyle='--', linewidth=0.5)

    return im

    
# Plot both regions
im1 = plot_region(axes[0], north_data, 'North Region (2015, Day 10)')
im2 = plot_region(axes[1], peninsula_data, 'Peninsula Region (2015, Day 10)')

# Colorbar
cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.05)
cbar.set_label('Avg Temperature at 100m (°C)')

plt.tight_layout()
plt.show()


# %%

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Extract data
peninsula_data = temp_peninsula_tilted.avg_temp.isel(years=year_idx, days=day_idx)

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6),
                       subplot_kw={'projection': ccrs.PlateCarree()})

# Plot parameters
plot_kwargs = dict(cmap='coolwarm', vmin=-2, vmax=4)

# Draw temperature field
im = ax.pcolormesh(peninsula_data.lon_rho,
                   peninsula_data.lat_rho,
                   peninsula_data,
                   transform=ccrs.PlateCarree(),
                   **plot_kwargs)

# Add map features
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k', facecolor='lightgrey')

# Zoom in on the tilted rectangle region (adjust as needed)
ax.set_extent([270, 320, -80, -60], crs=ccrs.PlateCarree())
# Add gridlines
gl = ax.gridlines(draw_labels=True,
                  linewidth=0.5,
                  color='gray',
                  alpha=0.7,
                  linestyle='--')

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Title and colorbar
ax.set_title("Tilted Peninsula Region", fontsize=14)
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
cbar.set_label("Temperature (°C)")

plt.tight_layout()
plt.show()

# %% ==================== Mean Temperature =====================
selected_years = [1989, 2000, 2016]
selected_years_idx = np.array(selected_years) - 1980  # [9, 20, 36]

temp_peninsula_tilted_avg = temp_peninsula_tilted.isel(years=selected_years_idx[0]).mean(dim=['eta_rho', 'xi_rho']) #shape (181,)
temp_north_avg = temp_north.isel(years=selected_years_idx[0]).mean(dim=['eta_rho', 'xi_rho']) #shape (181,)

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

days = temp_north_avg.days
days_xaxis = np.where(days < 304, days + 365, days).astype(int)

# Base year and date mapping (from your code)
base_year = 2021  # non-leap year
doy_list = list(range(304, 365)) + list(range(0 + 365, 121 + 365))  # 181 days total
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

# Plot
plt.figure(figsize=(12, 5))
temp = temp_peninsula_tilted_avg.avg_temp

plt.plot(days_xaxis, temp, color='darkred', linewidth=2)

# Set ticks: choose ticks every ~15 days for clarity
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15)
tick_labels = [date_dict.get(day, '') for day in tick_positions]

plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Mean Temperature Time Series (Tilted Peninsula Region, 1989)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# %%

"""
Created on Mon 17 Feb 09:32:05 2025

Defining climatology according to the methodology of Hobday et al. (2016)

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import sys
import os
import warnings
import copy as cp
import gsw
import xarray as xr
import numpy as np
import seaborn as sns

import pandas as pd
import geopandas as gpd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import datetime
import cftime

import dask
from dask.distributed import Client
from joblib import Parallel, delayed

# %% Settings
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


# -- Define Climatology - baseline type
# baseline = 'fixed1980' 
baseline = 'fixed30yrs' 

if baseline=='fixed1980':
    description = "Detected events" + f'T°C > T°C 1980' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline1980/'
if baseline=='fixed30yrs':
    description = "Detected events" + f'T°C > climatology (1980-2010)' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold
percentile = 90 
pmod = 'perc' + str(percentile)


# -- Handling time
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) #defining months with days within a year
month_names = np.array(['Jan','Feb','Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov','Dec'])

season_bins = np.array([0, 90, 181, 273, 365]) #defining seasons with days within a year
season_names = np.array(['DJF (Summer)', 'MAM (Fall)', 'JJA (Winter)', 'SON (Spring)']) #southern ocean!

# %%# %% ---- Climatology for each eta
def calculate_climSST(ieta, baseline):

    print(f"Processing eta {ieta}...")

    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:31,0:365,:,:] #Extracts daily data : only 30yr + consider 365 days per year. shape:(30, 365, 35, 1442)
    print(np.unique(ds_original.lat_rho.values))
    # ds_original.values[np.isnan(ds_original.values)] = 0 # Set to 0 so that det = False at nans
    
    # Calculate SST climatology and threshold
    ds_avg30yrs = ds_original.mean(dim='year') # 30-year mean 

    # Initialization arrays - shape: (365, 35, 1442)
    nyears, ndays, nz, nxi = ds_original.shape
    relative_threshold = np.full((ndays, nz, nxi),np.nan,dtype=np.float32)
    climatology30yrs = np.full((ndays, nz, nxi),np.nan,dtype=np.float32)

    # Deal with NANs values
    mask_nanvalues = np.where(np.isnan(ds_avg30yrs.values), False, True)  # Mask: True where valid, False where NaN

    if baseline == 'fixed1980':
        climatology_sst = ds_original[0, :, :, :]  # Only 1980
    elif baseline == 'fixed30yrs':
        # Moving window of 11days - method from Hobday et al. (2016)
        for dy in range(0,ndays-1): #days index going from 0 to 364
            if dy<=4:
                window11d_sst = ds_avg30yrs.isel(day=np.concatenate([np.arange(360+dy,365,1), np.arange(0, dy+6,1)]))
            elif dy>=360:
                window11d_sst = ds_avg30yrs.isel(day=np.concatenate([np.arange(dy-5, 365,1), np.arange(0,dy-359,1)]))
            else:
                window11d_sst = ds_avg30yrs.isel(day=np.arange(dy-5, dy+6, 1))

            relative_threshold[dy,:, :] = np.percentile(window11d_sst, 90, axis=(0,1)) # 90th percentile
            climatology30yrs[dy,:, :]  = np.nanmedian(window11d_sst, axis=(0,1)) # median -- compute the median along the specified axis, while ignoring NaNs)
    
    # Apply mask
    relative_threshold = np.where(mask_nanvalues, relative_threshold, np.nan) 
    climatology30yrs = np.where(mask_nanvalues, climatology30yrs, np.nan)

    # Write to dataset
    dict_vars = {}
    dict_vars['relative_threshold'] = (["day","nz", "xi_rho"], relative_threshold)
    dict_vars['climatology'] = (["day","nz", "xi_rho"], climatology30yrs)

    ds_vars = xr.Dataset(
        data_vars= dict(dict_vars),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
            ),
        attrs = {
            'relative_threshold': 'Daily climatological relative threshold (90th percentile) - computed using a seasonally varying 11‐day moving window ',
            'climatology': 'Daily climatology SST - median value obtained from a seasonally varying 11‐day moving window - baseline 1980-2009 (30yrs)'
            }
        ) 

    # Save output
    output_file_clim = os.path.join(output_path_clim, f"clim_thresh_{ieta}.nc")
    if not os.path.exists(output_file_clim):
        ds_vars.to_netcdf(output_file_clim, mode='w')  
    
    return ds_vars

# Calling function
results = Parallel(n_jobs=30)(delayed(calculate_climSST)(ieta, baseline) for ieta in range(0, neta)) # calculate climatology for each latitude in parallel - results (list) ~12min computing

# ---> error to fix: <ipython-input-3-bbb4c30461f8>:36: RuntimeWarning: All-NaN slice encountered


# %% Merging all eta in same dataset
# Datasets initialization 
clim_sst = np.full((ndays, nz, neta, nxi), False, dtype=np.float32) #dim (365, 35, 434, 1442)
relative_threshold =  np.full((ndays, nz, neta, nxi), False, dtype=np.float32) #dim (365, 35, 434, 1442)

# Loop over neta and write all eta in same Dataset - aggregation 
for ieta in range(0, neta):
    clim_sst[:, :, ieta, :] = results[ieta].climatology  
    relative_threshold[:, :, ieta, :] = results[ieta].relative_threshold  

# Reformating
ds_clim_sst = xr.Dataset(
    data_vars=dict(clim_sst = (["days", "z_rho", "eta_rho", "xi_rho"], clim_sst)),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Daily climatology SST - median value obtained from a seasonally varying 11‐day moving window - baseline 1980-2009 (30yrs)'),
        ) 

ds_rel_threshold = xr.Dataset(
    data_vars=dict(relative_threshold = (["days", "z_rho", "eta_rho", "xi_rho"], relative_threshold)),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Daily climatological relative threshold (90th percentile) - computed using a seasonally varying 11‐day moving window '),
        ) 

# Save outputs
output_file_clim_sst = os.path.join(output_path_clim, f"climSST_all_eta.nc")
output_file_rel_threshold = os.path.join(output_path_clim, f"rel_threshold_all_eta.nc")

if not os.path.exists(output_file_clim_sst):
    ds_clim_sst.to_netcdf(output_file_clim_sst, mode='w') 

if not os.path.exists(output_file_rel_threshold):
    ds_rel_threshold.to_netcdf(output_file_rel_threshold, mode='w') 

#%% ---------------------------------------------------------- PLOTS (slide) ----------------------------------------------------------
# Settings and data

choice_eta = 200
choice_xi = 1000

# Surface temperature for 1 location (eta, xi)
temp_surf = xr.open_dataset(path_mhw + file_var + 'eta' + str(choice_eta) + '.nc')[var][1:31, 0:365, 0, :]  # 30yrs - 365days per year
selected_temp_surf = temp_surf.sel(xi_rho=choice_xi)

# Climatology computed above
clim_sst_surf = xr.open_dataset(output_path_clim +  'climSST_all_eta.nc' )['clim_sst'][:,0,:,:]
lon = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi).lon_rho.item()
lat = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi).lat_rho.item()
value = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi, days=230).item()

#%% --- Linear SST time serie for 30yrs
fig, ax = plt.subplots(figsize=(10, 5))
cmap = cm.OrRd
norm = mcolors.Normalize(vmin=0, vmax=30)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

for i in range(30):
    ax.plot(selected_temp_surf['day'], selected_temp_surf.isel(year=i), color= cmap(norm(i)))

ax.set_xlabel('Day of the Year')
ax.set_ylabel('SST (°C)')
ax.set_title(f'Sea Surface Temperature (from 1980 to 2009) \nLocation: ({round(lat)}°S, {round(lon)}°E)')
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.99)
cbar.set_label('Year ', fontsize=12)
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.show()

#%% --- Circular time serie
theta = 2 * np.pi * (np.arange(0, 365) / 365)  # Convert days to angles (0 to 2π)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
cmap = cm.OrRd
norm = mcolors.Normalize(vmin=0, vmax=30)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
for i in range(30):
    temp = selected_temp_surf.isel(year=i).values  # (365,)
    ax.plot(theta, temp, color=cmap(norm(i)), linewidth=1.5, alpha=0.8)

# Mean 
ds_avg30yrs = selected_temp_surf.mean(dim='year')
ax.plot(theta, ds_avg30yrs, color='black', linewidth=3, alpha=0.8, label='Mean')

ax.set_ylim(np.nanmin(selected_temp_surf), np.nanmax(selected_temp_surf))
month_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(month_angles)
ax.set_xticklabels(month_labels, fontsize=10)

ax.set_ylabel('SST (°C)', fontsize=15, labelpad=20)
ax.set_title(f'Sea Surface Temperature (from 1980 to 2009)\nLocation: ({round(lat)}°S, {round(lon)}°E)', va='bottom', fontsize=14)
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, shrink=0.9)
cbar.set_label('Year', fontsize=12)
cbar.ax.tick_params(labelsize=10)
plt.legend()
plt.tight_layout()
plt.show()

# %% --- Moving window explanation
# Day to center the window
center_day = 98
window_start = center_day - 5
window_end = center_day + 5

ds_avg30yrs = selected_temp_surf.mean(dim='year')
selected_window = ds_avg30yrs.sel(day=slice(window_start, window_end))

# Example fixed median and threshold for illustration (just to plot)
example_median = float(selected_window.median().values) 
example_threshold =  np.percentile(selected_window.values, 90) 

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ds_avg30yrs.day, ds_avg30yrs, color='black', linewidth=1.5, label='Mean SST (1980–2009)') # Mean daily SST line

# Shaded window (±5 days around center day)
ax.axvspan(window_start, window_end, color='#98C4D7', alpha=0.3, label='11-day window')

# Horizontal lines representing "median" and "90th percentile threshold"
ax.hlines(example_median, window_start, window_end, color='#3F37C9', linestyle='-', linewidth=1.5, label='Median climatology')
ax.hlines(example_threshold, window_start, window_end, color='#D00000', linestyle='-', linewidth=1.5, label='90th percentile threshold')
ax.fill_between(ds_avg30yrs.day.values, ds_avg30yrs.values, example_threshold,
                where=(ds_avg30yrs.values >= example_threshold) &  # Fill above the threshold
                     (ds_avg30yrs.day.values >= window_start) &
                     (ds_avg30yrs.day.values <= window_end),
                color='#D00000', alpha=0.3, label='SST above 90th percentile')

ax.set_xlabel('Day of the Year')
ax.set_ylabel('SST (°C)')
ax.set_title(f'Illustration of 11days moving window calculation\nLocation: ({round(lat)}°S, {round(lon)}°E)')
ax.set_xlim(center_day - 20, center_day +20)
ax.set_ylim(2.2, 3)

# Set x-ticks at desired locations and ensure center_day is included
tick_positions = [tick for tick in ax.get_xticks() if center_day - 20 <= tick <= center_day + 20]
tick_labels = [f"{int(tick)}" for tick in tick_positions]

# Highlight the three specific days: (i-5), (i), (i+5)
highlight_ticks = [center_day - 5, center_day, center_day + 5]
for tick in highlight_ticks:
    if tick not in tick_positions:
        tick_positions.append(tick)

# Set the x-ticks and labels
ax.set_xticks(tick_positions)  # Set x-ticks explicitly

# Set custom labels for the highlighted ticks: "i-5", "i", "i+5"
tick_labels = []
for tick in tick_positions:
    if tick == center_day - 5:
        tick_labels.append("i -5")
    elif tick == center_day:
        tick_labels.append("i")
    elif tick == center_day + 5:
        tick_labels.append("i+5")
    else:
        tick_labels.append(str(int(tick)))

ax.set_xticklabels(tick_labels)  # Set the corresponding labels

# Highlight the specific ticks in red
for i, tick in enumerate(tick_positions):
    if tick in highlight_ticks:
        ax.get_xticklabels()[i].set_color('red')  # Change the label color to red

# Plot month lines and labels
month_days = {
    'January': 1, 'February': 32, 'March': 60, 'April': 91, 'May': 120, 'June': 151, 
    'July': 182, 'August': 213, 'September': 244, 'October': 274, 'November': 305, 'December': 335
}
month_days_filtered = {month: day for month, day in month_days.items() if center_day - 20 <= day <= center_day + 20}
for month, day in month_days_filtered.items():
    ax.axvline(day, color='#014F86', linestyle='--', alpha=0.9)
    ax.text(day-0.4, ax.get_ylim()[1]-0.09, month, rotation=90, verticalalignment='bottom', horizontalalignment='center', color='#014F86', fontsize=10)

ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()



# %% --- Climatology time serie
theta = 2 * np.pi * (np.arange(0, 365) / 365)  # Convert days to angles (0 to 2π)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
# mean
ds_avg30yrs = selected_temp_surf.mean(dim='year')
ax.plot(theta, ds_avg30yrs, color='black', linewidth=1.5, alpha=0.8, label='Mean')

# Clim
clim_to_plot = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi)
ax.plot(theta, clim_to_plot.values, color= '#C2ADFF', linewidth=3, label= 'Climatology (median)')

ax.set_ylim(np.nanmin(selected_temp_surf), np.nanmax(selected_temp_surf))
month_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(month_angles)
ax.set_xticklabels(month_labels, fontsize=10)
ax.set_ylabel('SST (°C)', fontsize=15, labelpad=20)
ax.set_title(f'Sea Surface Temperature (with median climatology)\nLocation: ({round(lat)}°S, {round(lon)}°E)', va='bottom', fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor =(0.98, 0.9))
plt.tight_layout()
plt.show()


# %% --- Climatology map
fn = output_path_clim +  'climSST_all_eta.nc' #dim: (days: 365, z_rho: 35, eta_rho: 434, xi_rho: 1442)
clim_sst_surf = xr.open_dataset(fn)['clim_sst'][:,0,:,:]

lon = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi).lon_rho.item()
lat = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi).lat_rho.item()
value = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi, days=230).item()

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# -------- Plot the full map --------
pcolormesh = clim_sst_surf.sel(days=98).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False, 
    vmin=-5, vmax=5,
    cmap='coolwarm'
)

# --- Plot 1 latitude band
# temp_selected = temp_surf.isel(year=20, day=98)

# Duplicate the latitude slightly to create a thin band
# lat_band = np.vstack([
#     temp_selected.lat_rho.values - 0.2,  # slightly south
#     temp_selected.lat_rho.values + 0.2   # slightly north
# ])  # shape (2, n_lon)

# # Repeat longitudes to match
# lon_band = np.vstack([
#     temp_selected.lon_rho.values,
#     temp_selected.lon_rho.values
# ])  # shape (2, n_lon)

# # Duplicate temp values to match
# temp_band = np.vstack([
#     temp_selected.values,
#     temp_selected.values
# ])  # shape (2, n_lon)

# Then use pcolormesh
# pcolormesh = ax.pcolormesh(
#     lon_band, lat_band, temp_band,
#     transform=ccrs.PlateCarree(),
#     vmin=-5, vmax=5,
#     cmap='plasma'
# )

# temp_DC_BC_surface = xr.open_dataset( os.path.join(path_mhw, f"temp_DC_BC_surface.nc"))[var][1:, 0:365, :, :]  # 30yrs - 365days per year
# temp_DC_BC_surface_1990 = temp_DC_BC_surface.sel(year=1990)

# pcolormesh = temp_DC_BC_surface_1990.sel(day=98).plot.pcolormesh(
#     ax=ax, transform=ccrs.PlateCarree(),
#     x="lon_rho", y="lat_rho",
#     add_colorbar=False, 
#     vmin=-5, vmax=5,
#     cmap='coolwarm'
# )

# -------- Plot scatter point on top --------
sc = ax.scatter(lon, lat, c=[value], cmap='coolwarm', vmin=-5, vmax=5,
                transform=ccrs.PlateCarree(), s=100, edgecolor='black', zorder=3, label='Selected Cell')

# -------- Add colorbar --------
cbar = plt.colorbar(pcolormesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
cbar.set_label('°C', fontsize=13)
cbar.ax.tick_params(labelsize=12)

# -------- Add map features --------
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgray')
ax.set_facecolor('lightgrey')

# -------- Title --------
ax.set_title(f"SST median climatolgy (day=98) \nLocation: ({round(lat)}°S, {round(lon)}°E)", fontsize=16, pad=30)
# ax.set_title(f"SST in 1990 (day=98) \nLocation: ({round(lat)}°S, {round(lon)}°E)", fontsize=16, pad=30)

plt.tight_layout()
plt.show()


# %% --- Relative threshold map
fn = output_path_clim +  'rel_threshold_all_eta.nc' #dim: (days: 365, z_rho: 35, eta_rho: 434, xi_rho: 1442)
rel_threshold_surf = xr.open_dataset(fn)['relative_threshold'][:, 0, :, :]

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# -------- Plot the full map --------
pcolormesh = ds_rel_threshold.isel(z_rho=0, days=98).relative_threshold.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False, 
    vmin=-5, vmax=5,
    cmap='coolwarm'
)

# -------- Add colorbar --------
cbar = plt.colorbar(pcolormesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
cbar.set_label('°C', fontsize=13)
cbar.ax.tick_params(labelsize=12)

# -------- Add map features --------
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgray')
ax.set_facecolor('lightgrey')

# -------- Title --------
ax.set_title(f"MHW intensity \n relative threshold: 90th percentile threshold", fontsize=16, pad=30)

plt.tight_layout()
plt.show()
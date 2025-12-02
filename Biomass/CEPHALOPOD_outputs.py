"""
Created on Tue 25 Nov 10:51:30 2025

CEPHALOPOD model outputs for biomass and abundance

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


# %% ====================== Load Biomass data ======================
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
# Save 
output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/CEPHALOPOD/1st_run/euphausia_output/'
output_file_biomass = os.path.join(output_path, "euphausia_biomass_SO.nc")
if not os.path.exists(output_file_biomass):
    biomass_cephalopod_60S.to_netcdf(output_file_biomass, engine="netcdf4")

# %% ====================== Plot ======================
ds_1d = biomass_cephalopod_60S.euphausia_biomass.isel(months=10, bootstrap=8)
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
cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.7, pad=0.05, extend='both')
cbar.set_label('Biomass')

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
mean_biomass_nov = biomass_cephalopod_60S.euphausia_biomass.mean(dim='bootstrap')
std_biomass_nov = biomass_nov.std(dim='bootstrap')

# %% ====================== Load abundance data ======================
# --- Load
abundance_data = xr.open_dataset('/net/meso/work/aschickele/CEPHALOPOD/output/Marguerite_krill_SO_2025-11-21 16:44:03.790757/236217/euphausia_abundance.nc') #shape (bootstrap=10, northing=64800, time=12)

# --- Reformatting
nlat, nlon = 180, 360
lat = np.linspace(-89.5, 89.5, nlat)
lon = np.linspace(-179.5, 179.5, nlon)

# Raw data
arr = abundance_data.euphausia_abundance.values  # shape: (bootstrap=10, northing=64800, time=12)

# Transpose to (months, bootstrap, northing)
arr = arr.transpose(2, 0, 1)  # (12, 10, 64800)

# Reshape northing into (lat, lon)
arr = arr.reshape(12, 10, 180, 360)  # (months, bootstrap, lat, lon)

# Flip latitude (before it was going from -90 to 90, i.e. from south pole to north pole)
arr_global = arr[:, :, ::-1, :]  # flip along lat axis
lat_flipped = lat[::-1]

# Create new Dataset
abundance_cephalopod = xr.Dataset(
    data_vars=dict(euphausia_abundance=(["months", "bootstrap", "lat", "lon"], arr_global)), 
    coords=dict(months=np.arange(1, 13),
                bootstrap=np.arange(1, 11),
                lat=lat_flipped, lon=lon),
    attrs=abundance_data.attrs)

# Add info in attributed
abundance_cephalopod.attrs.update({
    "model_name": "Cephalopod",
    "model_resolution": "1 degree",
    "model_extent": "global",
    "model_inputs": "WOA",
    "ensemble_member_selection": "RF and SVM",
    "note": "Under assumption that krill spend most of their time in the 0-100m, all observations are integrated (median concentration on depth)."
})

# -- Select only the Southern Ocean (south of 60°S)
arr = abundance_data.euphausia_abundance.values.transpose(2, 0, 1).reshape(12, 10, 180, 360)
lat = np.linspace(89.5, -89.5, 180)  # north → south

lat_mask = lat <= -60
arr_60S = arr[:, :, lat_mask, :]
lat_60S = lat[lat_mask]  # lat_60S: -60.5 → -89.5 (north → south)

arr_60S_flipped = arr_60S[:, :, ::-1, :]
lat_60S_flipped = lat_60S[::-1]  # now first row = south pole

abundance_cephalopod_60S = xr.Dataset(
    data_vars=dict(euphausia_abundance=(["months", "bootstrap", "lat", "lon"], arr_60S_flipped)),
    coords=dict(
        months=np.arange(1, 13),
        bootstrap=np.arange(1, 11),
        lat=lat_60S_flipped,
        lon=np.linspace(-179.5, 179.5, 360)
    )
)

# Save 
output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/CEPHALOPOD/1st_run/euphausia_output/'
output_file_abundance = os.path.join(output_path, "euphausia_abundance_SO.nc")
if not os.path.exists(output_file_abundance):
    abundance_cephalopod_60S.to_netcdf(output_file_abundance, engine="netcdf4")

# %% ====================== Regridding Biomass and Abundance to ROMS grid ======================
import xesmf as xe

# Run only if file don't exist already
output_file_regrid = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/CEPHALOPOD/1st_run/euphausia_output/regridded_outputs'
output_file_abundance_regridd= os.path.join(output_file_regrid, "euphausia_abundance_SO_regridded.nc")
output_file_biomass_regridd= os.path.join(output_file_regrid, "euphausia_biomass_SO_regridded.nc")

if not (os.path.exists(output_file_abundance_regridd) and os.path.exists(output_file_biomass_regridd)):
    
    # -- Load dataset with correct grid
    area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc')['area'].isel(z_t=0)
    area_SO = area_roms.where(area_roms['lat_rho'] <= -60, drop=True) #shape (231, 1442)

    # Select bootstrap
    # abundance_SO_1bootstrap = abundance_cephalopod_60S.isel(bootstrap=0)

    # -- From monthly to daily dataset
    # Repeat monthly value for each day of the month
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day_index = np.concatenate([np.repeat(month, days_in_month[month-1]) for month in range(1, 13)])
    assert day_index.shape[0] == 365
    day_index_xr = xr.DataArray(day_index, dims="days", name="month")
    abundance_daily = abundance_cephalopod_60S.sel(months=day_index_xr) #shape: (365, 10, 30, 360)
    biomass_daily = biomass_cephalopod_60S.sel(months=day_index_xr) #shape: (365, 10, 30, 360)

    # -- Select only austral summer and early spring
    jan_april_abund = abundance_daily.sel(days=slice(0, 120)) # 1 Jan to 30 April (Day 0-119) - last idx excluded
    jan_april_abund.coords['days'] = jan_april_abund.coords['days'] #keep info on day
    nov_dec_abund = abundance_daily.sel(days=slice(304, 366)) # 1 Nov to 31 Dec (Day 304–364) - last idx excluded
    nov_dec_abund.coords['days'] = np.arange(304, 365) #keep info on day

    abundance_daily_austral = xr.concat([nov_dec_abund, jan_april_abund], dim="days") #shape: (181, 10, 30, 360)

    jan_april_biomass = biomass_daily.sel(days=slice(0, 120))
    jan_april_biomass.coords['days'] = jan_april_biomass.coords['days'] 
    nov_dec_biomass = biomass_daily.sel(days=slice(304, 366))
    nov_dec_biomass.coords['days'] = np.arange(304, 365)
        
    biomass_daily_austral = xr.concat([nov_dec_biomass, jan_april_biomass], dim="days") #shape: (181, 10, 30, 360)

    # -- Fix longitudes
    # ROMS (24.125, 383.875) - put to (0, 360)
    roms_fixed = area_SO.assign_coords(lon_rho=(area_SO.lon_rho % 360))

    # CEPHALOPOD longitude (-180, 180) - put to (0, 360)
    abundance_fixed = abundance_daily_austral.assign_coords(lon=((abundance_daily_austral.lon % 360))).sortby("lon")
    biomass_fixed = biomass_daily_austral.assign_coords(lon=((biomass_daily_austral.lon % 360))).sortby("lon")

    # -- Target grids
    in_ds = xr.Dataset(
        {"lon": (("lon",), abundance_fixed.lon.values),
        "lat": (("lat",), abundance_fixed.lat.values),})

    out_ds = xr.Dataset(
        {"lon": (("eta_rho", "xi_rho"), roms_fixed.lon_rho.values),
        "lat": (("eta_rho", "xi_rho"), roms_fixed.lat_rho.values),})

    # -- Regridding 
    # Build xESMF regridder 
    regridder = xe.Regridder(
        in_ds,
        out_ds,
        method="bilinear", #weighted avg of the 4 nearest neighbors
        periodic=True, 
        extrap_method="nearest_s2d"  # fill edges with nearest valid neighbor (otherwise, invalid data -> since do not have 4 neighbors)
    ) 

    # Perform regridding
    abundance_regridded = regridder(abundance_fixed) #shape (181, 10, 231, 1442)
    biomass_regridded = regridder(biomass_fixed) #shape (181, 10, 231, 1442)

    # -- Save to file 
    abundance_regridded.to_netcdf(output_file_abundance_regridd, engine="netcdf4")
    biomass_regridded.to_netcdf(output_file_biomass_regridd, engine="netcdf4")

else:
    abundance_regridded = xr.open_dataset(output_file_abundance_regridd)
    biomass_regridded = xr.open_dataset(output_file_biomass_regridd)

# %% ====================== Plot regridded product ======================

show_differences=True

if show_differences:

    # ----------------- Circular boundary -----------------
    theta = np.linspace(0, 2*np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)

    # ----------------- Figure -----------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})

    # ----------------- Titles -----------------
    fig.text(0.27, 0.95, "BEFORE regridding (1°)", ha='center', fontsize=16)
    fig.text(0.73, 0.95, "AFTER regridding (ROMS 0.25°)", ha='center', fontsize=16)
    fig.text(0.03, 0.72, "Biomass [ind/m²]", va='center', rotation='vertical', fontsize=15)
    fig.text(0.03, 0.27, "Abundance [ind/m³]", va='center', rotation='vertical', fontsize=15)
    fig.text(0.5, 0.92, "1st November — Bootstrap N°0", ha='center', fontsize=14)

    # ----------------- Color norms -----------------
    norm_biomass = mcolors.Normalize(*np.nanpercentile(biomass_daily_austral.euphausia_biomass.isel(days=0, bootstrap=0), [5, 95]))
    norm_abundance = mcolors.Normalize(*np.nanpercentile(abundance_daily_austral.euphausia_abundance.isel(days=0, bootstrap=0), [5, 95]))

    # ----------------- Helper function -----------------
    def plot_panel(ax, lon, lat, data, cmap, norm):
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        return ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1)

    # ----------------- Row 0: Biomass -----------------
    mesh_biomass_before = plot_panel(axes[0,0],
                                    biomass_daily_austral.lon,
                                    biomass_daily_austral.lat,
                                    biomass_daily_austral.euphausia_biomass.isel(days=0, bootstrap=0),
                                    cmap='coolwarm', norm=norm_biomass)
    mesh_biomass_after = plot_panel(axes[0,1],
                                    area_SO.lon_rho,
                                    area_SO.lat_rho,
                                    biomass_regridded.euphausia_biomass.isel(days=0, bootstrap=0),
                                    cmap='coolwarm', norm=norm_biomass)

    # ----------------- Row 1: Abundance -----------------
    mesh_abundance_before = plot_panel(axes[1,0],
                                    abundance_daily_austral.lon,
                                    abundance_daily_austral.lat,
                                    abundance_daily_austral.euphausia_abundance.isel(days=0, bootstrap=0),
                                    cmap='viridis', norm=norm_abundance)
    mesh_abundance_after = plot_panel(axes[1,1],
                                    area_SO.lon_rho,
                                    area_SO.lat_rho,
                                    abundance_regridded.euphausia_abundance.isel(days=0, bootstrap=0),
                                    cmap='viridis', norm=norm_abundance)

    # ----------------- Colorbars on the right -----------------
    cbar_ax_biomass = fig.add_axes([0.93, 0.55, 0.015, 0.35])  # top row
    fig.colorbar(mesh_biomass_before, cax=cbar_ax_biomass, orientation='vertical', extend='both', label='[ind/m²]')

    cbar_ax_abundance = fig.add_axes([0.93, 0.1, 0.015, 0.35])  # bottom row
    fig.colorbar(mesh_abundance_before, cax=cbar_ax_abundance, orientation='vertical', extend='both', label='[ind/m³]')

    # ----------------- Adjust spacing -----------------
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05, hspace=0.3, wspace=0.1)

    plt.show()



# %% ====================== Plot ======================
# Months to plot (Nov → Apr)
months_idx = [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]
month_names = ['4', '5', '6', '7', '8', '9', '10', 'November', 'December', 'January', 'February', 'March', 'April']

# Mean across bootstrap
mean_abundance = abundance_cephalopod_60S.euphausia_abundance.mean(dim='bootstrap')

# Colors Settings
epsilon = 1e-3
log_data_all = [np.log10(mean_abundance.isel(months=m).values + epsilon) for m in months_idx] # Log-transform 
vmin, vmax = np.nanpercentile(np.array(log_data_all), [5, 95])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(3, 4, figsize=(15, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})
axes = axes.flatten()

for i, ax in enumerate(axes):
    ds_1d = mean_abundance.isel(months=months_idx[i])
    data = np.log10(ds_1d.values + epsilon)
    lon2d, lat2d = np.meshgrid(ds_1d.lon.values, ds_1d.lat.values)

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Base map
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Plot data
    pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(),
                        cmap='viridis', norm=norm, shading='auto')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.7,
                      linestyle='--', linewidth=0.4, zorder=7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'rotation': 0}
    gl.ylabel_style = {'rotation': 0}

    ax.set_title(f"{month_names[i]}", fontsize=14)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(pcm, cax=cbar_ax, extend='both')
cbar.set_label('log$_{10}$(Abundance)', fontsize=12)

plt.subplots_adjust(wspace=0.1, hspace=0.25)
plt.show()


# %%
# %% ====================== Mean Map ======================
# Months to average (Nov → Apr)
months_idx = [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]

# Mean across bootstrap
mean_abundance = abundance_cephalopod_60S.euphausia_abundance.mean(dim='bootstrap')

# Mean across selected months
mean_over_months = mean_abundance.isel(months=months_idx).median(dim='months')

# Prepare data
epsilon = 1e-3
# data = np.log10(mean_over_months.values + epsilon)
data = mean_over_months.values

lon2d, lat2d = np.meshgrid(mean_over_months.lon.values,
                           mean_over_months.lat.values)

# Compute color scale from *all* values
vmin, vmax = np.nanpercentile(data, [5, 95])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# ---- Plot ----
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

# Map features
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Plot
pcm = ax.pcolormesh(lon2d, lat2d, data,
                    transform=ccrs.PlateCarree(),
                    cmap='viridis', norm=norm,
                    shading='auto')

# Gridlines
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.7,
                  linestyle='--', linewidth=0.4, zorder=7)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'rotation': 0}
gl.ylabel_style = {'rotation': 0}

ax.set_title("Median Euphausia abundance (Nov–Apr)", fontsize=13)

# Colorbar
cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.7, pad=0.05, extend='both')
cbar.set_label('Abundance', fontsize=12)

plt.tight_layout()
plt.show()

# %%

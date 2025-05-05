#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 31 March 09:54:34 2025

1st attempt - Growth model according to Atkinson et al. (2006) 

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
doy_list = list(range(305, 365)) + list(range(0, 121))
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

# %% Defining constants
# ---- Coefficients of models predicting DGR and GI from length, food, and temperature in Eq. 4 (Atkinson et al., 2006), Here we use model4, i.e. sex and maturity considered (krill length min 35mm)
a, std_a= np.mean([-0.196, -0.216]), 0.156  # constant term. mean value between males and mature females 

# Length
b, std_b = 0.00674,  0.00611 #linear term 
c, std_c = -0.000101, 0.000071 #quadratic term 

# Food
d, std_d = 0.377, 0.087 #maximum term
e, std_e = 0.321, 0.232 #half saturation constant

# Temperature
f, std_f = 0.013, 0.0163 #linear term
g, std_g = -0.0115, 0.00420 #quadratic term 

# H = #random effect for unexplained variation

length=35 # mean body length in adult krill (Michael et al. 2021 / Tarling 2020)
print(type(a), type(b), type(c), type(d), type(e), type(f), type(g))
print(type(length))

# %% Load data
# ==== MHWs events detected -- only surface for now
# det_files = glob.glob(os.path.join(path_combined_thesh, "det_*.nc"))#detected event (T°C > abs & rel thresholds) - boolean - shape (40, 181, 434, 1442)
det_files= os.path.join(path_combined_thesh, 'det_depth5m.nc')
import re
def extract_depth(filename):
    """Extract integer depth value from filename like 'det_depth44m.nc' → 44"""
    match = re.search(r'depth(\d+)m', filename)
    return int(match.group(1)) if match else None

# Add depth as coord (forgot before)
depth = extract_depth(det_files)
det_combined_ds = xr.open_dataset(det_files)
det_combined_ds = det_combined_ds.assign_coords(depth=depth) 

# ==== Temperature [°C] -- Weighted averaged temperature of the first 100m - Austral summer - 60S
temp_avg_100m = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 181, 231, 1442)
temp_avg_100m = temp_avg_100m.rename({'__xarray_dataarray_variable__':'avg_temp'})
temp_avg_100m = temp_avg_100m.rename({'year': 'years'})
# temp_avg_100m.avg_temp.isel(years=30, days=30).plot()

# ==== Chla [mh Chla/m3] -- Weighted averaged chla of the first 100m - Austral summer - 60S
chla_avg_100m= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_avg100m_allyears.nc')) 
chla_avg_100m = chla_avg_100m.rename({'year': 'years'})
# chla_avg_100m.raw_chla.isel(years=30, days=30).plot()

# Reformating - stacking time dimension #shape (231, 1442, 7240)
temp_100m_stack = temp_avg_100m.stack(time= ['years', 'days'])
chla_100m_stack = chla_avg_100m.stack(time= ['years', 'days']) 


# %% Check Zones
ds = temp_avg_100m
south_mask = ds['lat_rho'] <= -65
test1 = ds.where(south_mask, drop=True)
test1 = test1.isel(years=slice(30,40))
mean_1 = test1.avg_temp.mean(dim=('eta_rho','xi_rho','days')) 
print(f"South of 65°S - mean temperature (2010-2019): {mean_1.mean(dim='years').values}°C")
mean_1.plot.hist(bins=10, color='#005D8F', edgecolor='black')
plt.title("Yearly Average Temperature (South of 65°S)\n period 2010-2019")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


north_mask = ds['lat_rho'] >= -65
test2 = ds.where(north_mask, drop=True)
mean_2 = test2.avg_temp.mean(dim=('eta_rho','xi_rho','days'))#, 'years'))
print(f"North of 65°S - mean temperature (2010-2019): {mean_2.mean(dim='years').values}°C")
mean_2.plot.hist(bins=10, color='#780000', edgecolor='black')
plt.title("Yearly Average Temperature (North of 65°S)\n period 2010-2019")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()

# %% Growth model - Eq. 4 (Atkinson et al. 2006)
growth_da = a + b* length + c *length**2 + (d*chla_100m_stack.raw_chla)/(e+chla_100m_stack.raw_chla) + f*temp_100m_stack.avg_temp + g*temp_100m_stack.avg_temp**2 #[mm] - predicted daily (computing time ~40min)
growth_redimensioned = growth_da.unstack('time')
# growth_redimensioned.isel(years=1991-1980, days=349-1).plot() 
# growth_redimensioned.isel(years=2010-1980, days=35-1).plot() 

# Find min and max
# flat_index_max = np.nanargmax(growth_redimensioned.values)
# flat_index_min = np.nanargmin(growth_redimensioned.values)
# max_pos = np.unravel_index(flat_index_max, growth_da.shape)  # (eta, xi, time)
# min_pos = np.unravel_index(flat_index_min, growth_da.shape)  # (eta, xi, time)

# max_eta, max_xi = max_pos[0], max_pos[1]
# min_eta, min_xi = min_pos[0], min_pos[1]
# year_max, day_max = growth_da.coords['years'].values[max_pos[2]], growth_da.coords['days'].values[max_pos[2]]
# year_min, day_min = growth_da.coords['years'].values[min_pos[2]], growth_da.coords['days'].values[min_pos[2]]

# print(f"📈 Max growth in SO at (eta_rho={max_eta}, xi_rho={max_xi}): {growth_da.values[max_pos]} mm")
# print(f'Associated SST and Chla: {temp_100m_stack.values[max_pos]}°C and {chla_stack.raw_chla.values[max_pos]} mgChla/m3')
# print(f"Date: {year_max}, Day {day_max}\n")

# print(f"📉 Min growth in SO at (eta_rho={min_eta}, xi_rho={min_xi}): {growth_da.values[min_pos]} mm")
# print('Associated SST and Chla:', temp_100m_stack.values[min_pos], "°C and ", chla_stack.raw_chla.values[min_pos], "mgChla/m3")
# print(f"Date: {year_min}, Day {day_min}\n")


# Write to file
growth_ds =xr.Dataset(
    {"growth": (["eta_rho", "xi_rho", "years", "days"], growth_redimensioned.data)},
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lon_rho.values),
        lat_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lat_rho.values),
        years = (['years'], growth_redimensioned.years.values),
        days = (['days'], growth_redimensioned.days.values),
        depth= depth
    ),
    attrs={"description": "Growth of krill based on Atkinson et al (2006) equation, model4 (sex and maturity considered)"}
)

growth_clim = growth_ds.isel(years=slice(0,30))
growth_2010_2019 = growth_ds.isel(years=slice(30,40))

# growth_ds.to_netcdf(path=os.path.join(path_growth, "growth_1st_attempt.nc"), mode='w')

# %% Influence of SST and Chla on growth - 2D Histogram
growth_ds = xr.open_dataset(os.path.join(path_growth, "growth_1st_attempt.nc"))

import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

# Normalize color scale with center at 0
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_ds.growth), vcenter=0, vmax=np.nanmax(growth_ds.growth))

# === Plot Location on map ===
eta = 100 #150 #100 #205 #220 #230 #200 #220 #45
xi = 800 #200 #1150 #1105 #100 #600 #1000 #950 #1110
lat = np.unique(temp_100m_stack.isel(eta_rho=eta, xi_rho=xi).lat_rho)[0]
lon = np.unique(temp_100m_stack.isel(eta_rho=eta, xi_rho=xi).lon_rho)[0]
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Plot the data for the current day
pcolormesh = growth_ds.growth.isel(years=37, days= growth_ds.coords['days'].values.tolist().index(0)).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False,
    cmap='PuOr_r', norm=norm)

# pcolormesh = temp_avg_100m.avg_temp.isel(years=37, days= temp_avg_100m.coords['days'].values.tolist().index(0)).plot.pcolormesh(
#     ax=ax, transform=ccrs.PlateCarree(),
#     x="lon_rho", y="lat_rho",
#     add_colorbar=False,
#     cmap='coolwarm', norm=norm)

# pcolormesh = test1.avg_temp.isel(years=37, days= test1.coords['days'].values.tolist().index(0)).plot.pcolormesh(
#     ax=ax, transform=ccrs.PlateCarree(),
#     x="lon_rho", y="lat_rho",
#     add_colorbar=False,
#     cmap='Blues', norm=norm)
# pcolormesh = test2.avg_temp.isel(years=37, days= test2.coords['days'].values.tolist().index(0)).plot.pcolormesh(
#     ax=ax, transform=ccrs.PlateCarree(),
#     x="lon_rho", y="lat_rho",
#     add_colorbar=False,
#     cmap='Reds', norm=norm)

point_data = growth_ds.growth.isel(eta_rho=eta, xi_rho=xi, years=37, days= growth_ds.coords['days'].values.tolist().index(0))
sc = ax.scatter(point_data.lon_rho.item(), point_data.lat_rho.item(), color='red', marker='*', s=200, transform=ccrs.PlateCarree(), edgecolor='red', zorder=3)

# Map features
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
ax.set_facecolor('lightgrey')

# Sectors
ax.plot([-85, -85], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Atlantic sector
ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Pacific sector
ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Indian sector

plt.suptitle("")
plt.tight_layout()
plt.show()

# === Scatter plot - 1 location ===
chl_1980_2009 = chla_avg_100m.raw_chla.isel(eta_rho=eta, xi_rho=xi, years=slice(0, 30)).values.flatten()
chl_2010_2019 = chla_avg_100m.raw_chla.isel(eta_rho=eta, xi_rho=xi, years=slice(30, 40)).values.flatten()
temp_1980_2009 = temp_avg_100m.avg_temp.isel(eta_rho=eta, xi_rho=xi, years=slice(0, 30)).values.flatten()
temp_2010_2019 = temp_avg_100m.avg_temp.isel(eta_rho=eta, xi_rho=xi, years=slice(30, 40)).values.flatten()

# Determine shared x-axis limits per variable
chl_min = min(np.min(chl_1980_2009), np.min(chl_2010_2019))
chl_max = max(np.max(chl_1980_2009), np.max(chl_2010_2019))
temp_min = min(np.min(temp_1980_2009), np.min(temp_2010_2019))
temp_max = max(np.max(temp_1980_2009), np.max(temp_2010_2019))

# 3 subplots : (1980-2019), (1980-2009), (2010,2019)
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# Full period (1980–2019)
sc0 = axs[0].scatter(x=temp_avg_100m.avg_temp.isel(eta_rho=eta, xi_rho=xi).values, 
                     y=chla_avg_100m.raw_chla.isel(eta_rho=eta, xi_rho=xi).values, 
                     c=growth_ds.growth.isel(eta_rho=eta, xi_rho=xi).values,
                     cmap='PuOr_r', norm=norm, s=10, alpha=0.7)
axs[0].set_title('Full period (1980–2019)', fontsize=13)

# Climatology (first 30 years)
sc1 = axs[1].scatter(x=temp_avg_100m.avg_temp.isel(eta_rho=eta, xi_rho=xi, years=slice(0, 30)).values,
                     y=chla_avg_100m.raw_chla.isel(eta_rho=eta, xi_rho=xi, years=slice(0, 30)).values,
                     c=growth_ds.growth.isel(eta_rho=eta, xi_rho=xi, years=slice(0, 30)).values,
                     cmap='PuOr_r', norm=norm, s=10, alpha=0.7)
axs[1].set_title('Climatology (1980–2009)', fontsize=13)

# Recent decade (2010–2019)
sc2 = axs[2].scatter(x=temp_avg_100m.avg_temp.isel(eta_rho=eta, xi_rho=xi, years=slice(30, 40)).values,
                     y=chla_avg_100m.raw_chla.isel(eta_rho=eta, xi_rho=xi, years=slice(30, 40)).values,
                     c=growth_ds.growth.isel(eta_rho=eta, xi_rho=xi, years=slice(30, 40)).values,
                     cmap='PuOr_r', norm=norm, s=10, alpha=0.7)
axs[2].set_title('Recent (2010–2019)', fontsize=13)

# Shared labels
for ax in axs:
    ax.set_xlabel('Temperature (°C)', fontsize=12)
axs[0].set_ylabel('Chlorophyll-a (mg/m³)', fontsize=12)

for ax in axs:
    ax.set_xlim(temp_min, temp_max)
    ax.set_ylim(chl_min, chl_max)

# Colorbar (common for all)
pos = axs[-1].get_position()
plt.subplots_adjust(right=1.5)
cbar_ax = fig.add_axes([pos.x1+ 0.1, pos.y0-0.015, 0.01, pos.height-0.022]) #[right, bottom, wideness, height]
cbar = fig.colorbar(sc2, cax=cbar_ax) #since same normalization doesn't matter which sc to put in the colorbar
cbar.set_label('Growth [mm]', fontsize=12)
cbar.ax.tick_params(labelsize=11)

fig.suptitle(f'Growth at ({np.round(lat)}°S, {np.round(lon)}°E)', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# === CHeck temperature and chla for regions plotted
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

# Plot Chl-a 1980–2009
axes[0, 0].hist(chl_1980_2009, bins=40, color='seagreen', edgecolor='black')
axes[0, 0].set_xlim(chl_min, chl_max)
axes[0, 0].set_ylabel("Frequency", fontsize=12)
axes[0, 0].set_title("1980–2009", fontsize=14)
axes[0, 0].tick_params(labelsize=12)
axes[0, 0].set_xlabel("Chl-a (mg/m³)", fontsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

# Plot Chl-a 2010–2019
axes[0, 1].hist(chl_2010_2019, bins=40, color='seagreen', edgecolor='black')
axes[0, 1].set_xlim(chl_min, chl_max)
axes[0, 1].set_title("2010–2019", fontsize=14)
axes[0, 1].tick_params(labelsize=12)
axes[0, 1].set_xlabel("Chl-a (mg/m³)", fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.5)

# Plot Temp 1980–2009
axes[1, 0].hist(temp_1980_2009, bins=40, color='firebrick', edgecolor='black')
axes[1, 0].set_xlim(temp_min, temp_max)
axes[1, 0].set_ylabel("Frequency", fontsize=12)
axes[1, 0].set_xlabel("Temperature (°C)", fontsize=12)
axes[1, 0].tick_params(labelsize=12)
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

# Plot Temp 2010–2019
axes[1, 1].hist(temp_2010_2019, bins=40, color='firebrick', edgecolor='black')
axes[1, 1].set_xlim(temp_min, temp_max)
axes[1, 1].set_xlabel("Temperature (°C)", fontsize=12)
axes[1, 1].tick_params(labelsize=12)
axes[1, 1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# %% === 2D Histogram - full SO ===
# # Flatten
# x = temp_avg_100m.avg_temp.values.flatten()
# y = chla_avg_100m.raw_chla.values.flatten()
# z_all = growth_ds.growth.values.flatten()
# z_clim = growth_clim.growth.values.flatten()
# z_2010_2019 = growth_2010_2019.growth.values.flatten()

# # Filter NaNs
# mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z_clim))
# x, y, z = x[mask], y[mask], z_clim[mask]

# # Bin the data
# from scipy.stats import binned_statistic_2d
# import matplotlib.colors as mcolors
# from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# stat, xedges, yedges, binnumber = binned_statistic_2d(x, y, z, statistic='mean', bins=[40, 40]) #~40min
# norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(stat), vcenter=0, vmax=np.nanmax(stat))

# # 2D histogram
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(xedges, yedges, stat.T, cmap='PuOr_r', norm=norm, shading='auto')
# plt.colorbar(label='Growth [mm]')
# plt.ylim(1, 10)
# plt.xlabel('Temperature [°C]')
# plt.ylabel('Chl-a [mg/m³]')
# plt.title('E. Superba Growth in the Southern Ocean (1980-2019) \n Atkison et al. (2006) model')
# plt.tight_layout()
# plt.show()

#%% === Plot averaged growth ===

import cmocean
import matplotlib.colors as mcolors

def plot_comparison(varname, ds, cmap_var=None, ticks=None, cbar_label=''):
    # Prepare figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    
    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    # Time periods
    data_1980_2009 = ds.isel(years=slice(0, 30)).mean(dim=('years', 'days'))
    data_2010_2019 = ds.isel(years=slice(30, 40)).mean(dim=('years', 'days'))
    data_diff = data_2010_2019 - data_1980_2009

    # Set normalization
    if varname in ['temp', 'growth']:
        vmin = min(data_1980_2009.min(), data_2010_2019.min())
        vmax = max(data_1980_2009.max(), data_2010_2019.max())
        abs_max = max(abs(vmin), abs(vmax))
        norm_main = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    elif varname == 'chla':
        vmax = data_1980_2009.max()  # Max value for chla colormap
        norm_main = mcolors.Normalize(vmin=0, vmax=vmax)
    else:
        norm_main = None

    # Difference normalization (centered at zero for difference)
    abs_diff_max = np.max(np.abs(data_diff))
    norm_diff = mcolors.TwoSlopeNorm(vmin=-abs_diff_max, vcenter=0, vmax=abs_diff_max)
    cmap_diff = plt.cm.RdBu_r

    # Titles and datasets
    plot_data = [(data_1980_2009, f"Avg {varname} (1980–2009)", norm_main, cmap_var),
                 (data_2010_2019, f"Avg {varname} (2010–2019)", norm_main, cmap_var),
                 (data_diff, "Difference${_{({warming}-{climatology})}}$", norm_diff, cmap_diff),]
    # Plotting data
    scs = []  # List to hold the scatter plot objects for each subplot
    for ax, (data, title, norm, cmap_used) in zip(axs, plot_data):
        sc = data.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            x="lon_rho", y="lat_rho",
            add_colorbar=False, cmap=cmap_used, norm=norm, zorder=1)
        scs.append(sc)  # Store the plot object
        
        ax.set_title(title, fontsize=16)
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Draw the land feature after the pcolormesh
        ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
        ax.coastlines(color='black', linewidth=1)
        
        ax.set_facecolor('#F6F6F3')
        
        # Sector boundaries
        ax.plot([-85, -85], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
        ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
        ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.3)

    # Colorbar for the first and second subplots
    if varname in ['chla', 'growth', 'temp']:
        ticks = ticks
    else:
        ticks = np.arange(np.floor(norm.vmin)-1, np.ceil(norm.vmax) + 1, 1)
    
    pos1 = axs[1].get_position()
    cbar_ax1 = fig.add_axes([pos1.x1 + 0.01, pos1.y0, 0.01, pos1.height])
    cbar1 = fig.colorbar(scs[0], cax=cbar_ax1, cmap=cmap_var, ticks=ticks)
    cbar1.set_label(cbar_label, fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    # Second colorbar (for the difference)
    pos2 = axs[2].get_position()
    cbar_ax2 = fig.add_axes([pos2.x1 + 0.045, pos2.y0, 0.01, pos2.height])
    cbar2 = fig.colorbar(scs[2], cax=cbar_ax2, cmap=cmap_diff)
    cbar2.set_label("Difference", fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    plt.show()


# Choose variable to plot
variable = 'growth'  #'growth', 'temp', 'chla'

if variable == 'growth':
    ds = growth_ds.growth
    cmap_var = 'PuOr_r'
    label = 'Growth [mm]'
    ticks = [-0.2, -0.1, 0, 0.1, 0.2]

elif variable == 'temp':
    ds = temp_avg_100m.avg_temp
    cmap_var =  cmocean.cm.thermal  #'inferno'
    label = 'Temperature [°C]'
    ticks = [-6, -4, -2, 0, 2, 4, 6]

elif variable == 'chla':
    ds = chla_avg_100m.raw_chla
    cmap_var = cmocean.cm.algae 
    label = 'Chla [mg/m³]'
    ticks =  [0, 0.5, 1, 1.5, 2, 2.5, 3] 

plot_comparison(variable, ds, cmap_var=cmap_var, ticks=ticks, cbar_label=label)



# %% Growth during MHW events
# === Select extent - south of 60°S
south_mask = det_combined_ds['lat_rho'] <= -60
det_combined_ds_60S = det_combined_ds.where(south_mask, drop=True) #shape (40, 181, 231, 1442)

# Combining grwoth with MHW events
masked_growth_dict = {}
for deg in range(1, 5):
    print(f'----T°C > {deg}°C and 90th perc ----')
    # deg=1
    # Extract data
    det_mask = det_combined_ds_60S[f'det_{deg}deg'].astype(bool)
    # det_mask.isel(years=30, days=180, eta_rho=230, xi_rho=1249) #True for 1°C
    # det_mask.isel(years=30, xi_rho=1000, eta_rho=200, days=0) #False for 4°C

    # Associate growth during a mhw
    growth_MHW = xr.where(det_mask, growth_ds.growth, np.nan)
    # growth_ds.growth.isel(years=30, days=180, eta_rho=230, xi_rho=1249).values #-0.02353081
    # growth_MHW.isel(years=30, xi_rho=1000, eta_rho=200, days=0).values #nan

    # Store in dict
    masked_growth_dict[f'growth_{deg}deg'] = growth_MHW

# === PLOTs
# Avg grwoth over the 'warm' period - mean (excluding nan)
datasets = [masked_growth_dict['growth_1deg'].mean(dim=('years', 'days')), 
            masked_growth_dict['growth_2deg'].mean(dim=('years', 'days')), 
            masked_growth_dict['growth_3deg'].mean(dim=('years', 'days')), 
            masked_growth_dict['growth_4deg'].mean(dim=('years', 'days'))]

# Prepare figure and axes
fig, axs = plt.subplots(1, 4, figsize=(16, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

for  i, (ax, ds, title) in enumerate(zip(axs, datasets, [f"Avg Growth during MHW 1°C", f"Avg Growth during MHW 2°C", "Avg Growth during MHW 3°C", "Avg Growth during MHW 4°C"])):
    
    vmin = np.min(ds)
    vmax = np.max(ds)
    # abs_max = max(abs(vmin), abs(vmax))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    sc = ds.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                            x="lon_rho", y="lat_rho",
                            add_colorbar=False, cmap='PuOr_r', norm=norm)
    ax.set_title(title, fontsize=18)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
    ax.set_facecolor('#F6F6F3')
    # Sector boundaries
    ax.plot([-85, -85], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)

# Add common colorbar
pos = axs[1].get_position()
plt.subplots_adjust(right=1.2, wspace=0.4)
cbar_ax = fig.add_axes([pos.x1 + 0.51, pos.y0 - 0.015, 0.01, pos.height - 0.022])
ticks = [-0.2, -0.1, 0, 0.1, 0.2]
cbar = fig.colorbar(sc, cax=cbar_ax, ticks=ticks)
cbar.set_label('Growth [mm]', fontsize=16)
cbar.ax.tick_params(labelsize=13)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%  Selecting 1 location for plotting
choice_eta = 220  #190, 200
choice_xi = 950  #600, 1000

# Temperature, Chlorophyll and Growth
temp_100m_selected_loc_stack = temp_100m_stack.isel(eta_rho=choice_eta, xi_rho=choice_xi)
chla_100m_selected_loc_stack = temp_100m_stack.isel(eta_rho=choice_eta, xi_rho=choice_xi)
growth_all_selected_loc_stack = growth_da.isel(eta_rho=choice_eta, xi_rho=choice_xi)
growth_clim_selected_loc = growth_clim.isel(eta_rho=choice_eta, xi_rho=choice_xi)
growth_2010_2019_selected_loc = growth_2010_2019.isel(eta_rho=choice_eta, xi_rho=choice_xi)

# Find detected events (SST > 90th perc and i°C) in selected location and remove Nans
det_selected_location = det_combined_ds.isel(eta_rho=choice_eta, xi_rho=choice_xi)

# Reformat 
det_selected_location_stack = det_selected_location.stack(time= ['years', 'days']) # shape (time: 7240)
growth_clim_selected_loc_stack = growth_clim_selected_loc.stack(time= ['years', 'days'])#shape (time: 5430)
growth_2010_2019_selected_loc_stack = growth_2010_2019_selected_loc.stack(time= ['years', 'days']) #shape (time: 1810)

det_1deg = det_selected_location_stack.det_1deg[~np.isnan(det_selected_location_stack.det_1deg)] 
det_2deg = det_selected_location_stack.det_2deg[~np.isnan(det_selected_location_stack.det_2deg)] 
det_3deg = det_selected_location_stack.det_3deg[~np.isnan(det_selected_location_stack.det_3deg)] 
det_4deg = det_selected_location_stack.det_4deg[~np.isnan(det_selected_location_stack.det_4deg)] 

# %% -- PLOT
fig = plt.figure(figsize=(15, 5))
# fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

# Plot absolute threshold lines
for thresh in absolute_thresholds:
    ax2.axhline(y=thresh, xmin=0, xmax=1, linestyle='--', color='gray', alpha=0.7, lw=1)

# Define colors and labels
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['SST>90th and 1°C', 'SST>90th and 2°C', 'SST>90th and 3°C', 'SST>90th and 4°C']

# Dictionary of detected temperature datasets
detected_temps = {
    '1deg': det_1deg,
    '2deg': det_2deg,
    '3deg': det_3deg,
    '4deg': det_4deg
}

# Loop through each dataset and add vertical bands
for (key, det_data), color in zip(detected_temps.items(), threshold_colors):
    if det_data is None or det_data.size == 0:  # Skip empty datasets
        continue
    time_idx = (det_data['years'].values - det_data['years'].values.min()) * 365 + det_data['days'].values  # Convert to days since base_year
    
    # Identify breaks in continuous segments
    time_diff = np.diff(time_idx)
    idx_split = np.where(time_diff > 1)[0] + 1
    time_segments = np.split(time_idx, idx_split)

    # Add vertical shading
    for segment in time_segments:
        if len(segment) > 1:
            ax.axvspan(segment[0], segment[-1], alpha=0.8, color=color, label=threshold_labels[int(key[0])-1])

# SST ad growth rate plot
lns1 = ax.plot(growth_all_selected_loc_stack, '-', color='#3E6F8E')
lns2 = ax2.plot(temp_100m_selected_loc_stack.avg_temp,  '-', color='black')

# Add legend (only one label per threshold)
handles, labels = [], []
for i, label in enumerate(threshold_labels):
    handles.append(plt.Rectangle((0, 0), 1, 1, color=threshold_colors[i], alpha=0.5))
# ax.legend(handles, threshold_labels, loc='upper right', bbox_to_anchor=(1.005, 1.15), ncol=2)

# -- Axis settings
ax.grid(alpha=0.5)
# y-axis left
ax.yaxis.label.set_color('#3E6F8E')
ax.spines['left'].set_color('#3E6F8E')
ax.tick_params(axis='y', colors='#3E6F8E')  
# ax.set_ylim(0, 0.3)
ax.set_ylabel(r"Growth [mm]")

# y-axis right
ax2.spines['right'].set_color('black')
ax2.yaxis.label.set_color('black')
ax2.set_ylabel(r"Temperature ($^\circ$C)")

# x-axis --  TO CHANGE NOW A YEAR IS NOT *&% DAYS SINCE ONLY 1 SEASON
ax.tick_params(axis='x', colors='black')
ticks_years = np.arange(0, 40*365, 365)
ax.set_xticks(ticks_years)  
tick_labels = np.arange(1980, 2020)
ax.set_xticklabels(tick_labels)
ax.set_xlabel("Time (days)")
ax.set_xlim(0*365, 30*365) #2014-2019
# ax.set_xlim(35*365+150, 36*365) #above 1°C
# ax.set_xlim(36*365+150, 37*365) #above 3°C
# ax.set_xlim(37*365-50, 37*365+150) #above 3°C

plt.title(f'Growth for Antarctic krill \n location: ({np.int32(np.round(temp_100m_selected_loc_stack.lat_rho.values))}°S, {np.int32(np.round(temp_100m_selected_loc_stack.lon_rho.values))}°E)')
plt.tight_layout()
plt.show()

# %% Temperature VS Chla Contribution


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue 06 Mai 07:57:14 2025

How a krill grow during 1 season under MHWs 

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
    "font.size": 10,           
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,   
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
combined_file = os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))
ds_mhw_duration = xr.open_dataset(combined_file) #shape: (40, 181, 231, 1442)

# %% Find longest and more intense MHW
det3deg = False #True #False
det4deg = True #False #True 
if det3deg:
    threshold= 'det_3deg'
if det4deg:
    threshold= 'det_4deg'
period = 'recent' #full

if period == 'recent':
    ds = ds_mhw_duration.isel(years=slice(30,39))
else: 
    ds= ds_mhw_duration

# Maximum duration of 4/3deg event
duration_filtered = ds['duration'].where(ds[threshold])
max_duration = duration_filtered.max()
# Find when and where
index = np.unravel_index(np.nanargmax(duration_filtered.values), duration_filtered.shape)
year_idx, day_idx, eta_idx, xi_idx = index
year = ds['years'].values[year_idx]
day = ds['days'].values[day_idx]
lat = ds['lat_rho'].values[eta_idx, xi_idx]
lon = ds['lon_rho'].values[eta_idx, xi_idx]
duration = duration_filtered.values[index]

print(f"Longest {threshold} MHW lasted {duration:.1f} days")
print(f"Year: {year}, Day-of-year: {day}")
print(f"Location: lat={lat:.2f}, lon={lon:.2f}")


# ==== full temporal extent ====
# Longest 4°C MHW lasted 504.0 days
# Year: 1982, Day-of-year: 349
# Location: lat=-76.58, lon=27.88

# ==== 2010-2019 temporal extent ====
# Longest det_4deg MHW lasted 613.0 days
# Year: 2017, Day-of-year: 327
# Location: lat=-60.82, lon=169.62

# Longest det_3deg MHW lasted 613.0 days
# Year: 2017, Day-of-year: 304
# Location: lat=-60.82, lon=169.62

# Select extent based on this event
growth_seasons = xr.open_dataset(os.path.join(path_growth, "growth_Atkison2006_seasonal.nc"))

# === Select extent to investigate
def subset_spatial_domain(ds, lat_range=(lat-10, 60), lon_range=(lon-10, lon+30)):
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    return ds.where((ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max) & (ds['lon_rho'] >= lon_min) & (ds['lon_rho'] <= lon_max), drop=True)

growth_study_area = subset_spatial_domain(growth_seasons) #shape : (39, 106, 161, 181)
mhw_duration_study_area = subset_spatial_domain(ds_mhw_duration) #shape (40, 181, 147, 120)

#%% === Plot Study area on map 
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

# Set year of interest
selected_year = 2017
year_index = selected_year - 1980

# Threshold info
threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']

# Create figure and subplots
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1,1], wspace=0)  # very small space

ax0 = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
ax1 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())


# === Subplot 1: Growth ===
growth_data = growth_study_area.growth.isel(years=year_index).mean(dim='days')
growth_plot = growth_data.plot.pcolormesh(
    ax=ax0, transform=ccrs.PlateCarree(),
    x='lon_rho', y='lat_rho', cmap='PuOr_r', add_colorbar=False,
    norm=mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_study_area.growth), vcenter=0, vmax=np.nanmax(growth_study_area.growth)),
    rasterized=True
)
ax0.set_title(f"Mean growth during the growing season 2017-2018", fontsize=14)

# Add colorbar for growth
cbar1 = fig.colorbar(growth_plot, ax=ax0, orientation='vertical', shrink=0.8, pad=0.05)
cbar1.set_label("Growth [mm/d]", fontsize=12)
cbar1.ax.tick_params(labelsize=11)  # Adjust tick font size here

# === Subplot 2: Detected Events ===
# Base: where no MHW was detected at any threshold
no_event_mask = (
    (mhw_duration_study_area['det_1deg'].isel(years=year_index).mean(dim='days') == 0) &
    (mhw_duration_study_area['det_2deg'].isel(years=year_index).mean(dim='days') == 0) &
    (mhw_duration_study_area['det_3deg'].isel(years=year_index).mean(dim='days') == 0) &
    (mhw_duration_study_area['det_4deg'].isel(years=year_index).mean(dim='days') == 0)
).fillna(True)

# Plot light grey background where no MHW is detected
ax1.contourf(
    mhw_duration_study_area.lon_rho, mhw_duration_study_area.lat_rho,
    no_event_mask, levels=[0.5, 1], colors=['white'],
    transform=ccrs.PlateCarree(), zorder=1
)

# Overlay detected MHW events for each threshold
for var, color in zip(threshold_vars, threshold_colors):
    event_mask = mhw_duration_study_area[var].isel(years=year_index).mean(dim='days').fillna(0)
    binary_mask = (event_mask >= 0.166).astype(int)  # 1 if event ≥ 30 days else 0
    ax1.contourf(
        mhw_duration_study_area.lon_rho, mhw_duration_study_area.lat_rho,
        binary_mask,
        levels=[0.5, 1],
        colors=[color],
        transform=ccrs.PlateCarree(),
        alpha=0.8,
        zorder=2
    )

ax1.set_title(f"Frequent MHW events during the growing season 2017-2018", fontsize=14)

# Custom legend for MHW thresholds
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='white', edgecolor='black', label='No MHW event', linewidth=0.5)]
legend_handles += [
    Patch(facecolor=c, edgecolor='black', label=l, linewidth=0.5)
    for c, l in zip(threshold_colors, threshold_labels)
]

ax1.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.045, 0.13),
    fontsize=8,
    ncol=1,
    frameon=True
)

for ax in [ax0, ax1]:
    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map extent and features
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sectors
    for i in [-85, 20, 150]:
        ax.plot([i, i], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

# fig.suptitle(f"Growth and MHW events in {selected_year}", fontsize=16, y=1.02)
fig.suptitle(f"Growth and MHW events from 1st November 2017 until 30th April 2018", fontsize=18, y=1.05)
fig.text(
    0.5, -0.1,
    f"Note that on panel2, frequent corresponds to events lasting $\geq${round(181*0.166)} days during the growing season",
    fontsize=14, ha='center'
)

# Final layout
plt.tight_layout(rect=[0, 0, -0.2, 0.95]) #[right, left. bottom, top]
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_impactMHWs/study_area.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %% Select only 1 year of interest (1st attempt)
def defining_season(ds, starting_year):

    # Get the day-of-year slices
    days_nov_dec = ds.sel(days=slice(304, 364), years=starting_year)
    days_jan_apr = ds.sel(days=slice(0, 119), years=starting_year + 1)

    # Concatenate with original day-of-year coordinate preserved
    ds_season = xr.concat([days_nov_dec, days_jan_apr], dim=xr.DataArray(np.concatenate([days_nov_dec['days'].values, days_jan_apr['days'].values]), dims="days", name="days"))

    return ds_season

# Define the starting year for the growth season (Nov–Apr)
target_start_year = year

# Calling function
growth_1season = defining_season(growth_study_area, target_start_year)
mhw_duration_1season = defining_season(mhw_duration_study_area, target_start_year)

# %% INPUTS growth calculation
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

# ==== Temperature [°C] -- Weighted averaged temperature of the first 100m - Austral summer - 60S
temp_avg_100m = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 181, 231, 1442)
temp_avg_100m = temp_avg_100m.rename({'year': 'years'})
temp_avg_100m_study_area = subset_spatial_domain(temp_avg_100m) #select spatial extent
temp_avg_100m_1season = defining_season(temp_avg_100m_study_area, target_start_year) #select temporal extent

# ==== Chla [mh Chla/m3] -- Weighted averaged chla of the first 100m - Austral summer - 60S
chla_surf= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears.nc')) 
chla_surf = chla_surf.rename({'year': 'years'})
chla_surf_study_area = subset_spatial_domain(chla_surf) #select spatial extent
chla_surf_1season = defining_season(chla_surf_study_area, target_start_year) #select temporal extent

#%% === TEST === 
temp_mean_ts = temp_avg_100m_1season['avg_temp'].mean(dim=['eta_rho', 'xi_rho'])
chla_mean_ts = chla_surf_1season['raw_chla'].mean(dim=['eta_rho', 'xi_rho'], skipna=True)
days = temp_avg_100m_1season['avg_temp'].days.values
days_xaxis = np.where(days < 304, days+ 365, days).astype(int)
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0+365, 121+365)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot temperature on ax1 (left y-axis)
ax1.plot(days_xaxis, temp_mean_ts, color='#F3722C', label='Mean Temperature (100m)')
ax1.set_xlabel("Date", fontsize=14)
ax1.set_ylabel("Temperature (°C)", color='#F3722C', fontsize=14)
ax1.tick_params(axis='y', labelcolor='#F3722C', labelsize=14)  # bigger y-axis ticks

# Create a second y-axis for chlorophyll
ax2 = ax1.twinx()
ax2.plot(days_xaxis, chla_mean_ts, color='green', label='Mean Surface Chla')
ax2.set_ylabel("Chlorophyll (mg/m³)", color='green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='green', labelsize=14)  # bigger y-axis ticks

# Define labels to keep for the shared x-axis
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []
for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)

ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, fontsize=14)  # bigger x-axis tick labels

# Title and grid
plt.title(f"Spatial mean time series of Temperature and Chlorophyll \n Austral Summer {target_start_year}", fontsize=18)
ax1.grid(True)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_impactMHWs/drivers_study_area.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %% Calculating growth
# Funtion to simulate daily growth based on CHLA, temperature, and initial length
def run_growth_model(chla, temp, initial_length, a, b, c, d, e, f, g):
    # Get dimensions
    n_days = chla.sizes['days'] #181 days
    shape = chla.isel(days=0).shape #(106, 161)
    # Initialisation - dataarray to store length
    length = xr.DataArray(np.full((n_days, *shape), np.nan), 
                          dims=("days", "eta_rho", "xi_rho"),
                          coords={"days": chla.days, "lat_rho": chla.lat_rho, "lon_rho": chla.lon_rho})

    # First step -- initial length (hypothethis - on Nov1st krill length = 35mm)
    length[0] = initial_length

    # Simulate growth day by day
    for t in range(1, n_days):
        chl = chla.isel(days=t)
        tmp = temp.isel(days=t)
        prev_len = length[t-1]
        growth = (a + b * prev_len + c * prev_len**2 + (d * chl) / (e + chl) + f * tmp + g * tmp**2) # Growth model - Eq. 4 (Atkinson et al. 2006)
        length[t] = prev_len + growth

    return length


simulated_length_study_area = run_growth_model(chla=chla_surf_1season.raw_chla, temp=temp_avg_100m_1season.avg_temp, initial_length=35, a=a, b=b, c=c, d=d, e=e, f=f, g=g)
mean_length_area_study_area = simulated_length_study_area.mean(dim=["eta_rho", "xi_rho"])
std_length_area_study_area = simulated_length_study_area.std(dim=["eta_rho", "xi_rho"])

# %% Growth during MHWs
# === Find the cells where there is 1 mhw of 4deg are happening in the season
mask = np.isclose(simulated_length_study_area.lat_rho, lat, atol=1e-4) & np.isclose(simulated_length_study_area.lon_rho, lon, atol=1e-4)
print(np.unique(mask))
indices = np.argwhere(mask)
eta_idx, xi_idx = indices[0]

# === Extract length time series for these cells
length_extreme_event = simulated_length_study_area[:, eta_idx, xi_idx]

# === When MHW happening - 1cell
mhw_flags =  mhw_duration_1season['duration'][:, eta_idx, xi_idx]>0
print(f"MHW event duration indices: {np.where(mhw_flags.values == True)[0]}")

# === Find which treshold this mhw exceed
def check_threshold_exceeded(mhw_flags, det_flags, threshold_value, threshold_name, exceed_info):
    exceed_indices = np.where(mhw_flags.values == True)[0]
    exceed_info[threshold_name] = np.zeros_like(mhw_flags.values, dtype=bool)
    
    for i in exceed_indices:
        if det_flags[i]:  # Check if threshold is exceeded
            exceed_info[threshold_name][i] = True  # Mark exceedance for that day
            
    return exceed_info

exceed_info = {}
thresholds = {
    1: mhw_duration_1season['det_1deg'],  # 1°C threshold
    2: mhw_duration_1season['det_2deg'],  # 2°C threshold
    3: mhw_duration_1season['det_3deg'],  # 3°C threshold
    4: mhw_duration_1season['det_4deg'],  # 4°C threshold
}

for threshold_value, det_flags in thresholds.items():
    exceed_info = check_threshold_exceeded(mhw_flags, det_flags[:, eta_idx, xi_idx], threshold_value, f'{threshold_value}°C and 90th perc', exceed_info)

print(exceed_info.keys())

# %% PLotting length over 1 season
from collections import OrderedDict

# Define colors and labels
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']


# Deal with day to have continuous x-axis
days_xaxis = np.where(simulated_length_study_area.days.values < 304, simulated_length_study_area.days.values + 365, simulated_length_study_area.days.values).astype(int)
from datetime import datetime, timedelta
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0+365, 121+365)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)
fig, (ax_mhw, ax_len) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05})

# --- Top subplot: MHW events filled ---
for threshold_name, color in zip(threshold_labels, threshold_colors):
    exceed = exceed_info[threshold_name]
    edges = np.diff(exceed.astype(int))
    start_indices = np.where(edges == 1)[0] + 1
    end_indices = np.where(edges == -1)[0] + 1

    if exceed[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if exceed[-1]:
        end_indices = np.append(end_indices, len(exceed) - 1)

    for start, end in zip(start_indices, end_indices):
        ax_mhw.axvspan(days_xaxis[start], days_xaxis[end], color=color,
                       label=threshold_name if start == start_indices[0] else "")

ax_mhw.set_yticks([])
ax_mhw.set_ylabel("MHW", fontsize=12)
ax_mhw.set_title(f"MHW events and Krill Length\n Growth period {target_start_year}-{target_start_year + 1}\n", fontsize=18)

# --- Bottom subplot: Length time series ---
ax_len.plot(days_xaxis, length_extreme_event, label="MHW cell", color="black")
mean_values = mean_length_area_study_area.values
std_values = std_length_area_study_area.values
ax_len.plot(days_xaxis, mean_values, label="Average Length", color="black", linestyle='--')
ax_len.fill_between(days_xaxis,
                mean_values - std_values,
                mean_values + std_values,
                color="gray", alpha=0.2, label="±1 std across area")

# Define labels to keep
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []
for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)
ax_len.set_xticks(tick_positions)
ax_len.set_xticklabels(tick_labels, rotation=45, fontsize=12)
ax_len.set_xlabel("Date", fontsize=14)
ax_len.set_ylabel("Length (mm)", fontsize=14)
ax_len.grid(True, alpha=0.3)

ax_mhw.tick_params(axis='both', labelsize=12)
ax_len.tick_params(axis='both', labelsize=12)

# Deduplicate legend
handles1, labels1 = ax_mhw.get_legend_handles_labels()
handles2, labels2 = ax_len.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
by_label = OrderedDict(zip(labels, handles))
ax_len.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=12)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_impactMHWs/length_mhw_{target_start_year}.pdf'), dpi =150, format='pdf', bbox_inches='tight')

#%% === Plot cell on map 
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
pcolormesh = growth_study_area.growth.isel(years=target_start_year-1980).mean(dim=('days')).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False,
    cmap='PuOr_r', norm=mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_study_area.growth), vcenter=0, vmax=np.nanmax(growth_study_area.growth))
)
point_data = growth_study_area.growth.isel(eta_rho=eta_idx, xi_rho=xi_idx, years=year_idx, days= growth_study_area.coords['days'].values.tolist().index(day_idx))
sc = ax.scatter(point_data.lon_rho.item(), point_data.lat_rho.item(), color='red', marker='*', s=200, transform=ccrs.PlateCarree(), edgecolor='red', zorder=3)

# Map features
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
ax.set_facecolor('lightgrey')
# Sectors
ax.plot([-90, -90], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Atlantic sector
ax.plot([120, 120], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Pacific sector
ax.plot([0, 0], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Indian sector
# Adding coords
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
plt.suptitle("")
plt.tight_layout()
plt.show()
# pcolormesh.set_rasterized(True)
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/mhw_impacts/location_length_mhw_{target_start_year}.pdf'), dpi =150, format='pdf', bbox_inches='tight')

#%% Plot temperature and chla time series for the 'extreme' cell
#  === Extract time series for the selected grid cell ===
temp_timeseries = temp_avg_100m_1season.avg_temp[:, eta_idx, xi_idx]
chla_timeseries = chla_surf_1season.raw_chla[:, eta_idx, xi_idx]
days = temp_avg_100m_1season['avg_temp'].days.values
days_xaxis = np.where(days < 304, days+ 365, days).astype(int)
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0+365, 121+365)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)

# === Create subplots ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[1, 1], gridspec_kw={'hspace': 0.1})

# --- Plot temperature ---
ax1.plot(days_xaxis, temp_timeseries, color='darkorange', label='Temperature [°C]')
ax1.set_ylabel("Temp [°C]", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=12)
ax1.legend(fontsize=12)

# --- Plot chlorophyll-a ---
ax2.plot(days_xaxis, chla_timeseries, color='seagreen', label='Chla [mg Chl-a/m³]')
ax2.set_ylabel("Chla [mg/m³]", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=12)

# Define labels to keep for the shared x-axis
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []
for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)

ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, fontsize=14)  # bigger x-axis tick labels

# Improve layout
fig.suptitle(f"Inputs of growth for the MHW cell — {target_start_year}-{target_start_year+1}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_impactMHWs/mhw_cell_temp_chla_timeseries.pdf'), dpi =150, format='pdf', bbox_inches='tight')
# %%

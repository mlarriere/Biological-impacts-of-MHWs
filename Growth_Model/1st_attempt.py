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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import time

from joblib import Parallel, delayed

# %% -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

# file_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/temp_DC_BC_surface.nc' # drift and bias corrected temperature files
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
path_growth = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model'

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
pmod = 'perc' + str(percentile)


# %% Defining constants
# Assumption: in first attempt we use constant food and temperature, 

# Surface value from 21 sites (Scotia Sea)
# food= np.mean([6.5, 2.9, 2.9, 2.6, 10,10,10,0.22,0.22,0.97,2.7,1.2,6.3,3.3,1.2,1.9,2.5,2.1,0.072,0.096,0.17,0.15,0.57,0.66, 0.92, 1.1, 0.57, 0.55, 0.98, 0.2, 0.13, 0.2, 0.14, 0.090, 0.065, 0.84, 0.6, 0.18, 12, 5.4])

# ---- Constants, i.e. Coefficients of models predicting DGR and GI from length, food, and temperature in Eq. 4 (Atkinson et al., 2006), Here we use model4, i.e. sex and maturity considered
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

# %% Load data
# Temperature [°C]
det_combined_ds = xr.open_dataset(os.path.join(path_det, f"det_rel_abs_combined.nc")) #detected event (SST>abs and rel threshold) - boolean
temp_avg_100m = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_yearly_60S.nc'))['avg_temp'] #Averaged temperature of the first 100m - lot of Nan values (land)
# temp_avg_100m.isel(years=39, days=230).plot()

# Chla from ROMS [mh Chla/m3]
ds_chla= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_avg100m_yearly_60S.nc')) 
ds_chla = ds_chla.rename({'year': 'years'}) # Rename dimension
# chla.raw_chla.isel(year=30, time=30).plot()

# Reformating - stacking time dimension #shape (231, 1442, 14600)
temp_100m_stack = temp_avg_100m.stack(time= ['years', 'days'])
chla_stack = ds_chla.stack(time= ['years', 'days']) 

# %% Associating temperature occurring during detected events
# det_temp_surf = surf_temp.where(det_combined_ds) #extremely long computing!! 
# det_temp_surf.to_netcdf(path_det+'det_sst_combined.nc', mode='w')



# %% Growth model - Eq. 4 (Atkinson et al. 2006)
growth_da = a + b* length + c *length**2 + (d*chla_stack.raw_chla)/(e+chla_stack.raw_chla) + f*temp_100m_stack + g*temp_100m_stack**2 #[mm] - predicted daily (computing time ~40min)
growth_redimensioned = growth_da.unstack('time')
growth_redimensioned.isel(years=1991-1980, days=349-1).plot() 
growth_redimensioned.isel(years=2010-1980, days=35-1).plot() 


# Find min and max
flat_index_max = np.nanargmax(growth_redimensioned.values)
flat_index_min = np.nanargmin(growth_redimensioned.values)
max_pos = np.unravel_index(flat_index_max, growth_da.shape)  # (eta, xi, time)
min_pos = np.unravel_index(flat_index_min, growth_da.shape)  # (eta, xi, time)

max_eta, max_xi = max_pos[0], max_pos[1]
min_eta, min_xi = min_pos[0], min_pos[1]
year_max, day_max = growth_da.coords['years'].values[max_pos[2]], growth_da.coords['days'].values[max_pos[2]]
year_min, day_min = growth_da.coords['years'].values[min_pos[2]], growth_da.coords['days'].values[min_pos[2]]

print(f"📈 Max growth in SO at (eta_rho={max_eta}, xi_rho={max_xi}): {growth_da.values[max_pos]} mm")
print(f'Associated SST and Chla: {temp_100m_stack.values[max_pos]}°C and {chla_stack.raw_chla.values[max_pos]} mgChla/m3')
print(f"Date: {year_max}, Day {day_max}\n")

print(f"📉 Min growth in SO at (eta_rho={min_eta}, xi_rho={min_xi}): {growth_da.values[min_pos]} mm")
print('Associated SST and Chla:', temp_100m_stack.values[min_pos], "°C and ", chla_stack.raw_chla.values[min_pos], "mgChla/m3")
print(f"Date: {year_min}, Day {day_min}\n")


# Write to file
growth_ds =xr.Dataset(
    {"growth_redimensioned": (["eta_rho", "xi_rho", "years", "days"], growth_redimensioned.data)},
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lon_rho.values),
        lat_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lat_rho.values),
        years=np.arange(1980,2020),
        days = np.arange(1,366)
    ),
    attrs={"description": "Growth of krill based on Atkinson et al (2006) equation, model4 (sex and maturity considered)"}
)

growth_ds.to_netcdf(path=os.path.join(path_growth, "growth_1st_attempt.nc"), mode='w')

# %% Influence of SST and Chla on growth - 2D Histogram
growth_ds = xr.open_dataset(os.path.join(path_growth, "growth_1st_attempt.nc"))

# === Scatter plot - 1 location ===
eta=220
xi=950
plt.figure(figsize=(6, 6))
sc = plt.scatter(x= temp_100m_stack.isel(eta_rho=eta, xi_rho=xi).values, 
                 y= chla_stack.raw_chla.isel(eta_rho=eta, xi_rho=xi).values, 
                 c= growth_ds.growth_redimensioned.isel(eta_rho=eta, xi_rho=xi).values, 
                 cmap='plasma', s=10, alpha=0.7)
plt.colorbar(sc, label='Growth [mm]')
plt.xlabel('Temperature (°C)')
plt.ylabel('Chlorophyll-a (mg/m³)')
lat=np.unique(temp_100m_stack.isel(eta_rho=eta, xi_rho=xi).lat_rho)[0]
lon=np.unique(temp_100m_stack.isel(eta_rho=eta, xi_rho=xi).lon_rho)[0]
plt.title(f'Growth in location ({np.round(lat)}°S, {np.round(lon)}°E) \nover 1980-2019 period')
# plt.axis('equal')
plt.tight_layout()
plt.show()

# === 2D Histogram - full SO ===
# Flatten
x = temp_100m_stack.values.flatten()
y = chla_stack.raw_chla.values.flatten()
z = growth_ds.growth_redimensioned.values.flatten()

# Filter NaNs
mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z))
x, y, z = x[mask], y[mask], z[mask]

# Bin the data
from scipy.stats import binned_statistic_2d
stat, xedges, yedges, binnumber = binned_statistic_2d(x, y, z, statistic='mean', bins=[40, 40])

# 2D histogram
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, stat.T, cmap='plasma', shading='auto')
plt.colorbar(label='Growth [mm]')
plt.xlabel('Temperature [°C]')
plt.ylabel('Chl-a [mg/m³]')
plt.title('E. Superba Growth in the Southern Ocean (1980-2019) \n Atkison et al. (2006) model')
plt.tight_layout()
plt.show()



#%%  Selecting 1 location for plotting
choice_eta = 220  #190, 200
choice_xi = 950  #600, 1000

# Temperature, Chlorophyll and Growth
temp_100m_stack_selected_location = temp_100m_stack.isel(eta_rho=choice_eta, xi_rho=choice_xi)
chla_stack_selected_location = chla_stack.isel(eta_rho=choice_eta, xi_rho=choice_xi)
growth_selected_location = growth_ds.isel(eta_rho=choice_eta, xi_rho=choice_xi)

# Find detected events (SST > 90th perc and i°C) in selected location and remove Nans
det_temp_surf = xr.open_dataset(path_det+'det_sst_combined.nc') #sst values of extreme events (computed above)
det_selected_location = det_temp_surf.isel(years=slice(0,40), eta_rho=choice_eta, xi_rho=choice_xi)

# Reformat
det_selected_location_stack = det_selected_location.stack(time= ['years', 'days'])
growth_selected_location_stack = growth_selected_location.stack(time= ['years', 'days'])

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
lns1 = ax.plot(growth_selected_location_stack.growth_redimensioned, '-', color='#3E6F8E')
lns2 = ax2.plot(temp_100m_stack_selected_location,  '-', color='black')

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

# x-axis
ax.tick_params(axis='x', colors='black')
ticks_years = np.arange(0, 40*365, 365)
ax.set_xticks(ticks_years)  
tick_labels = np.arange(1980, 2020)
ax.set_xticklabels(tick_labels)
ax.set_xlabel("Time (days)")
ax.set_xlim(0*365, 20*365) #2014-2019
# ax.set_xlim(35*365+150, 36*365) #above 1°C
# ax.set_xlim(36*365+150, 37*365) #above 3°C
# ax.set_xlim(37*365-50, 37*365+150) #above 3°C

plt.title(f'Growth for Antarctic krill \n location: ({np.int32(np.round(temp_100m_stack_selected_location.lat_rho.values))}°S, {np.int32(np.round(temp_100m_stack_selected_location.lon_rho.values))}°E)')
plt.tight_layout()
plt.show()

# %% Temperature VS Chla Contribution


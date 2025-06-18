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
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
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

import sys
sys.path.append(working_dir+'Growth_Model') 
from growth_model import growth_Atkison2006  # import growth function

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

# %% Load data
# ==== Temperature [°C] -- Weighted averaged temperature of the first 100m - Austral summer - 60S
temp_avg_100m = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 181, 231, 1442)
temp_avg_100m = temp_avg_100m.rename({'year': 'years'})
# temp_avg_100m.avg_temp.isel(years=30, days=30).plot()

# ==== Chla [mh Chla/m3] -- Weighted averaged chla of the first 100m - Austral summer - 60S
chla_surf= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears.nc')) 
chla_surf = chla_surf.rename({'year': 'years'})
# chla_avg_100m.raw_chla.isel(years=30, days=30).plot()

# Reformating - stacking time dimension #shape (231, 1442, 7240)
temp_100m_stack = temp_avg_100m.stack(time= ['years', 'days'])
chla_surf_stack = chla_surf.stack(time= ['years', 'days']) 


# %% Prepare data
temp1 = temp_avg_100m.isel(years=37, days=slice(304,365))
temp2 = temp_avg_100m.isel(years=37, days=slice(0,120))
temp = xr.concat([temp1, temp2], dim='days')
temp_jan2017 = temp.sel(days=slice(0,30)).mean(dim='days') #shape: (eta_rho, xi_rho) : (231, 1442)

chla1 = chla_surf.isel(years=37, days=slice(304,365))
chla2 = chla_surf.isel(years=37, days=slice(0,120))
chla = xr.concat([chla1, chla2], dim='days')
chla_jan2017 = chla.sel(days=slice(0,30)).mean(dim='days') #shape: (eta_rho, xi_rho) : (231, 1442)

# %% Investigate chla 
# Extract data and flatten, excluding NaNs
data = chla_jan2017.raw_chla.values.flatten()
# data = temp_jan2017.avg_temp.values.flatten()
data = data[~np.isnan(data)]

# Get min and max
data_min = np.min(data)
data_max = np.max(data)

print(f"Chla min: {data_min:.4f} mg/m³")  # °C")
print(f"Chla max: {data_max:.4f} mg/m³")  # °C")

# Plot histogram
plt.figure(figsize=(6, 4))
plt.hist(data, bins=50, color='seagreen', edgecolor='black') #seagreen or darkred
# plt.title('Histogram of Temperature \n(January 2017)', fontsize=14)
plt.title('Histogram of Chla concentration \n(January 2017)', fontsize=14)
plt.xlabel('Chla [mg/m3]', fontsize=12)
# plt.xlabel('Temperature [°C]', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot ROMS outputs
norm_chla=mcolors.Normalize(vmin=0, vmax=2)

# ---- Figure and Axes ----
fig_width, fig_height = 6, 5
fig = plt.figure(figsize=(fig_width *2, fig_height))
gs = gridspec.GridSpec(1, 2, wspace=0.08, hspace=0.2)

axs = []
for j in range(2): 
    ax = fig.add_subplot(gs[0, j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    axs.append(ax)

# ---- Plot Definitions ----
plot_data = [
    (temp_jan2017.avg_temp, "Mean Temperature (100m)", 'inferno', None),
    (chla_jan2017.raw_chla, "Chlorophyll-a (5m)", 'YlGn', norm_chla)
]

ims = []
for i, (data, title, cmap, norm) in enumerate(plot_data):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    import matplotlib.path as mpath
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Coastlines and land
    ax.coastlines(color='black', linewidth=1)
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=1, zorder=5)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=2)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Plot data if available
    if data is not None:
        lon_name = [name for name in data.coords if 'lon' in name][0]
        lat_name = [name for name in data.coords if 'lat' in name][0]

        im = ax.pcolormesh(
            data[lon_name], data[lat_name], data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading='auto',
            rasterized=True
        )
        ims.append(im)
    else:
        ims.append(None)

    ax.set_title(title, fontsize=14)

# ---- Colorbars ----
# Temperature colorbar
cbar_ax1 = fig.add_axes([0.13, 0.01, 0.32, 0.03])  # [left, bottom, width, height]
cbar1 = fig.colorbar(ims[0], cax=cbar_ax1, orientation='horizontal', extend='both')
cbar1.set_label("°C", fontsize=12)

# Chlorophyll colorbar
cbar_ax2 = fig.add_axes([0.56, 0.01, 0.32, 0.03])
cbar2 = fig.colorbar(ims[1], cax=cbar_ax2, orientation='horizontal', extend='both')
cbar2.set_label("mg/m³", fontsize=12)


plt.suptitle("ROMS outputs \n January 2017", fontsize=18, y=1.05)
plt.show()



# %% ========== Hypothetical Growth ==========
# Investigate max/min values
max_obs_chla = chla_surf_stack.raw_chla.max() #5 mg/m3 Chla
min_obs_chla = chla_surf_stack.raw_chla.min() #0 mg/m3
max_obs_temp = temp_100m_stack.avg_temp.max() #7.3°C
min_obs_temp = temp_100m_stack.avg_temp.min() #-5.03°C

# 2D grid of all combinations
chla_hyp= np.arange(min_obs_chla, 5.05, 0.01)
temp_hyp = np.arange(min_obs_temp, max_obs_temp, 0.01)

# Calcualting growth for all combination
CHLA, TEMP = np.meshgrid(chla_hyp, temp_hyp)
growth_hyp = growth_Atkison2006(CHLA, TEMP)#[mm] array of shape: (1234, 505)

# %% Growth model - Eq. 4 (Atkinson et al. 2006)
file_growth = os.path.join(path_growth, "growth_Atkison2006_fullyr.nc")
if not os.path.exists(file_growth):
    # Calculating growth
    growth_da = growth_Atkison2006(chla_surf_stack.raw_chla, temp_100m_stack.avg_temp) #[mm]
    growth_redimensioned = growth_da.unstack('time')

    # Write to file
    growth_ds =xr.Dataset(
        {"growth": (["eta_rho", "xi_rho", "years", "days"], growth_redimensioned.data)},
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lon_rho.values),
            lat_rho=(["eta_rho", "xi_rho"], growth_redimensioned.lat_rho.values),
            years = (['years'], growth_redimensioned.years.values),
            days = (['days'], growth_redimensioned.days.values),
            depth= '5'
        ),
        attrs={"description": "Growth of krill based on Atkinson et al (2006) equation, model4 (sex and maturity considered) -- FULL YEAR"}
    )

    growth_ds.to_netcdf(file_growth, mode='w')
else:
    # Load data
    growth_redimensioned = xr.open_dataset(file_growth)

# %% Defining seasonal extent for growth (austral summer - early spring)
def defining_season(ds, starting_year):
    # Get the day-of-year slices
    days_nov_dec = ds.sel(days=slice(304, 364), years=starting_year)
    days_jan_apr = ds.sel(days=slice(0, 119), years=starting_year + 1)

    # Combine while keeping doy as coords
    ds_season = xr.concat([days_nov_dec, days_jan_apr], dim=xr.DataArray(np.concatenate([days_nov_dec['days'].values, days_jan_apr['days'].values]), dims="days", name="days"))

    return ds_season

# %% ======================== Growth in yearly context - 1 plot ========================
# Target year: start in July 1980 -> so need Jul-Dec 1980 and Jan-Jun 1981
target_start_year = 2010
target_end_year = target_start_year + 1

# --- Full year data ---
# Extract data across July–June (full year across two calendar years)
CHLA_jul_dec = chla_surf.raw_chla.sel(years=target_start_year).isel(days=slice(181, 365)).mean(dim=["eta_rho", "xi_rho"])#, eta_rho=200, xi_rho=1000)
TEMP_jul_dec = temp_avg_100m.avg_temp.sel(years=target_start_year).isel(days=slice(181, 365)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

CHLA_jan_jun = chla_surf.raw_chla.sel(years=target_end_year).isel(days=slice(0, 181)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)
TEMP_jan_jun = temp_avg_100m.avg_temp.sel(years=target_end_year).isel(days=slice(0, 181)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

# Concatenate to get full July–June year
CHLA_full = np.concatenate([CHLA_jul_dec, CHLA_jan_jun])
TEMP_full = np.concatenate([TEMP_jul_dec, TEMP_jan_jun])

# --- Growth season data ---
# Identify Nov 1 – Apr 30 for the red line (Nov–Dec from current year, Jan–Apr from next year)
CHLA_nov_dec = chla_surf.raw_chla.sel(years=target_start_year).isel(days=slice(305, 365)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)
TEMP_nov_dec = temp_avg_100m.avg_temp.sel(years=target_start_year).isel(days=slice(305, 365)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

CHLA_jan_apr = chla_surf.raw_chla.sel(years=target_end_year).isel(days=slice(0, 121)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)
TEMP_jan_apr = temp_avg_100m.avg_temp.sel(years=target_end_year).isel(days=slice(0, 121)).mean(dim=["eta_rho", "xi_rho"]) #, eta_rho=200, xi_rho=1000)

CHLA_season = np.concatenate([CHLA_nov_dec.values, CHLA_jan_apr.values])
TEMP_season = np.concatenate([TEMP_nov_dec.values, TEMP_jan_apr.values])

# ============ Plot ============
fig, ax = plt.subplots(figsize=(10, 6))
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

# Background color map
pcm = ax.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto', cmap='coolwarm_r', norm=norm, rasterized= True)
contours = ax.contour(CHLA, TEMP, growth_hyp, levels=20, colors='white', linewidths=0.5)
ax.clabel(contours, inline=True, fontsize=12, fmt="%.2f")

# --- Full year growth path -- no need to calcualte the growth just plot the dot (temp, chla) 
valid_mask_full = ~np.isnan(CHLA_full) & ~np.isnan(TEMP_full) # Mask NaNs for full path
CHLA_full_clean = CHLA_full[valid_mask_full]
TEMP_full_clean = TEMP_full[valid_mask_full]
ax.plot(CHLA_full_clean, TEMP_full_clean, color='#4D7C8A', linewidth=2, label='Growth Path (Jul–Jun)') 

# --- Growth season (Nov–Apr)
valid_mask_season = ~np.isnan(CHLA_season) & ~np.isnan(TEMP_season) # Mask NaNs for full path
CHLA_season_clean = CHLA_season[valid_mask_season]
TEMP_season_clean = TEMP_season[valid_mask_season]
ax.plot(CHLA_season_clean, TEMP_season_clean, color="#643888", linewidth=2.5, label='Growth Season (Nov–Apr)') #plot clean data

# --- Scatter start and end dates 
# Date of all datapoints
from datetime import datetime, timedelta
jul1 = datetime(target_start_year, 7, 1) # Starting date
dates_full = [jul1 + timedelta(days=int(i)) for i in range(len(CHLA_full))]
dates_valid = np.array(dates_full)[valid_mask_full] #without nans

# Start and end date = first and last VALID points
start_date_str = dates_valid[0].strftime('%d %b')
end_date_str = dates_valid[-1].strftime('%d %b')

ax.scatter(CHLA_full_clean[0], TEMP_full_clean[0], facecolor='white', edgecolor='black',
           s=100, zorder=20, label=f'Start ({start_date_str})')
ax.scatter(CHLA_full_clean[-1], TEMP_full_clean[-1], facecolor='black', edgecolor='white',
           s=100, zorder=20, label=f'End ({end_date_str})')

# Axis settings
ax.set_title(f'Krill Growth from 1st July {target_start_year} to 30th June {target_end_year}', fontsize=18)
ax.set_xlabel('Chlorophyll-a [mg/m³]', fontsize=16)
ax.set_ylabel('Temperature [°C]', fontsize=16)
ax.set_xlim(0, 2)
ax.set_ylim(-5, 5)
ax.set_yticks(np.arange(-5, 6, 1))
ax.tick_params(labelsize=13)
ax.legend(fontsize=13)

# Colorbar
cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
cbar.set_label('Growth [mm]', fontsize=16)
cbar.ax.tick_params(labelsize=13)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/temp_chla_diagrams/{target_start_year}_combined.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %% ======================== Growth in yearly context - 2 subplots ========================
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_hyp), vcenter=0, vmax=np.nanmax(growth_hyp))

# Define the starting year for the growth season (Nov–Apr)
target_start_year = 1980
year_to_plot = target_start_year - 1980

fig = plt.figure(figsize=(16, 6))

# ========== FULL YEAR ==========
ax1 = fig.add_subplot(1, 2, 1)

# Extract CHLA and TEMP for the full year
CHLA_full = chla_surf.raw_chla.isel(years=year_to_plot, xi_rho=1000, eta_rho=200)
TEMP_full = temp_avg_100m.avg_temp.isel(years=year_to_plot, xi_rho=1000, eta_rho=200)

pcm1 = ax1.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto', cmap='coolwarm_r', norm=norm)
contours1 = ax1.contour(CHLA, TEMP, growth_hyp, levels=20, colors='w', linewidths=0.5)
ax1.clabel(contours1, inline=True, fontsize=12, fmt="%.2f")
ax1.plot(CHLA_full, TEMP_full, color='#4D7C8A', linewidth=2, label='Krill Growth Path')
ax1.scatter(CHLA_full[0], TEMP_full[0], color='white', edgecolor='black', s=100, zorder=10, label='Start (1st Jan)')
ax1.scatter(CHLA_full[-1], TEMP_full[-1], color='black', s=100, zorder=10, label='End (31st Dec)')
ax1.set_title(f'Full Year {year_to_plot+1980}', fontsize=18)
ax1.set_xlabel('Chlorophyll-a [mg/m³]', fontsize=16)
ax1.set_ylabel('Temperature [°C]', fontsize=16)
ax1.set_xlim(0, 2)
ax1.set_ylim(-5, 5)
ax1.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2 , 3, 4, 5])
# ax1.set_xticks([0, 1, 2 , 3, 4, 5])
ax1.legend(fontsize=13)
ax1.tick_params(labelsize=13)

# ========== GROWTH SEASON ==========
# Calling function
# growth_1season = defining_season(growth_redimensioned, target_start_year)
chla_1season = defining_season(chla_surf, target_start_year)
temp_1season = defining_season(temp_avg_100m, target_start_year)

ax2 = fig.add_subplot(1, 2, 2)

CHLA_season = chla_1season.raw_chla.isel(xi_rho=1000, eta_rho=200)
TEMP_season = temp_1season.avg_temp.isel(xi_rho=1000, eta_rho=200)

pcm2 = ax2.pcolormesh(CHLA, TEMP, growth_hyp, shading='auto', cmap='coolwarm_r', norm=norm)
contours2 = ax2.contour(CHLA, TEMP, growth_hyp, levels=20, colors='w', linewidths=0.5)
ax2.clabel(contours2, inline=True, fontsize=12, fmt="%.2f")
ax2.plot(CHLA_season, TEMP_season, color='#4D7C8A', linewidth=2, label='Krill Growth Path')
ax2.scatter(CHLA_season[0], TEMP_season[0], color='white', edgecolor='black', s=100, zorder=10, label='Start (1st Nov)')
ax2.scatter(CHLA_season[-1], TEMP_season[-1], color='black', s=100, zorder=10, label='End (30th Apr)')
ax2.set_title(f'Growth Season {target_start_year}-{target_start_year+1}', fontsize=18)
ax2.set_xlabel('Chlorophyll-a [mg/m³]', fontsize=16)
ax2.set_ylabel('Temperature [°C]', fontsize=16)
ax2.set_xlim(0, 2)
ax2.set_ylim(-5, 5)
ax2.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2 , 3, 4, 5])
# ax2.set_xticks([0, 1, 2 , 3, 4, 5])
ax2.legend(fontsize=13)
ax2.tick_params(labelsize=13)

# Shared colorbar
cbar_ax = fig.add_axes([1.02, 0.1, 0.013, 0.84])  # [left, bottom, width, height]
cbar = fig.colorbar(pcm2, cax=cbar_ax)
cbar.set_label('Growth [mm]', fontsize=16)
cbar.ax.tick_params(labelsize=13)
plt.tight_layout(w_pad=4.0)  
plt.show()
# Converts all plot elements to raster inside the PDF --> reducing size while keeping the vector type
# pcm1.set_rasterized(True)
# pcm2.set_rasterized(True)
# for i in contours1.collections: c.set_rasterized(True)
# for i in contours2.collections: c.set_rasterized(True)
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/temp_chla_diagrams/{year_to_plot+1980}.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %% Select growth season - all years
def define_season_all_years(ds):
    # ds = growth_redimensioned
    all_years = ds['years'].values
    season_list = []
    all_years_set = set(all_years)
    
    for y in all_years: #about 40s per year
        print(y)
        if (y + 1) in all_years_set:  
        # Verify if next year exist - otherwise incomplete season
            try: 
                days_nov_dec = ds.sel(days=slice(304, 364), years=y)
                days_jan_apr = ds.sel(days=slice(0, 119), years=y + 1)

                # Combine days
                season = xr.concat(
                    [days_nov_dec, days_jan_apr],
                    dim=xr.DataArray(np.concatenate([days_nov_dec['days'].values, 
                                                     days_jan_apr['days'].values]),
                                                     dims="days", 
                                                     name="days"))

                season = season.expand_dims(season_year=[y])  # tag with season start year
                season_list.append(season)

            except Exception as e: # Skip incomplete seasons
                print(f"Skipping year {y} due to error: {e}")
                continue

    if not season_list:
        raise ValueError("No valid seasons found.")

    # Concatenate
    return xr.concat(season_list, dim="season_year")

growth_seasons = define_season_all_years(growth_redimensioned) 
growth_seasons = growth_seasons.rename({'season_year': 'years', 'years': 'day_years'})
growth_seasons.attrs['description'] = ("Krill growth estimates during the growth season (Nov 1 – Apr 30) based on Atkinson et al. (2006), model 4")
growth_seasons.to_netcdf(path=os.path.join(path_growth, "growth_Atkison2006_seasonal.nc"), mode='w')

chla_filtered_seasons = define_season_all_years(chla_surf)
chla_filtered_seasons = chla_filtered_seasons.rename({'season_year': 'years', 'years': 'day_years'})
temp_avg_100m_seasons = define_season_all_years(temp_avg_100m)
temp_avg_100m_seasons = temp_avg_100m_seasons.rename({'season_year': 'years', 'years': 'day_years'})

# %% Plot growth season for 1 year
growth_seasons2017 = growth_seasons.isel(years=37)

# Select January
growth_ROMS_2017_jan = growth_seasons2017.sel(days=slice(0,30)).mean(dim='days') #shape: (eta_rho, xi_rho) : (231, 1442)

# Setup figure
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width
fig = plt.figure(figsize=(fig_width, fig_height))  # Only 1 plot now
gs = gridspec.GridSpec(1, 1)

axs = []
ax = fig.add_subplot(gs[0, 0], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
axs.append(ax)

# Plot configuration
plot_data = [
    (growth_ROMS_2017_jan.growth, "Mean Growth")
]

# Color normalization
vmin, vmax = -0.2, 0.2
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
cmap = 'PuOr_r'

# -----------------------------
# Plotting
# -----------------------------
ims = []
for data, title in plot_data:
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map features
    ax.coastlines(color='black', linewidth=1, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=1, zorder=5)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=2)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot data
    im = ax.pcolormesh(
        growth_seasons2017.lon_rho,
        growth_seasons2017.lat_rho,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading='auto',
        zorder=1,
        rasterized=True
    )
    ims.append(im)
    # ax.set_title(title, fontsize=15)

# -----------------------------
# Colorbar
# -----------------------------
cbar_ax = fig.add_axes([0.25, 0.01, 0.5, 0.03])  # [left, bottom, width, height]
tick_positions = [-0.2, -0.1, 0.0, 0.1, 0.2]
cbar = fig.colorbar(
    ims[0], cax=cbar_ax, orientation='horizontal',
    ticks=tick_positions, extend='both'
)
cbar.set_label("Growth [mm]", fontsize=14)
cbar.ax.tick_params(labelsize=13)

plt.suptitle("Growth with ROMS - January 2017", fontsize=18, y=0.98)
plt.show()


#%% == Equation decomposition
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
    
length=35 # mean body length in adult krill (Michael et al. 2021 / Tarling 2020)

# Food term
print(f'For maximum chla ({chla_filtered_seasons.raw_chla.max():.2f}mg/m3), in the eq result as {a+(d*chla_filtered_seasons.raw_chla.max())/(e+chla_filtered_seasons.raw_chla.max()):.2f}mm/d')
print(f'For minimum chla ({chla_filtered_seasons.raw_chla.min():.2f}mg/m3), in the eq result as {a+(d*chla_filtered_seasons.raw_chla.min())/(e+chla_filtered_seasons.raw_chla.min()):.2f}mm/d')
food_term = a + (d*chla_filtered_seasons.raw_chla)/(e+chla_filtered_seasons.raw_chla)

# Temperature term 
print(f'For maximum T°C ({temp_avg_100m_seasons.avg_temp.max():.2f}°C), in the eq result as {f*temp_avg_100m_seasons.avg_temp.max() +g * (temp_avg_100m_seasons.avg_temp.max())**2:.2f}mm/d')
print(f'For minimum T°C ({temp_avg_100m_seasons.avg_temp.min():.2f}°C), in the eq result as {f*temp_avg_100m_seasons.avg_temp.min() + g * (temp_avg_100m_seasons.avg_temp.min())**2:.2f}mm/d')
print(f'For optimum T°C (0.5°C), in the eq result as {f*0.5 + g * 0.5**2:.2f}mm/d')
temp_term = f*temp_avg_100m_seasons.avg_temp + g*temp_avg_100m_seasons.avg_temp**2

# Length term 
length_term = b* length + c *length**2
print(f'Length term: {length_term:.2f}mm for 35mm krill')

#%% Baseline vs warming periods
# === Averages
food_term_1980_2009_avg = food_term.isel(years=slice(0,30)).mean(dim=('years', 'days'))
temp_term_1980_2009_avg = temp_term.isel(years=slice(0,30)).mean(dim=('years', 'days'))
growth_1980_2009_avg = growth_seasons.isel(years=slice(0,30)).mean(dim=('years', 'days'))

food_term_2010_2019_avg = food_term.isel(years=slice(30, 40)).mean(dim=('years', 'days'))
temp_term_2010_2019_avg  = temp_term.isel(years=slice(30, 40)).mean(dim=('years', 'days'))
growth_2010_2019_avg  = growth_seasons.isel(years=slice(30, 40)).mean(dim=('years', 'days'))

# === Norm color
from matplotlib.colors import TwoSlopeNorm
food_min = min(food_term_1980_2009_avg.min().item(), food_term_2010_2019_avg.min().item()) #-0.2mm/d
food_max = max(food_term_1980_2009_avg.max().item(), food_term_2010_2019_avg.max().item()) #0.15 mm/d
norm_chla = TwoSlopeNorm(vmin=food_min, vcenter=0, vmax=food_max)
ticks_food = [-0.2, -0.1, 0.0, 0.1, 0.2]

temp_min = min(temp_term_1980_2009_avg.min().item(), temp_term_2010_2019_avg.min().item())
temp_max = max(temp_term_1980_2009_avg.max().item(), temp_term_2010_2019_avg.max().item())
norm_temp = mcolors.Normalize(vmin=temp_min, vmax=temp_max)
ticks_temp = [-0.2, -0.15, -0.1, -0.05, 0.0]

norm_growth = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
ticks_growth = [-0.2, -0.1, 0.0, 0.1, 0.2]

# %% == Plotting
import cmocean
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

# === Circle for south polar projection ===
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# === Titles and data for each subplot ===
datasets = [
    # 1980-2009 
    {'data': growth_1980_2009_avg.growth, 'cmap': 'PuOr_r', 'norm': norm_growth, 'label': '[mm/d]', 'title': 'Growth (1980–2009)'},
    {'data': food_term_1980_2009_avg, 'cmap': 'RdYlGn', 'norm': norm_chla, 'label': '[mm/d]', 'title': 'Food term (1980–2009)'},
    {'data': temp_term_1980_2009_avg, 'cmap': cmocean.cm.thermal, 'norm': norm_temp, 'label': '[mm/d]', 'title': 'Temperature term (1980–2009)'},
    # 2010-2019 
    {'data': growth_2010_2019_avg.growth, 'cmap': 'PuOr_r', 'norm': norm_growth, 'label': '[mm/d]', 'title': 'Growth (2010–2019)'},
    {'data': food_term_2010_2019_avg, 'cmap': 'RdYlGn', 'norm': norm_chla, 'label': '[mm/d]', 'title': 'Food term (2010–2019)'},
    {'data': temp_term_2010_2019_avg, 'cmap': cmocean.cm.thermal, 'norm': norm_temp, 'label': '[mm/d]', 'title': 'Temperature term (2010–2019)'}
]

# === Create figure and subplots ===
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width * 2 / 3  # adjust for 2 rows
fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height),
                        subplot_kw={'projection': ccrs.SouthPolarStereo()},
                        gridspec_kw={'hspace': 0.5, 'wspace': 0.4})

# Flatten axes for plotting
axs = axs.flatten()
pcms = []
for ax, ds in zip(axs, datasets):
    ax.set_boundary(circle, transform=ax.transAxes)
    pcm = ds['data'].plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        x='lon_rho',
        y='lat_rho',
        cmap=ds['cmap'],
        norm=ds['norm'],
        add_colorbar=False, 
        rasterized=True
    )
    ax.set_title(ds['title'], fontsize=10)
    ax.coastlines(color='black', linewidth=1.0, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')
    # ax.plot([-85, -85], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    # ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    # ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    pcms.append(pcm)


# === One shared horizontal colorbar per column ===
tick_list = [ticks_growth, ticks_food, ticks_temp]
# Get position of the top axes in each column to place colorbars beneath them
for col in range(3):
    # Position of top row subplot in this column
    pos_top = axs[col].get_position()
    # Position of bottom row subplot in this column
    pos_bottom = axs[col + 3].get_position()

    # Calculate the horizontal center of the column's subplots
    x_center = (pos_top.x0 + pos_top.x1) / 2

    # Define colorbar axes: centered under the two subplots, height small, width roughly the subplot width
    cbar_width = pos_top.width * 1.1
    cbar_height = 0.02
    cbar_x = x_center - cbar_width / 2
    cbar_y = pos_bottom.y0 - 0.1  # a bit below the bottom subplot

    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(pcms[col + 3], cax=cbar_ax, orientation='horizontal', ticks=tick_list[col])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(
        datasets[col]['label'],
        fontsize=11,
        labelpad=8
    )



# # === One shared colorbar per column ===
# cbar_positions = [
#     [0.34, 0.1, 0.01, 0.8],  # for column 0 (growth)
#     [0.62, 0.1, 0.01, 0.8],  # for column 1 (food)
#     [0.92, 0.1, 0.01, 0.8]   # for column 2 (temp)
# ]

# for i in range(3):
#     cbar_ax = fig.add_axes(cbar_positions[i])
#     cbar = fig.colorbar(pcms[i + 3], cax=cbar_ax, ticks=tick_list[i])
#     cbar.ax.tick_params(labelsize=10)
#     cbar.set_label(
#         datasets[i]['label'],
#         fontsize=11,
#         rotation=270,
#         labelpad=12,
#         verticalalignment='center'
#     )


plt.suptitle("Decomposition of the Growth Equation (Atkinson et al. 2006)", fontsize=14, y=1.1, x=0.52)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/eq_decomposition/equation_decomp_2periods.png'), dpi =200, format='png', bbox_inches='tight')

#%% ==== Periods comparison ====
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
        norm_main = mcolors.Normalize(vmin=0, vmax=5)
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
        ax.plot([-90, -90], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
        ax.plot([120, 120], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
        ax.plot([0, 0], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)

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
    # Converts all plot elements to raster inside the PDF --> reducing size while keeping the vector type
    # for sc in scs:
    #     sc.set_rasterized(True)

    # if 'contours1' in locals():
    #     for c in contours1.collections:
    #         c.set_rasterized(True)
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/eq_decomposition/{varname}_diff.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# Choose variable to plot
variable = 'growth'  #'growth', 'temp', 'chla'

if variable == 'growth':
    ds = growth_seasons.growth
    cmap_var = 'PuOr_r'
    label = 'Growth [mm/d]'
    ticks = [-0.2, -0.1, 0, 0.1, 0.2]

elif variable == 'temp':
    ds = temp_avg_100m_seasons.avg_temp
    cmap_var =  cmocean.cm.thermal  #'inferno'
    label = 'Temperature [°C]'
    ticks = [-6, -4, -2, 0, 2, 4, 6]

elif variable == 'chla':
    ds = chla_filtered_seasons.raw_chla
    cmap_var = cmocean.cm.algae 
    label = 'Chla [mg/m³]'
    ticks =  [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5 ,5] 

plot_comparison(variable, ds, cmap_var=cmap_var, ticks=ticks, cbar_label=label)


#%% ==== Decadal comparison ====
import matplotlib.ticker as mticker

def plot_variables_decades(growth_ds, chla_ds, temp_ds):
    fig, axs = plt.subplots(3, 4, figsize=(24, 16), subplot_kw={'projection': ccrs.SouthPolarStereo()})

    # Circular boundary for polar plot
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # Define decade slices (assumes 'years' dimension is ordered 1980...2019)
    decades = [(0, 10), (10, 20), (20, 30), (30, 40)]
    titles = ['1980-1989', '1990-1999', '2000-2009', '2010-2019']

    # Set color maps and norms for each variable
    growth_cmap = 'PuOr_r'
    growth_norm = mcolors.Normalize(vmin=-0.2, vmax=0.2)
    growth_ticks = [-0.2, -0.1, 0, 0.1, 0.2]

    chla_cmap = cmocean.cm.algae
    chla_norm = mcolors.Normalize(vmin=0, vmax=5)
    chla_ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    temp_cmap = cmocean.cm.thermal
    temp_norm = mcolors.Normalize(vmin=-6, vmax=6)
    temp_ticks = [-6, -4, -2, 0, 2, 4, 6]

    # Variable info for looping
    vars_info = [
        (growth_ds, growth_cmap, growth_norm, growth_ticks, 'Growth [mm/d]'),
        (chla_ds, chla_cmap, chla_norm, chla_ticks, 'Chla [mg/m³]'),
        (temp_ds, temp_cmap, temp_norm, temp_ticks, 'Temperature [°C]')
    ]

    for row, (ds, cmap, norm, ticks, label) in enumerate(vars_info):
        for col, (start, end) in enumerate(decades):
            ax = axs[row, col]
            data_decade = ds.isel(years=slice(start, end)).mean(dim=('years', 'days'))
            pcm = data_decade.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                x="lon_rho", y="lat_rho",
                add_colorbar=False,
                cmap=cmap,
                norm=norm,
                zorder=1
            )
            # Titles only on top row
            if row == 0:
                ax.set_title(titles[col], fontsize=16)

            # Y-axis label on left column
            if col == 0:
                ax.set_ylabel(label, fontsize=14)

            ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
            ax.set_boundary(circle, transform=ax.transAxes)
            ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
            ax.coastlines(color='black', linewidth=1)
            ax.set_facecolor('#F6F6F3')

            # Sector boundaries
            ax.plot([-90, -90], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
            ax.plot([120, 120], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
            ax.plot([0, 0], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)

        # Add a single colorbar to the right of each row (last column)
        pos = axs[row, -1].get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.015, pos.height])
        cbar = fig.colorbar(pcm, cax=cbar_ax, ticks=ticks)
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))

    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.08, wspace=0.2, hspace=0.15)
    plt.show()


plot_variables_decades(growth_seasons.growth,
                       chla_filtered_seasons.raw_chla,
                       temp_avg_100m_seasons.avg_temp)


# %% Growth during MHW events
# ==== MHWs events detected -- only surface
mhw_det = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))) #shape: (40, 181, 231, 1442)
growth_seasons= xr.open_dataset(os.path.join(path_growth, "growth_Atkison2006_seasonal.nc"))

# ---------------- MHWs
# Write or load
growth_mhw_file= os.path.join(path_growth, "growth_Atkison2006_mhw.nc")

if not os.path.exists(growth_mhw_file):

    variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
    growth_var_names = ['growth_1deg', 'growth_2deg', 'growth_3deg', 'growth_4deg']

    growth_mhw_dict = {}

    for mhw_var, growth_name in zip(variables, growth_var_names):
        print(f'------------{mhw_var}------------')
        # testing
        # var='det_1deg'

        # MHW mask for current threshold and align MHW years to match growth_seasons (1980-2018)
        duration_mask = mhw_det['duration'].sel(years=growth_seasons.years) >= 30 #bool
        det_mask = mhw_det[mhw_var].sel(years=growth_seasons.years) == 1 #bool

        # Ensure both are boolean, then combine
        mhw_mask = duration_mask & det_mask

        # Growth during MHWs
        growth_masked = growth_seasons['growth'].where(mhw_mask)

        # Store DataArray to dictionnary
        growth_mhw_dict[growth_name] = growth_masked

        print(f'done with {growth_name}')

    # To Dataset
    growth_mhw_combined = xr.Dataset(
        data_vars=growth_mhw_dict,
        coords=growth_seasons.coords,
        attrs={
            "description": "Growth of krill during MHWs for the growth season (Nov 1 – Apr 30).",
            "depth": "5m.",
            "growth_ideg": (
                "Growth under different MHW intensity "
                "(MHWs defined as exceeding both the extended absolute threshold and the 90th percentile as well as lasting more than 30days)."
            )
        }
    )

    # Save output
    growth_mhw_combined.to_netcdf(growth_mhw_file, mode='w')
    
else:
    # Load data
    growth_mhw_combined = xr.open_dataset(growth_mhw_file)

# ---------------- NO MHWs
# Write or load
growth_no_mhw_file = os.path.join(path_growth, "growth_Atkison2006_nonMHW.nc")

if not os.path.exists(growth_no_mhw_file):

    # Align mask years
    mhw_mask = mhw_det['det_1deg'].sel(years=growth_seasons.years)

    # Invert mask to select growth when no MHWs
    growth_no_mhw = growth_seasons['growth'].where(mhw_mask == 0)

    # To Dataset
    growth_no_mhw_ds = xr.Dataset(
        data_vars={"growth_noMHW": growth_no_mhw},
        coords=growth_seasons.coords,
        attrs={
            "description": "Krill growth during **non-MHW** periods across the growth season (Nov 1 – Apr 30).",
            "depth": "5m",
            "growth_noMHW": (
                "Growth when no MHWs are detected (MHWs defined as exceeding extended absolute threshold + 90th percentile)."
            )
        }
    )

    # Save output
    growth_no_mhw_ds.to_netcdf(growth_no_mhw_file, mode='w')

else:
    growth_no_mhw_ds= xr.open_dataset(growth_no_mhw_file)


# %% Computing the mean values
# Mean value or 1980-2018
growth_no_mhw_mean = growth_no_mhw_ds.mean(dim=('years','days'))
growth_mhw_mean = growth_mhw_combined.mean(dim=('years','days'))


# %% -------------------- Plot Growth under MHWs --------------------
# -----------------------------
# Plot setup
# -----------------------------
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width
fig = plt.figure(figsize=(fig_width * 5, fig_height))  # Adjust if you add more plots
gs = gridspec.GridSpec(1, 5, wspace=0.1, hspace=0.2)

axs = []
for j in range(5): 
    ax = fig.add_subplot(gs[0, j], projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    axs.append(ax)

# -----------------------------
# Plot data setup
# -----------------------------
plot_data = [
    (growth_mhw_mean.growth_1deg, r"MHWs $>$ 1$^\circ$C"),
    (growth_mhw_mean.growth_2deg, r"MHWs $>$ 2$^\circ$C"),
    (growth_mhw_mean.growth_3deg, r"MHWs $>$ 3$^\circ$C"),
    (growth_mhw_mean.growth_4deg, r"MHWs $>$ 4$^\circ$C"),
    (growth_no_mhw_mean.growth_noMHW, r"No MHWs")

]

from matplotlib import colors
vmin, vmax = -0.2, 0.2
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

# -----------------------------
# Loop over plots
# -----------------------------
for i, (data, title) in enumerate(plot_data):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Map features
    ax.coastlines(color='black', linewidth=1, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Sector lines
    for lon_line in [-90, 0, 120]:
        ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
                color="#080808", linestyle='--', linewidth=1, zorder=5)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=2)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot data
    im = ax.pcolormesh(
        data.lon_rho,
        data.lat_rho,
        data,
        transform=ccrs.PlateCarree(),
        cmap='PuOr_r',
        norm=norm,
        shading='auto',
        zorder=1,
        rasterized=True
    )

    ax.set_title(title, fontsize=15)

# -----------------------------
# Common colorbar
# -----------------------------
tick_positions = [-0.2, -0.1, 0.0, 0.1, 0.2]
cbar = fig.colorbar(
    im, ax=axs, orientation='horizontal',
    fraction=0.09, pad=0.1, 
    ticks=tick_positions, 
    extend='both'
)
cbar.set_label("Mean growth [mm]", fontsize=14)
cbar.ax.tick_params(labelsize=13) 

plt.suptitle("Average krill growth under different MHW intensities \n Growth season (1Nov- 30Apr) for 1980–2018", fontsize=20, y=1.1)
# plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

# %%

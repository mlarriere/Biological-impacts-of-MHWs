#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 17 March 08:55:12 2025

Detect MHW in ROMS-SO using absolute threshold

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

path_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'

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

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold
percentile = 90 
pmod = 'perc' + str(percentile)


# -- Handling time
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]) #defining months with days within a year
month_names = np.array(['Jan','Feb','Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov','Dec'])

season_bins = np.array([0, 90, 181, 273, 365]) #defining seasons with days within a year
season_names = np.array(['DJF (Summer)', 'MAM (Fall)', 'JJA (Winter)', 'SON (Spring)']) #southern ocean!

# %% Combined relative and absolute thresholds
det_ds = xr.open_dataset(os.path.join('/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/', "det_all_eta.nc"))

# Dealing with Nans values -- No NaNs value in the dataset
# nans_rel = np.isnan(det_ds.mhw_rel_threshold).sum().item()#0
# nans_1deg = np.isnan(det_ds.mhw_abs_threshold_1_deg).sum().item()#0
# nans_2deg = np.isnan(det_ds.mhw_abs_threshold_2_deg).sum().item() #0
# nans_3deg = np.isnan(det_ds.mhw_abs_threshold_3_deg).sum().item() #0
# nans_4deg = np.isnan(det_ds.mhw_abs_threshold_4_deg).sum().item() #0

# Check of inconsistencies
print(det_ds.mhw_abs_threshold_1_deg.sum()>=det_ds.mhw_abs_threshold_2_deg.sum()>=det_ds.mhw_abs_threshold_3_deg.sum()>= det_ds.mhw_abs_threshold_4_deg.sum()) #True

# When 90th perc > certain temperature -> NOT NECESSARY OBSERVED
file_thresh = os.path.join(path_clim, f"threshold_90perc_surf.nc") # Relative threshold
relative_threshold_surf = xr.open_dataset(file_thresh)['relative_threshold']# shape:(day: 365, xi_rho: 1442)
det_1deg_theory = np.greater(relative_threshold_surf, 1) 
det_2deg_theory = np.greater(relative_threshold_surf, 2) 
det_3deg_theory = np.greater(relative_threshold_surf, 3) 
det_4deg_theory = np.greater(relative_threshold_surf, 4) 

# Check of inconsistencies
print(det_1deg_theory.sum()>=det_2deg_theory.sum()>=det_3deg_theory.sum()>=det_4deg_theory.sum()) #True

# When SST >= 90th perc AND certain temperature -> OBSERVED
det_1deg = np.where(det_ds.mhw_rel_threshold &  det_ds.mhw_abs_threshold_1_deg, True, False)
det_2deg = np.where(det_ds.mhw_rel_threshold & det_ds.mhw_abs_threshold_2_deg, True, False)
det_3deg = np.where(det_ds.mhw_rel_threshold & det_ds.mhw_abs_threshold_3_deg, True, False)
det_4deg = np.where(det_ds.mhw_rel_threshold & det_ds.mhw_abs_threshold_4_deg, True, False)

# Check of inconsistencies
print(det_1deg.sum()>=det_2deg.sum()>=det_3deg.sum()>=det_4deg.sum()) #True

# Reformating
det_combined_ds= xr.Dataset(
    data_vars=dict(
        det_1deg = (["years", "days", "eta_rho" ,"xi_rho"], det_1deg),
        det_2deg = (["years", "days", "eta_rho" ,"xi_rho"], det_2deg),
        det_3deg = (["years", "days", "eta_rho" ,"xi_rho"], det_3deg),
        det_4deg = (["years", "days", "eta_rho" ,"xi_rho"], det_4deg)
        ),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs = {
            'det_ideg': "Detected events where SST > (absolute threshold (i°C) AND 90th percentile) , boolean array"
            }                
        ) 

# Save output
output_file= os.path.join(path_det, f"det_rel_abs_combined.nc")
if not os.path.exists(output_file):
    det_combined_ds.to_netcdf(output_file, mode='w')

# test1 = relative_threshold_surf.isel(eta_rho=100, xi_rho=800, days=98).values #-1.5387573°C
# test2 = xr.open_dataset(path_temp + file_var + 'eta' + str(100) + '.nc' ).isel(xi_rho=800, year=37, day=98, z_rho=0).temp.values #-1.3606567 °C
# test3 = det_ds.isel(eta_rho=100, xi_rho=800, years=37, days=98).mhw_rel_threshold.values #above rel threshold : TRUE
# test4 = det_ds.isel(eta_rho=100, xi_rho=800, years=37, days=98).mhw_abs_threshold_4_deg.values #below abs threshold : FALSE
# test5 = det_combined_ds.isel(eta_rho=100, xi_rho=800, years=37, days=98).det_4deg.values #not in condition FALSE

# %% Visualization - load data
choice_eta = 200 #200, 220, 190
choice_xi =  1000 #1000, 950, 600
day_to_plot = 300 # 300, 98, 67
year_to_plot = 37
choice_year = slice(35,40)

# # Find interesting location
# test=np.where(xr.open_dataset(path_clim + 'threshold_90perc_surf.nc').relative_threshold >4)
# test2 = np.where(xr.open_dataset(path_temp + 'temp_DC_BC_surface.nc').temp > 4)

# Surface temperature for 1 location (eta, xi)
selected_temp_surf = xr.open_dataset(path_temp + file_var + 'eta' + str(choice_eta) + '.nc')[var][:, 0:365, 0, :].sel(xi_rho=choice_xi)  # 30yrs - 365days per year
selected_temp_surf = selected_temp_surf.isel(year=choice_year)
selected_temp_surf = selected_temp_surf.stack(time=['year', 'day'])

# 90th percentile threshold for 1 location (eta, xi)
selected_rel_thresh_surf = xr.open_dataset(path_clim + 'thresh_90perc_' + str(choice_eta) + '.nc').sel(xi_rho=choice_xi).relative_threshold  
selected_rel_thresh_surf=np.tile(selected_rel_thresh_surf, choice_year.stop-choice_year.start) #repeat the relative threshold 


# %% Visualization - time series
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
time_values = np.array([f"{year}-{day:03d}" for year, day in selected_temp_surf.time.values])

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(selected_temp_surf, color='black', label='SST')
ax.plot(selected_rel_thresh_surf, color='#7832AE', label='90th perc')

ax.hlines(y=1, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[0])
ax.hlines(y=2, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[1])
ax.hlines(y=3, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[2])
ax.hlines(y=4, xmin=0, xmax=selected_temp_surf.shape[0], linestyle=':', color=threshold_colors[3])

# Condition: temp>90th perc and i°C 
sst_values = selected_temp_surf.values  # to np array
threshold_values = selected_rel_thresh_surf  # to np array

for temp, color, alpha in zip([1, 2, 3, 4], threshold_colors[:], [0.7, 0.7, 0.9, 0.9]):
    condition = (sst_values >= threshold_values) & (sst_values >= temp)
    sst_masked = np.where(condition, sst_values, np.nan)
    threshold_masked = np.where(condition, threshold_values, np.nan)
    ax.fill_between(time_values, threshold_masked, sst_masked, color=color, alpha=alpha)


# Highlight specific day (plot after)
day_map = "2017-" + str(day_to_plot)
idx = np.where(time_values == day_map)[0]

ticks_years = np.arange(0, 365 * (choice_year.stop + 1 - choice_year.start), 365)
tick_labels = np.append(np.unique([int(time_values[i].split("-")[0]) for i in range(time_values.shape[0])]), 2019)

if idx.size > 0:
    day_index = idx[0]
    new_ticks = np.append(ticks_years, day_index)  # Add custom tick
    ax.set_xticks(new_ticks)
    
    # Use existing labels but leave space for custom tick
    new_labels = list(map(str, tick_labels)) + [""]
    ax.set_xticklabels(new_labels)
    
    ax.text(day_index, ax.get_ylim()[0] - 0.2, f"$d_{{{day_to_plot}}}$", color='red', ha='center') # Tick in red
    ax.tick_params(axis='x', direction='in', length=5) # Small tick visible


ax.set_xlim(0, 365 * (choice_year.stop - choice_year.start))
ax.set_xlabel('Time')
ax.set_ylabel('SST (°C)')
ax.set_title(f'SST above thresholds \nLocation: ({round(selected_temp_surf.lat_rho.item())}°S, {round(selected_temp_surf.lon_rho.item())}°E) from {1980+choice_year.start-1} to {1980+choice_year.stop-1}')
ax.legend(loc='upper right', fontsize=10, bbox_to_anchor = (1, 1))

plt.tight_layout()
plt.show()

# %% Visualization - map COMBINED thresholds
from matplotlib.lines import Line2D      
det_combined_ds = xr.open_dataset(os.path.join(path_det, f"det_rel_abs_combined.nc"))

titles = ["rel.threshold > 1°C", "rel.threshold > 2°C", "rel.threshold > 3°C", "rel.threshold > 4°C"]
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
ds_to_plot = [det_combined_ds.det_1deg, det_combined_ds.det_2deg, det_combined_ds.det_3deg, det_combined_ds.det_4deg]

# --- PLOT
fig, axs = plt.subplots(1, 4, figsize=(15, 10), subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)})
axs = axs.flatten()

# Loop through each subplot and dataset
for i, (ax, dataset, title, col) in enumerate(zip(axs, ds_to_plot, titles, threshold_colors)):
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular map boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Plot the data for the current day
    pcolormesh = dataset.isel(years=year_to_plot, days=day_to_plot).plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x="lon_rho", y="lat_rho",
        add_colorbar=False,
        cmap=plt.matplotlib.colors.ListedColormap(['lightgray', col])
    )

    point_data = dataset.isel(eta_rho=choice_eta, xi_rho=choice_xi, years=year_to_plot, days=day_to_plot)
    point_color = col if point_data.item() else 'lightgray'  # Adjust the color condition as needed
    sc = ax.scatter(point_data.lon_rho.item(), point_data.lat_rho.item(), c=[point_color], cmap=plt.matplotlib.colors.ListedColormap([point_color]),
                    transform=ccrs.PlateCarree(), s=50, edgecolor='black', zorder=3, label='Selected Cell')

    # Map features
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Legend
    threshold = title.split(" ")[2]

    # Create a binary legend for the current threshold
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label=f'T°C < {threshold}',
               markerfacecolor='lightgray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label=f'T°C ≥ {threshold}',
               markerfacecolor=col, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=14,
              borderpad=0.8, frameon=True, bbox_to_anchor=(0.5, -0.15))
    
    # Atlantic-Pacific boundary (near Drake Passage)
    ax.plot([-70, -70], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Atlantic sector

    # Pacific-Indian boundary
    ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Pacific sector

    # Indian-Atlantic boundary
    ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Indian sector

# Title
# plt.suptitle(f"Relative threshold above absolute ($d_{{{day_to_plot}}}$) \n- not observed (theory) -", fontsize=16)
plt.suptitle(f"SST above relative and absolute thresholds ($y_{{{1980 + year_to_plot}}}, d_{{{day_to_plot}}}$)", fontsize=20, y=0.7)

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# %% Visualization - map ABSOLUTE thresholds
titles = ["SST > 1°C", "SST > 2°C",  "SST > 3°C", "SST > 4°C"]
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
ds_to_plot = [det_ds.mhw_abs_threshold_1_deg, det_ds.mhw_abs_threshold_2_deg, det_ds.mhw_abs_threshold_3_deg, det_ds.mhw_abs_threshold_4_deg]

# --- PLOT
fig, axs = plt.subplots(1, 4, figsize=(15, 10), subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)})
axs = axs.flatten()

# Loop through each subplot and dataset
for i, (ax, dataset, title, col) in enumerate(zip(axs, ds_to_plot, titles, threshold_colors)):
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular map boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Plot the data for the current day
    pcolormesh = dataset.isel(years=year_to_plot, days=day_to_plot).plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x="lon_rho", y="lat_rho",
        add_colorbar=False,
        cmap=plt.matplotlib.colors.ListedColormap(['lightgray', col])
    )

    point_data = dataset.isel(eta_rho=choice_eta, xi_rho=choice_xi, years=year_to_plot, days=day_to_plot)
    point_color = col if point_data.item() else 'lightgray'  # Adjust the color condition as needed
    sc = ax.scatter(point_data.lon_rho.item(), point_data.lat_rho.item(), c=[point_color], cmap=plt.matplotlib.colors.ListedColormap([point_color]),
                    transform=ccrs.PlateCarree(), s=50, edgecolor='black', zorder=3, label='Selected Cell')

    # Map features
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Legend
    threshold = title.split(" ")[2]

    # Create a binary legend for the current threshold
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label=f'T°C < {threshold}',
               markerfacecolor='lightgray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label=f'T°C ≥ {threshold}',
               markerfacecolor=col, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=14,
              borderpad=0.8, frameon=True, bbox_to_anchor=(0.5, -0.15))
    
    # Atlantic-Pacific boundary (near Drake Passage)
    ax.plot([-70, -70], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Atlantic sector

    # Pacific-Indian boundary
    ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Pacific sector

    # Indian-Atlantic boundary
    ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1) #Indian sector

# Title
# plt.suptitle(f"Relative threshold above absolute ($d_{{{day_to_plot}}}$) \n- not observed (theory) -", fontsize=16)
plt.suptitle(f"SST above absolute thresholds ($y_{{{1980 + year_to_plot}}}, d_{{{day_to_plot}}}$)", fontsize=20, y=0.7)

# Adjust layout and show plot
plt.tight_layout()
plt.show()


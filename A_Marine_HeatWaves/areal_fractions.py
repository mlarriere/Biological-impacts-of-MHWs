#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 26 March 10:29:31 2025

Areal fraction of MHW defined as 90th percentile + exceeding a certain absolute threshold

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

from joblib import Parallel, delayed


# %% -------------------------------- SETTINGS --------------------------------
# -- Directories
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'

# -- Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute thresholds


# %% Load data 
det_combined_ds = xr.open_dataset(os.path.join(path_det, f"det_rel_abs_combined.nc"))

# -- Seasons 
def define_season(ds):
    # Defining seasons: [DJF, MAM, JJA, SON]
    season_labels = np.empty(365, dtype='<U3')  # array of strings
    season_labels[0:59] = 'DJF'
    season_labels[59:151] = 'MAM'
    season_labels[151:243] = 'JJA'
    season_labels[243:334] = 'SON'
    season_labels[334:] = 'DJF'

    ds = ds.assign_coords(season=('days', season_labels))
    seasons = {season: group for season, group in ds.groupby('season')} # dictionary with season names as keys and seasonal data arrays as values

    return seasons  


# %% COMPUTE SPATIAL SECTORS
det_combined_ds["lon_rho"] = det_combined_ds["lon_rho"] % 360 # Ensure longitude is wrapped to 0-360° range
# print(det_combined_ds.lon_rho.values)
# print(np.isnan(det_combined_ds.lon_rho).sum())

spatial_domain_atl = (det_combined_ds.lon_rho >= 290) | (det_combined_ds.lon_rho < 20) #Atlantic sector: From 290°E to 20°E -OK
mask_atl = xr.DataArray(spatial_domain_atl, dims=["eta_rho", "xi_rho"]) #shape: (434, 1442)
spatial_domain_pac = (det_combined_ds.lon_rho >= 150) & (det_combined_ds.lon_rho < 290) #Pacific sector: From 150°E to 290°E -OK
mask_pac = xr.DataArray(spatial_domain_pac, dims=["eta_rho", "xi_rho"])
mask_indian = ~(mask_atl | mask_pac)

# Mask --- value if inside sector, Nan otherwise
def apply_spatial_mask(yr, ds, variables, mask_da, sector):
    # Read only 1 year
    ds_yr = ds.isel(years=yr)

    # Mask
    masked_vars = {var: (ds_yr[var].dims, ds_yr[var].data * mask_da.data) for var in variables}

    # Reformating - new dataset
    data_spat_filtered = xr.Dataset(
        data_vars=masked_vars,
        coords=dict(
            lon_rho=(ds_yr.lon_rho.dims, ds_yr.lon_rho.values),
            lat_rho=(ds_yr.lat_rho.dims, ds_yr.lat_rho.values),
        ),
        attrs=dict(description=f'MHW duration in {sector} sector')
    )
    
    return data_spat_filtered

# Spatial masks -- parallelization on years -- computing time ~2min each
print('Spatial masks:')
print('Atlantic')
spatial_mask_atl = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, det_combined_ds, ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], mask_atl, 'atlantic') for yr in range(0, nyears))
print('Pacific')
spatial_mask_pac = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, det_combined_ds, ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], mask_pac, 'pacific') for yr in range(0, nyears))
print('Indian')
spatial_mask_ind = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, det_combined_ds, ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], mask_indian, 'indian') for yr in range(0, nyears))

# Put back to original dimensions -- computing time ~2min each
print('\nBack to original dimensions')
spatial_mask_atl_all = xr.concat(spatial_mask_atl, dim='years')
spatial_mask_pac_all = xr.concat(spatial_mask_pac, dim='years')
spatial_mask_ind_all = xr.concat(spatial_mask_ind, dim='years')

# %% COMPUTE AREAL FRACTION 
# Area dataset -- variables: 'area'= view from top, 'area_xi', 'area_eta'=view from side
area_SO = xr.open_dataset("/home/jwongmeng/work/ROMS/scripts/mhw_krill/area.nc") 
area_SO_surf = np.nan_to_num(area_SO.isel(z_t=0).area, 0) # Area only surface - contains Nans values 
total_area_SO = np.nansum(area_SO_surf)

# --- PLOT area SO
vizualise_area_cells = False
if vizualise_area_cells:
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    pcolormesh = ax.pcolormesh(
        area_SO.area[0, :,:].lon_rho, area_SO.area[0, :,:].lat_rho, area_SO.area[0, :,:],
        transform=ccrs.PlateCarree(), cmap="viridis"# , vmin=0, vmax=300
    )
    cbar = plt.colorbar(pcolormesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Area [km2]', fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2,  facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')
    ax.set_title(f"Area of each cell (ROMS model) - only surface \nTotal area of Southern Ocean: {total_area_SO:.2e} km²", fontsize=18)
    plt.tight_layout()
    plt.show()

# --- Area sectors
area_SO_da = xr.DataArray(area_SO_surf, dims=["eta_rho", "xi_rho"])
total_area_atl = np.nansum(area_SO_da.where(mask_atl, np.nan))
total_area_pac = np.nansum(area_SO_da.where(mask_pac, np.nan))
total_area_ind = np.nansum(area_SO_da.where(mask_indian, np.nan))

# --- For each cell, if mhw detected (boolean) associate area and sum over spatial dim - resulting dataset of areas for each cell under MHW
def areal_fraction_seasonally(ds, area_cells, total_area):
    # Define seasons
    seasons = define_season(ds)
    yearly_result = {}
    daily_result = {}

    for season_name, season_ds in zip(seasons.keys(), seasons.values()):
        # print(f"  Processing season: {season_name}")
        # Method2 - computing time ~2s
        combine_area_season = np.einsum('ijkl, kl -> ij', season_ds.values, area_cells)
        area_frac_season_daily = np.divide(combine_area_season, total_area)

        # Yearly mean
        yearly_result[season_name] = np.mean(area_frac_season_daily, axis=1) * 100
        print(f"  {season_name} area fraction sample: {yearly_result[season_name][:5]}")

    return yearly_result 


# Define spatial regions 
regions = {
    "SO": (det_combined_ds, total_area_SO), #(dataset, total area)
    "Atlantic": (spatial_mask_atl_all, total_area_atl),
    "Pacific": (spatial_mask_pac_all, total_area_pac),
    "Indian": (spatial_mask_ind_all, total_area_ind)
    }

# Compute areal fractions dynamically ~20min
area_fractions_yearly = {}
area_fractions_daily = {}

# Extract region of interest
for region_name, (ds_region, region_total_area) in regions.items():
    print(f"\nProcessing region: {region_name}")
    area_fractions_yearly[region_name] = {}
    area_fractions_daily[region_name] = {}

    # Loop over thresholds (1deg, 2deg, 3deg, 4deg)
    for threshold in absolute_thresholds:
        # Extract the corresponding data array for each threshold
        ds_threshold = getattr(ds_region, f"det_{threshold}deg", None) # shape (40, 365, 434, 1442)

        # If the threshold data exists, calculate the areal fraction
        if ds_threshold is not None:
            print(f"Processing threshold: {threshold}deg")
            # key = f"area_frac_{threshold}deg_{region_name.lower()}"
            area_fractions_yearly[region_name][threshold] = areal_fraction_seasonally(ds_threshold, area_SO_surf, region_total_area)
            # area_fractions_yearly[region_name][threshold] = areal_fraction_seasonally(ds_threshold, area_SO_surf, region_total_area, 'yearly')
            # area_fractions_daily[region_name][threshold] = areal_fraction_seasonally(ds_threshold, area_SO_surf, region_total_area, 'daily')



# %% Time series
# ---- 1. One graph for each extent with the 4 threshold conditions
# threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808'] # 90th percentile + [1°C, 2°C, 3°C, 4°C]

# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
# axes = axes.flatten() 

# regions_names = ['SO', 'Atlantic', 'Pacific', 'Indian']
# region_titles = ["Southern Ocean", "Atlantic Sector", "Pacific Sector", "Indian Sector"]

# for ax, region, title in zip(axes, regions_names, region_titles):
#     for threshold, color in zip(absolute_thresholds, threshold_colors):
#         ax.plot(
#             area_fractions[region][threshold], linestyle="-", color=color, linewidth=1.5,
#             label=f"> 90th percentile & {threshold}\u00b0C"
#         )

#     ax.set_title(title, fontsize=12)
#     ax.legend(fontsize=8, loc="upper left")
#     years= np.arange(1980,2019)
    
#     ticks_years = np.arange(0, len(years))
#     selected_years = years[::5].tolist() + [2019]  # Every 5 years + 2019
#     selected_ticks = ticks_years[::5].tolist() + [ticks_years[-1]]  # Match positions
#     ax.set_xticks(selected_ticks)
#     ax.set_xticklabels(selected_years, rotation=45)
#     ax.set_xlim(0, nyears-1)

# # Shared labels
# fig.text(0.5, 0.04, "Year", ha="center", fontsize=12)
# fig.text(0.04, 0.5, "Areal Fraction [%]", va="center", rotation="vertical", fontsize=12)

# plt.suptitle("Areal Fraction of Marine Heatwaves by Region", fontsize=14)
# plt.tight_layout(rect=[0.05, 0.05, 1, 1])
# plt.show()


# # ---- 2. One graph for threshold condition with the 4 extents
# sector_colors = ['black', '#778B04', '#BF3100', '#E09F3E'] #SO, Atlantic, Pacific, Indian

# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
# axes = axes.flatten()

# threshold_titles = [f"> 90th percentile & {t}\u00b0C" for t in absolute_thresholds]

# for ax, threshold, title in zip(axes, absolute_thresholds, threshold_titles):

#     for region, color in zip(regions_names, sector_colors):  # Assign correct color to each region

#         ax.plot(area_fractions[region][threshold], linestyle="-", linewidth=1.5, color=color, label=region)

#         if region == "SO":
#             m, b = np.polyfit(np.arange(0,40), area_fractions['SO'][threshold],  1) #m=0.547, b=20
#             trend_line = m * np.arange(0,40) + b  
#             ax.plot(trend_line, color='black', linestyle="--", linewidth=1)


#     ax.set_title(title, fontsize=12)
#     ax.legend(fontsize=8, loc="upper left")

#     years = np.arange(1980, 2020)
#     ticks_years = np.arange(0, len(years))
#     selected_years = years[::5].tolist() + [2019]  # Every 5 years + 2019
#     selected_ticks = ticks_years[::5].tolist() + [ticks_years[-1]]  # Match positions
#     ax.set_xticks(selected_ticks)
#     ax.set_xticklabels(selected_years, rotation=45)
#     ax.set_xlim(0, nyears-1)

# # Shared labels
# fig.text(0.5, 0.04, "Year", ha="center", fontsize=12)
# fig.text(0.04, 0.5, "Areal Fraction [%]", va="center", rotation="vertical", fontsize=12)

# plt.suptitle("Areal Fraction of Marine Heatwaves by Threshold Condition", fontsize=14)
# plt.tight_layout(rect=[0.05, 0.05, 1, 1])
# plt.show()

# %% Detect year with peak values
from scipy.signal import find_peaks
from scipy.signal import find_peaks

# Loop over the seasons
for season in ["DJF", "MAM", "JJA", "SON"]:
    print(f"\nSeason: {season}")
    common_peaks = None #common peaks in season for the 4 thresholds 
    
    # Loop over the thresholds
    for threshold in absolute_thresholds:
        # print(f"  Southern Ocean, {threshold}°C")

        yearly_series = area_fractions_yearly['SO'][threshold][season]
        yearly_peaks, _ = find_peaks(yearly_series)
        peak_years = set([1980 + idx for idx in yearly_peaks])  # Use set to make it easy to take intersections
        
        if common_peaks is None:
            common_peaks = peak_years
        else:
            common_peaks &= peak_years
            
    if common_peaks:
        print(f"  Common peak years in SO for all thresholds in season {season}: {sorted(common_peaks)}")
    else:
        print(f"  No common peak years for all thresholds in season {season}")

# %% Seasonal plot
fig, ax = plt.subplots(figsize=(10, 6))

threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']  # Colors for thresholds
season_styles = ["-", "--", ":", "-."]  # Line styles for seasons
season_labels = ["DJF", "MAM", "JJA", "SON"]

so_seasonal_data = area_fractions_yearly["SO"]  # Extract SO data

# Store legend elements
threshold_legend = [plt.Line2D([0], [0], color=c, lw=2, label=f"> {t}\u00b0C") for t, c in zip(absolute_thresholds, threshold_colors)]
season_legend = [plt.Line2D([0], [0], color='black', linestyle=s, lw=2, label=season) for season, s in zip(season_labels, season_styles)]

for season, style in zip(season_labels, season_styles):
    for threshold, color in zip(absolute_thresholds, threshold_colors):
            # threshold = absolute_thresholds[0]
            # color = threshold_colors[0]
            if threshold in so_seasonal_data and season in so_seasonal_data[threshold]:
                ax.plot(so_seasonal_data[threshold][season], linestyle=style, color=color, linewidth=1.5)

# Formatting
ax.set_title("Seasonal Areal Fraction of Marine Heatwaves - Southern Ocean", fontsize=14)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Areal Fraction [%]", fontsize=12)
ax.set_ylim(3,20)
years = np.arange(1980, 2019)
ticks_years = np.arange(0, len(years))
selected_years = years[::5].tolist() + [2019]  # Every 5 years + 2019
selected_ticks = ticks_years[::5].tolist() + [ticks_years[-1]]  # Match positions

ax.set_xticks(selected_ticks)
ax.set_xticklabels(selected_years, rotation=45)
ax.set_xlim(0, len(years) - 1)

# Add two separate legends, ensuring the threshold legend appears only once
threshold_legend_handle = ax.legend(handles=threshold_legend, title="Thresholds", loc="upper left", fontsize=10)
ax.add_artist(threshold_legend_handle)  # Ensure the first legend stays
ax.legend(handles=season_legend, title="Seasons", loc="upper right", fontsize=10)

plt.tight_layout()
plt.show()



# %% Time Serie - 1 subplot per season, yearly time step, color=threshold condition
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']  # Colors for thresholds
season_labels = ["DJF", "MAM", "JJA", "SON"]
season_titles = ["Summer (DJF)", "Spring (MAM)", "Winter (JJA)", "Autumn (SON)"]
season_styles = ["-", "--", ":", "-."]  # Different line styles for seasons
threshold_titles = [f"> 90th percentile & {t}\u00b0C" for t in absolute_thresholds]

so_seasonal_data = area_fractions_yearly["SO"]  # Extract SO data

# --- PLOT
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axes = axes.flatten() 

# For legend
threshold_legend = [plt.Line2D([0], [0], color=c, lw=5, label=f"{t}") for t, c in zip(threshold_titles, threshold_colors)]

# Plot for each season
for i, (season, ax, style) in enumerate(zip(season_labels, axes, season_styles)):
    for threshold, color in zip(absolute_thresholds, threshold_colors):
        if threshold in so_seasonal_data and season in so_seasonal_data[threshold]:
            ax.plot(
                so_seasonal_data[threshold][season], linestyle=style, color=color, linewidth=1.5
            )

    # titles and labels
    ax.set_title(f"{season_titles[i]}", weight ='bold', fontsize=12)
    # yaxis
    ax.set_ylabel("Areal Fraction [%]", fontsize=10)
    ax.set_ylim(3, 20)

    # Add gridlines that span all subplots
    ax.grid(True, linestyle='--', alpha=0.7)

# x axis - only last subplot
years = np.arange(1980, 2019)
ticks_years = np.arange(0, len(years))
selected_years = years[::5].tolist() + [2019]  # Every 5 years + 2019
selected_ticks = ticks_years[::5].tolist() + [ticks_years[-1]]  # Match positions
ax.set_xticklabels(selected_years, rotation=45)
ax.set_xlim(0, len(years) - 1)
ax.set_xticks(selected_ticks)
ax.set_xlabel("Year", fontsize=10)

fig.suptitle('Area Fraction for the Southern Ocean')
fig.legend(handles=threshold_legend, title="Thresholds", loc="upper center", bbox_to_anchor=(0.5, -0.001), ncol=2, fontsize=10)
plt.tight_layout()
plt.show()

# %%
from matplotlib.lines import Line2D

# Load dataset
season_ds = define_season(det_combined_ds)

# Define the dataset for each season and threshold combination -- not area fraction but just boolean arrays
# Calculate the mean of the seasonal data fro each threhold
ds_to_plot = [season_ds["DJF"].det_1deg, season_ds["DJF"].det_2deg, season_ds["DJF"].det_3deg, season_ds["DJF"].det_4deg, 
              season_ds["MAM"].det_1deg, season_ds["MAM"].det_2deg, season_ds["MAM"].det_3deg, season_ds["MAM"].det_4deg, 
              season_ds["JJA"].det_1deg, season_ds["JJA"].det_2deg, season_ds["JJA"].det_3deg, season_ds["JJA"].det_4deg, 
              season_ds["SON"].det_1deg, season_ds["SON"].det_2deg, season_ds["SON"].det_3deg, season_ds["SON"].det_4deg]



# Defining title and colors - plot settings
year_to_plot = 18
day_to_plot = 30 #30th days of each season, i.e. approx DJF = 1st January, MAM = 1st April, JJA = 1st July, SON = 1st October
season_days = {
    "DJF": "January 30",
    "MAM": "April 30",
    "JJA": "July 30",
    "SON": "October 30"
}


threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_titles = [f"> 90th percentile & {t}\u00b0C" for t in absolute_thresholds]
seasons = ["DJF", "MAM", "JJA", "SON"]
season_titles = ["Summer (DJF)", "Spring (MAM)", "Winter (JJA)", "Autumn (SON)"]

# --- PLOT
fig, axs = plt.subplots(4, 4, figsize=(15, 20), subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)})
axs = axs.flatten()  # Ensure the grid is 4x4

# Loop over axis: column = threshold conditions, row = seasons
for idx, (ax, dataset) in enumerate(zip(axs, ds_to_plot)):
    row = idx // 4  # Get the row (season)
    col = idx % 4   # Get the column (threshold)

    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Circular map boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Plot the data for the current threshold and season
    pcolormesh = dataset.isel(years=year_to_plot, days= day_to_plot).plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x="lon_rho", y="lat_rho",
        add_colorbar=False,
        cmap=plt.matplotlib.colors.ListedColormap(['lightgray', threshold_colors[col]])
    )

    # Map features
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Set title for the first column (threshold) and the row (season)
    if row == 0:
        ax.set_title(threshold_titles[col] , fontsize=14)
    else:
        ax.set_title(' ')

    if col == 0:
        ax.set_ylabel(f"> {absolute_thresholds[col]}°C", fontsize=14, labelpad=20)
        ax.set_title(f"$\\bf{{{season_titles[row]}}}$ - {season_days[seasons[row]]}", fontsize=14, loc='left', y=1.1)

  
    # Adding line for sectors
    ax.plot([-70, -70], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    ax.plot([150, 150], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    ax.plot([20, 20], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)

# Title for the entire figure
plt.suptitle(f"SST above relative and absolute thresholds in $\\bf{{{1980 + year_to_plot}}}$",fontsize=20, y=1)
plt.tight_layout()
plt.show()


# %%

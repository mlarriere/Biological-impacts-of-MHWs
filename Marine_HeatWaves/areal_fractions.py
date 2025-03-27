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

# %% COMPUTE SPATIAL SECTORS
det_combined_ds["lon_rho"] = det_combined_ds["lon_rho"] % 360 # Ensure longitude is wrapped to 0-360° range
print(det_combined_ds.lon_rho.values)
print(np.isnan(det_combined_ds.lon_rho).sum())

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
spatial_mask_atl = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, det_combined_ds, ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], mask_atl, 'atlantic') for yr in range(0, nyears))
spatial_mask_pac = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, det_combined_ds, ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], mask_pac, 'pacific') for yr in range(0, nyears))
spatial_mask_ind = Parallel(n_jobs=30)(delayed(apply_spatial_mask)(yr, det_combined_ds, ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg'], mask_indian, 'indian') for yr in range(0, nyears))

# Put back to original dimensions -- computing time ~1min each
spatial_mask_atl_all = xr.concat(spatial_mask_atl, dim='years')
spatial_mask_pac_all = xr.concat(spatial_mask_pac, dim='years')
spatial_mask_ind_all = xr.concat(spatial_mask_ind, dim='years')

# %% COMPUTE AREAL FRACTION 
# Area dataset -- variables: 'area'= view from top, 'area_xi', 'area_eta'=view from side
area_SO = xr.open_dataset("/home/jwongmeng/work/ROMS/scripts/mhw_krill/area.nc") 
area_SO_surf = area_SO.isel(z_t=0).area # Area only surface - contains Nans values 
total_area_SO = np.nansum(area_SO_surf)

# --- PLOT area SO
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

# --- For each cell, if mhw detected (boolean) associate area and sum over spatial dim
def areal_fraction(ds, area_cells, total_area, time_dim):
    # Testing
    # yr=37
    # ds = det_combined_ds.det_1deg
    # area_cells = area_SO_surf
    # # total_area= total_area_SO

    # # Method1 - computing time ~ 1min
    # # Area in a MHW
    # detected_area = np.multiply(ds, area_cells)
    # area_det = detected_area.sum(axis=(2, 3)) 
    # area_frac_det1 = np.divide(area_det, total_area) #shape: (40, 365)

    # Method2 - computing time ~2s
    combine_area = np.einsum('ijkl,kl -> ij', ds.values, np.nan_to_num(area_cells)) #shape: (40, 365) --- NO 
    area_frac_det2 = np.divide(combine_area, total_area)

    # print(np.allclose(area_frac_det1, area_frac_det2))  # Should return True if results are the same

    if time_dim == 'yearly': 
        area_frac_det_yr = np.mean(area_frac_det2, axis=(1)) * 100
        return area_frac_det_yr
    else: 
        return area_frac_det2

# Spatial regions
regions = {
    "SO": (det_combined_ds, total_area_SO),
    "Atlantic": (spatial_mask_atl_all, total_area_atl),
    "Pacific": (spatial_mask_pac_all, total_area_pac),
    "Indian": (spatial_mask_ind_all, total_area_ind)
}

# Compute areal fractions dynamically
area_fractions = {}
for region_name, (region_ds, region_total_area) in regions.items():
    area_fractions[region_name] = {}
    for threshold in absolute_thresholds:
        key = f"area_frac_{threshold}deg_{region_name.lower()}_yr"
        area_fractions[region_name][threshold] = areal_fraction(getattr(region_ds, f"det_{threshold}deg"), area_SO_surf, region_total_area, 'yearly')


# %% Time series
# ---- 1. One graph for each extent with the 4 threshold conditions
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808'] # 90th percentile + [1°C, 2°C, 3°C, 4°C]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten() 

regions_names = ['SO', 'Atlantic', 'Pacific', 'Indian']
region_titles = ["Southern Ocean", "Atlantic Sector", "Pacific Sector", "Indian Sector"]

for ax, region, title in zip(axes, regions_names, region_titles):
    for threshold, color in zip(absolute_thresholds, threshold_colors):
        ax.plot(
            area_fractions[region][threshold], linestyle="-", color=color, linewidth=1.5,
            label=f"> 90th percentile & {threshold}\u00b0C"
        )

    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    years= np.arange(1980,2019)
    
    ticks_years = np.arange(0, len(years))
    selected_years = years[::5].tolist() + [2019]  # Every 5 years + 2019
    selected_ticks = ticks_years[::5].tolist() + [ticks_years[-1]]  # Match positions
    ax.set_xticks(selected_ticks)
    ax.set_xticklabels(selected_years, rotation=45)
    ax.set_xlim(0, nyears-1)

# Shared labels
fig.text(0.5, 0.04, "Year", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Areal Fraction [%]", va="center", rotation="vertical", fontsize=12)

plt.suptitle("Areal Fraction of Marine Heatwaves by Region", fontsize=14)
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()


# ---- 2. One graph for threshold condition with the 4 extents
sector_colors = ['black', '#778B04', '#BF3100', '#E09F3E'] #SO, Atlantic, Pacific, Indian

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

threshold_titles = [f"> 90th percentile & {t}\u00b0C" for t in absolute_thresholds]

for ax, threshold, title in zip(axes, absolute_thresholds, threshold_titles):

    for region, color in zip(regions_names, sector_colors):  # Assign correct color to each region

        ax.plot(area_fractions[region][threshold], linestyle="-", linewidth=1.5, color=color, label=region)

        if region == "SO":
            m, b = np.polyfit(np.arange(0,40), area_fractions['SO'][threshold],  1) #m=0.547, b=20
            trend_line = m * np.arange(0,40) + b  
            ax.plot(trend_line, color='black', linestyle="--", linewidth=1)


    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper left")

    years = np.arange(1980, 2020)
    ticks_years = np.arange(0, len(years))
    selected_years = years[::5].tolist() + [2019]  # Every 5 years + 2019
    selected_ticks = ticks_years[::5].tolist() + [ticks_years[-1]]  # Match positions
    ax.set_xticks(selected_ticks)
    ax.set_xticklabels(selected_years, rotation=45)
    ax.set_xlim(0, nyears-1)

# Shared labels
fig.text(0.5, 0.04, "Year", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Areal Fraction [%]", va="center", rotation="vertical", fontsize=12)

plt.suptitle("Areal Fraction of Marine Heatwaves by Threshold Condition", fontsize=14)
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()

# %%

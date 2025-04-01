#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon 26 March 10:34:17 2025

Frequency of the 4 different events: SST > (relative + absolute thresholds)

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

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold

# %% Load data 
det_combined_ds = xr.open_dataset(os.path.join(path_det, f"det_rel_abs_combined.nc"))

# %% COMPUTE FREQUENCY
# Frequency corresponds to the number of days per cell under a MHW, here defined using combined thresholds
variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
frequency_list = []

# Loop over variables - computing time ~2min for each var 
for var in variables:
    # Testing
    # var = 'det_1deg'
    # ds=det_combined_ds

    # Extract data
    ds_data = det_combined_ds[var].values  # NumPy array

    # Calculate frequency, i.e. mean number of extreme days per year
    frequency = np.nansum(ds_data, axis=1)  
    print(f"Maximum frequency for {var}: {np.max(frequency)}") 

    # Into DataArray
    frequency_da = xr.DataArray(
        frequency, 
        dims=['years', 'eta_rho', 'xi_rho'], 
        coords={'lon_rho': det_combined_ds['lon_rho'], 'lat_rho': det_combined_ds['lat_rho']},
        name=f'freq_{var[4]}deg' 
    )
    frequency_list.append(frequency_da)

mhw_frequency_ds = xr.merge(frequency_list)
mhw_frequency_ds.attrs = dict(description='MHW frequency, number of days under MHW in each cell')

# Yearly average
mhw_frequency_baseline = mhw_frequency_ds.isel(years=slice(0,30)).mean(dim=['years'])
mhw_frequency_yr_avg = mhw_frequency_ds.mean(dim=['years']) 
np.max(mhw_frequency_yr_avg.freq_4deg)

# %% --- PLOT area SO
variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']

fig, axs = plt.subplots(2, 2, figsize=(15, 15), subplot_kw={'projection': ccrs.Orthographic(central_latitude=-90, central_longitude=0)})
axs = axs.flatten()

for i, var in enumerate(variables):
    ax = axs[i]
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Plot
    pcolormesh = mhw_frequency_yr_avg[f'freq_{var[4]}deg'].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x="lon_rho", y="lat_rho",
        add_colorbar=False,
        cmap='magma'
        # vmin=0, vmax=36
    )

    # Colorbar
    cbar = plt.colorbar(pcolormesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Days per year', fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # Add features
    ax.coastlines(color='black', linewidth=1.5, zorder=1)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Title
    ax.set_title(f"Thresholds: 90th perc and {var[4]}°C \n 1980-2019 period", fontsize=16)

fig.suptitle("MHW Frequency Yearly Average", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()


# %%

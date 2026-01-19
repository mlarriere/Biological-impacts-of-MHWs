"""
Created on Tue 30 July 17:04:45 2025

Calcuating the Biomass of E. Superba

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
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.gridspec as gridspec

import time
from tqdm.contrib.concurrent import process_map

from joblib import Parallel, delayed

#%% -------------------------------- Server --------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% --------------------------------Figure settings --------------------------------
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
path_biomass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_masslength = os.path.join(path_surrogates, f'mass_length')
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')


# %% ======================== Parameters ========================
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
proportion = {'juvenile': 0.20, 'immature': 0.3, 'mature': 0.3, 'gravid':0.2}

# %% ======================== Area ========================
# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc') #in km2

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)

# %% ======================== Roms grid cell volume ========================
# --- Load data
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc') #in km3

# Volume on the first 100m depth
# volume_roms_surf = volume_roms['volume'].isel(z_rho=0)
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 

# --- Southern Ocean south of 60°S
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)
# volume_60S_SO_surf = volume_roms_surf.where(volume_roms['lat_rho'] <= -60, drop=True)

# print(f'Maximum grid cell volume: {volume_60S_SO_100m.max().values:.3f} km3')
# print(f'Average grid cell volume: {volume_60S_SO_100m.mean().values:.3f} km3')
# print(f'Median grid cell volume: {volume_60S_SO_100m.median().values:.3f} km3')
# print(f'Minimum grid cell volume: {volume_60S_SO_100m.min().values:.3f} km3')

# Total Biomass
# %% ======================== Biomass from CEPHALOPOD ========================
# N = 17.85 # Abundance ind/m2

# -- Load data
# Biomass are climatological products
biomass_regridded = xr.open_dataset(os.path.join(path_cephalopod, 'euphausia_biomass_SO_regridded.nc')) #in mgC/m3
original_product = xr.open_dataset(os.path.join(path_cephalopod, 'total_krill_biomass_SO.nc'))

# %% ======================== Load data ========================
# --- Load mass data [mg] for each maturity stage -- Southern Ocean  
# 1. Climatology
clim_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_mass_stages_SO.nc")) #shape (181, 231, 1442)
clim_krilllength_SO = xr.open_dataset(os.path.join(path_masslength, "clim_length_stages_SO.nc")) #shape (181, 231, 1442)

# 2. Actual
actual_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "actual_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
actual_krilllength_SO = xr.open_dataset(os.path.join(path_masslength, "actual_length_stages_SO.nc")) #shape (39, 181, 231, 1442)

# 3. No MHWs (MHWs replaced by clim)
noMHWs_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "noMHWs_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)

# 3. No warming (temperature signal detrended)
nowarming_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "warming_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)

# Check - should be True
print(f'Is initial length identical? {(clim_krilllength_SO.immature.isel(days=0).values == actual_krilllength_SO.immature.isel(days=0, years=0).values).all()}')
print(f'Is initial mass identical (3dec)? {np.isclose(clim_krillmass_SO.immature.isel(days=0).values, actual_krillmass_SO.immature.isel(days=0, years=0).values, rtol=1e-6).all()}')


# %% ======================== Defining Functions ========================
from functools import partial
from tqdm.contrib.concurrent import process_map
import gc


def evolution_biomass_yr(year_idx, ds_mass, proportion):
    # test
    # year_idx = 36
    # ds_mass = nowarming_krillmass_SO
    # start_time = time.perf_counter()

    # -- Climatology case, i.e. no years dim
    has_years = 'years' in ds_mass.dims

    # -- Compute growth factor
    n_days = 181
    B0_mean = biomass_regridded.euphausia_biomass.isel(days=0, algo_bootstrap=slice(0,50,10)).mean(dim=['algo_bootstrap']).values #shape: (231, 1442)
    growth_fact_pop = np.zeros((n_days-1, *B0_mean.shape), dtype=np.float32) #shape (180, 231, 1442)
    
    for stage, p in proportion.items():
        if has_years:
            # Specific year
            ds_mass_yr = ds_mass[stage].isel(years=year_idx).values
        else:
            ds_mass_yr = ds_mass[stage].values # climatology  

        g_fact_stage = ds_mass_yr[1:] / ds_mass_yr[:-1] 
        growth_fact_pop += p * g_fact_stage # shape (180, 231, 1442)


    # -- Calculate biomass
    # Initialisation
    B_median = np.empty((n_days, *B0_mean.shape), dtype=np.float32)
    B_std = np.empty_like(B_median)
    
    # Initial conditions
    B0 = biomass_regridded.euphausia_biomass.isel(days=0).values #shape (10, 231, 1442)
    B_median[0] = np.median(B0, axis=0)
    B_std[0] = np.std(B0, axis=0)

    # Calculation
    for t in range(1, n_days):
        B0 *= growth_fact_pop[t - 1]
        B_median[t] = np.median(B0, axis=0)
        B_std[t] = np.std(B0, axis=0)

    # To DataSet
    biomass_ds = xr.Dataset(
        data_vars=dict(biomass_median=(("days", "eta_rho", "xi_rho"), B_median),
                       biomass_std=(("days", "eta_rho", "xi_rho"), B_std),),
        coords=dict(days=ds_mass.days,
                    eta_rho=ds_mass.eta_rho,
                    xi_rho=ds_mass.xi_rho,),
                    )

    return biomass_ds


def compute_biomass_surrogates(ds_mass, label, label_da, proportion, output_folder, max_workers=10):
    # test
    # max_workers=20
    # proportion=proportion
    # ds_mass=nowarming_krillmass_SO
    # label='No Warming'  
    # label_da='nowarming'  
    # output_folder= os.path.join(path_surrogates, f'biomass_timeseries')
    
    print(f"  → {label}")

    output_file= os.path.join(output_folder, f'biomass_{label_da}.nc')

    if not os.path.exists(output_file):
        # Prepare function
        func = partial(evolution_biomass_yr, ds_mass=ds_mass, proportion=proportion)

        # -- Climatology case, i.e. no years dim
        has_years = 'years' in ds_mass.dims

        if has_years:
            # Run in parallel
            B_algo_median_da = process_map(func, range(39), max_workers=max_workers, chunksize=1, desc=f"{label} | Biomass ") #len = nyears and shape (181, 231, 1442)

            # Extract data
            B_algo_median_all_ds = xr.concat(B_algo_median_da, dim="years")  # shape: (39, 181, 231, 1442)

            # Put together into dataset
            B_algo_median_all_ds.attrs.update({"surrogate": label,
                                               "description": "Evolution of krill biomass weighted according to population proportions.\nMedian and std over the models and bootstraps.",
                                               "population_proportions": ", ".join(f"{k} : {v*100:.0f}%" for k, v in proportion.items()),
                                               "initial_biomass": "Cephalopod output for krill total (euphausia biomass ~80% of it).\nRegridded to ROMS resolution, i.e. 0.25°.",
                                               "assumptions": "Fixed stage proportions, no mortality, no recruitment, no stage transitions within a growth season.",
                                               "units": "mgC/m3",})
            
            # Save to file
            B_algo_median_all_ds.to_netcdf(output_file, mode="w", engine="netcdf4")
            
            # Clean memory
            del B_algo_median_da
            gc.collect()

        else:
            # Run function only for clim, no year dimension
            B_algo_median_da = [func(0)]

            # Extract data
            B_algo_median_all_ds  = B_algo_median_da[0]  # shape: (181, 231, 1442)

            # Put together into dataset
            B_algo_median_all_ds.attrs.update({"surrogate": label,
                                        "description": "Evolution of krill biomass weighted according to population proportions.",
                                        "population_proportions": ", ".join(f"{k} : {v*100:.0f}%" for k, v in proportion.items()),
                                        "initial_biomass": "Cephalopod output for krill total (euphausia biomass ~80% of it), regridded to ROMS resolution.\nMedian over the models and bootstraps.",
                                        "assumptions": "Fixed stage proportions, no mortality, no recruitment, no stage transitions within a growth season.",
                                        "units": "mgC/m3",})
            # Save to file
            B_algo_median_all_ds.to_netcdf(output_file)

            # Clean memory
            del B_algo_median_da
            gc.collect()

    else:
        print(f"  → {label} already exists, skipping.")
        return

# %% ==================================== Compute Biomass for surrogates ====================================
output_folder=os.path.join(path_surrogates, f'biomass_timeseries')
files = [os.path.join(output_folder, "biomass_clim.nc"),
         os.path.join(output_folder, "biomass_actual.nc"),
         os.path.join(output_folder, "biomass_nomhws.nc"),
         os.path.join(output_folder, "biomass_nowarming.nc")]

# Run only if files don't exist
if not all(os.path.exists(f) for f in files):
    # ---------- Surrogate 1: Climatology
    clim_biomass = compute_biomass_surrogates(clim_krillmass_SO, label="Climatology", label_da='clim', proportion=proportion, output_folder=output_folder, max_workers=20)

    # ---------- Surrogate 2: Actual
    actual_biomass = compute_biomass_surrogates(actual_krillmass_SO, label="Actual", label_da='actual', proportion=proportion, output_folder=output_folder, max_workers=20)

    # ---------- Surrogate 3: No MHWs
    nomhw_biomass = compute_biomass_surrogates(noMHWs_krillmass_SO, label="No MHWs", label_da='nomhws', proportion=proportion, output_folder=output_folder, max_workers=20)

    # ---------- Surrogate 4: No Warming
    nowarming_biomass = compute_biomass_surrogates(nowarming_krillmass_SO, label="No Warming", label_da='nowarming', proportion=proportion, output_folder=output_folder, max_workers=20)

else:
    # Load data
    clim_biomass = xr.open_dataset(files[0])
    clim_biomass = clim_biomass.assign_coords(lon_rho=(("eta_rho", "xi_rho"), clim_krillmass_SO.lon_rho.data),
                                              lat_rho=(("eta_rho", "xi_rho"), clim_krillmass_SO.lat_rho.data),)
    
    actual_biomass = xr.open_dataset(files[1])
    actual_biomass = actual_biomass.assign_coords(lon_rho=(("eta_rho", "xi_rho"), actual_krillmass_SO.lon.data),
                                                  lat_rho=(("eta_rho", "xi_rho"), actual_krillmass_SO.lat.data),)
    
    nomhw_biomass = xr.open_dataset(files[2])
    nomhw_biomass = nomhw_biomass.assign_coords(lon_rho=(("eta_rho", "xi_rho"), noMHWs_krillmass_SO.lon.data),
                                                lat_rho=(("eta_rho", "xi_rho"), noMHWs_krillmass_SO.lat.data),)
    
    nowarming_biomass = xr.open_dataset(files[3])
    nowarming_biomass = nowarming_biomass.assign_coords(lon_rho=(("eta_rho", "xi_rho"), nowarming_krillmass_SO.lon.data),
                                                lat_rho=(("eta_rho", "xi_rho"), nowarming_krillmass_SO.lat.data),)

# %% ================================= Plot climatologcical biomass =================================
# Plot initial and final climatological biomass (2 columns, 1 row)
# --- Prepare data
biomass_init = clim_biomass.biomass_median.isel(days=0).isel(xi_rho=slice(0, -1))    # 1st Nov
biomass_final = clim_biomass.biomass_median.isel(days=-1).isel(xi_rho=slice(0, -1)) # 30th Apr

data_list = [biomass_init, biomass_final]

# --- Figure setup
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.05)

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Color Setup
vmin, vmax = np.nanpercentile(np.stack([d.values for d in data_list]), [5, 95])
cmap = "Reds"

# --- Plot panels
for i, data in enumerate(data_list):
    ax = fig.add_subplot(gs[0, i], projection=ccrs.SouthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(linewidth=0.7, zorder=3)

    im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), zorder=1 )

    ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.3)
    ax.set_title("Initial (1 Nov)" if i == 0 else "Final (30 Apr)", fontsize=12)

# --- Colorbar
cax = fig.add_axes([0.92, 0.20, 0.02, 0.65])
plt.colorbar(im, cax=cax, extend='both').set_label("Biomass [mg m$^{-3}$]", fontsize=12)

# --- Title
plt.suptitle("Climatological Krill Biomass Concentration", fontsize=14, y=0.98)
plt.show()




# %% ================================= Plot Biomass concentration =================================
# --- Prepare data
dataset_interest =  nowarming_biomass #actual_biomass, nomhw_biomass
# title = "Environmental conditions without surface MHWs"
title = "Environmental conditions without global warming"
# title = "Actual environmental conditions"

# Years to show
years_to_plot = [1980, 1989, 2000, 2010, 2016]

# Row 1: Biomass 
initial_biomass = dataset_interest.biomass_median.isel(years=0, days=0).isel(xi_rho=slice(0, -1))  # 1st Nov
biomass_actual_30Apr = [dataset_interest.biomass_median.isel(years=i, days=-1).isel(xi_rho=slice(0, -1)) for i in range(len(years_to_plot))]  

# Row 2: Difference (end of season)
diff_actual = [biomass_actual_30Apr[i] - clim_biomass.biomass_median.isel(days=-1).isel(xi_rho=slice(0, -1)) for i in range(len(years_to_plot))]


# --- Figure setup
ncols = len(biomass_actual_30Apr)
fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(nrows=2, ncols=ncols, wspace=0.08, hspace=0.3)

# --- Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Color Setup
# Row 1: biomass
vmin_row1, vmax_row1 = np.nanpercentile(np.stack([d.values for d in biomass_actual_30Apr]), [5, 95])
cmap_row1 = 'Reds'

# Row 2: actual - clim
diff_stack2 = np.stack([d.values for d in diff_actual])
max_abs_diff2 = np.nanmax(np.abs(diff_stack2))
vmin_row2, vmax_row2 = -max_abs_diff2, max_abs_diff2
cmap_row2 = 'bwr'

# --- Row 1: Biomass
for i, data in enumerate(biomass_actual_30Apr):
    ax = fig.add_subplot(gs[0, i], projection=ccrs.SouthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(color='black', linewidth=0.7, zorder=3)

    im1 = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                        cmap=cmap_row1, vmin=vmin_row1, vmax=vmax_row1,
                        transform=ccrs.PlateCarree(), zorder=1)

    # if i == 0:
    gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 7, 'rotation': 0}
    gl.ylabel_style = {'size': 7, 'rotation': 0}
    
    ax.set_title(f'{years_to_plot[i]}', fontsize=14)

# --- Row 2: Difference
for i, diff in enumerate(diff_actual):
    ax = fig.add_subplot(gs[1, i], projection=ccrs.SouthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(color='black', linewidth=0.7, zorder=3)
    im2 = ax.pcolormesh(
        diff.lon_rho, diff.lat_rho, diff,
        cmap=cmap_row2, vmin=vmin_row2, vmax=vmax_row2,
        transform=ccrs.PlateCarree(), zorder=1
    )
    # if i == 0:
    gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 7, 'rotation': 0}
    gl.ylabel_style = {'size': 7, 'rotation': 0}

# --- Colorbars
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.35])
plt.colorbar(im1, cax=cbar_ax1, orientation='vertical', extend='both').set_label("Biomass [mg.m$^{-3}$]", fontsize=12)

cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.35])
plt.colorbar(im2, cax=cbar_ax2, orientation='vertical', extend='both').set_label("Biomass [mg.m$^{-3}$]", fontsize=12)

# --- Row titles
fig.text(0.52, 0.95, title, ha='center', fontsize=16)
fig.text(0.52, 0.5, "Comparison with Climatology ", ha='center', fontsize=16)

# --- Overall figure title
plt.suptitle("Krill Biomass on 30th April", fontsize=18, y=1.02, x=0.52)
plt.show()



# %%

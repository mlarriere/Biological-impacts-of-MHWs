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
         os.path.join(output_folder, "biomass_nowarming.nc"),]

# Run only if files don't exist
if all(os.path.exists(f) for f in files):
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
    actual_biomass = xr.open_dataset(files[1])
    nomhw_biomass = xr.open_dataset(files[2])
    nowarming_biomass = xr.open_dataset(files[3])


# -- Put to Datasets
# clim_biomass_ds = clim_biomass.to_dataset(name="biomass")
# actual_biomass_ds = actual_biomass.to_dataset(name="biomass")
# nomhw_biomass_ds = nomhw_biomass.to_dataset(name="biomass")
# nowarming_biomass_ds = nowarming_biomass.to_dataset(name="biomass")

# # Add metadata
# clim_biomass_ds = add_biomass_metadata(clim_biomass_ds, "Climatological environmental conditions.")
# actual_biomass_ds = add_biomass_metadata(actual_biomass_ds, "Actual environmental conditions.")
# nomhw_biomass_ds = add_biomass_metadata(nomhw_biomass_ds, "MHWs removed from temperature signal.")
# nowarming_biomass_ds = add_biomass_metadata(nowarming_biomass_ds, "Warming trend removed from temperature signal.")

# # -- Save to files - t redo
# clim_biomass_ds.to_netcdf(os.path.join(path_surrogates, "clim_biomass.nc"))
# actual_biomass_ds.to_netcdf(os.path.join(path_surrogates, "actual_biomass.nc"))
# nomhw_biomass_ds.to_netcdf(os.path.join(path_surrogates, "nomhws_biomass.nc"))
# nowarming_biomass_ds.to_netcdf(os.path.join(path_surrogates, "nowarming_biomass.nc"))


# %% ================================= Plot Biomass concentration =================================
# # Plot initial and final climatological biomass (2 columns, 1 row)
# # --- Prepare data
# biomass_init = clim_biomass_ds.biomass.isel(days=0)    # 1st Nov
# biomass_final = clim_biomass_ds.biomass.isel(days=-1) # 30th Apr

# data_list = [biomass_init, biomass_final]

# # --- Figure setup
# fig = plt.figure(figsize=(10, 5))
# gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.05)

# # Circular boundary
# theta = np.linspace(0, 2 * np.pi, 200)
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * 0.5 + 0.5)

# # --- Color Setup
# vmin, vmax = np.nanpercentile(np.stack([d.values for d in data_list]), [5, 95])
# cmap = "Reds"

# # --- Plot panels
# for i, data in enumerate(data_list):
#     ax = fig.add_subplot(gs[0, i], projection=ccrs.SouthPolarStereo())
#     ax.set_boundary(circle, transform=ax.transAxes)
#     ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
#     ax.coastlines(linewidth=0.7, zorder=3)

#     im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
#                        cmap=cmap, vmin=vmin, vmax=vmax,
#                        transform=ccrs.PlateCarree(), zorder=1 )

#     ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.3)
#     ax.set_title("Initial (1 Nov)" if i == 0 else "Final (30 Apr)", fontsize=12)

# # --- Colorbar
# cax = fig.add_axes([0.92, 0.20, 0.02, 0.65])
# plt.colorbar(im, cax=cax, extend='both').set_label("Biomass [mg m$^{-3}$]", fontsize=12)

# # --- Title
# plt.suptitle("Climatological Krill Biomass Concentration", fontsize=14, y=0.98)
# plt.show()




# # %% ================================= Plot Biomass concentration =================================
# # --- Prepare data
# dataset_interest = actual_biomass_ds
# # Years to show
# years_to_plot = [1980, 1990, 2000, 2010, 2018]

# # Row 1: Biomass 
# initial_biomass = dataset_interest.biomass.isel(years=0, days=0)  # 1st Nov
# biomass_actual_30Apr = [dataset_interest.biomass.isel(years=i, days=-1) for i in range(len(years_to_plot))]  

# # Row 2: Difference (end of season)
# diff_actual = [biomass_actual_30Apr[i] - clim_biomass_ds.biomass.isel(days=-1) for i in range(len(years_to_plot))]


# # --- Figure setup
# ncols = len(biomass_actual_30Apr)
# fig = plt.figure(figsize=(20, 8))
# gs = gridspec.GridSpec(nrows=2, ncols=ncols, wspace=0.08, hspace=0.3)

# # --- Circular boundary
# theta = np.linspace(0, 2 * np.pi, 200)
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * 0.5 + 0.5)

# # --- Color Setup
# # Row 1: biomass
# vmin_row1, vmax_row1 = np.nanpercentile(np.stack([d.values for d in biomass_actual_30Apr]), [5, 95])
# cmap_row1 = 'Reds'

# # Row 2: actual - clim
# diff_stack2 = np.stack([d.values for d in diff_actual])
# max_abs_diff2 = np.nanmax(np.abs(diff_stack2))
# vmin_row2, vmax_row2 = -max_abs_diff2, max_abs_diff2
# cmap_row2 = 'bwr'

# # --- Row 1: Biomass
# for i, data in enumerate(biomass_actual_30Apr):
#     ax = fig.add_subplot(gs[0, i], projection=ccrs.SouthPolarStereo())
#     ax.set_boundary(circle, transform=ax.transAxes)
#     ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
#     ax.coastlines(color='black', linewidth=0.7, zorder=3)

#     im1 = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
#                         cmap=cmap_row1, vmin=vmin_row1, vmax=vmax_row1,
#                         transform=ccrs.PlateCarree(), zorder=1)

#     # if i == 0:
#     gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     gl.xlabel_style = {'size': 7, 'rotation': 0}
#     gl.ylabel_style = {'size': 7, 'rotation': 0}
    
#     ax.set_title(f'{years_to_plot[i]}', fontsize=14)

# # --- Row 2: Difference
# for i, diff in enumerate(diff_actual):
#     ax = fig.add_subplot(gs[1, i], projection=ccrs.SouthPolarStereo())
#     ax.set_boundary(circle, transform=ax.transAxes)
#     ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
#     ax.coastlines(color='black', linewidth=0.7, zorder=3)
#     im2 = ax.pcolormesh(
#         diff.lon_rho, diff.lat_rho, diff,
#         cmap=cmap_row2, vmin=vmin_row2, vmax=vmax_row2,
#         transform=ccrs.PlateCarree(), zorder=1
#     )
#     # if i == 0:
#     gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     gl.xlabel_style = {'size': 7, 'rotation': 0}
#     gl.ylabel_style = {'size': 7, 'rotation': 0}

# # --- Colorbars
# cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.35])
# plt.colorbar(im1, cax=cbar_ax1, orientation='vertical', extend='both').set_label("Biomass [mg.m$^{-3}$]", fontsize=12)

# cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.35])
# plt.colorbar(im2, cax=cbar_ax2, orientation='vertical', extend='both').set_label("Biomass [mg.m$^{-3}$]", fontsize=12)

# # --- Row titles
# fig.text(0.52, 0.95, "Actual environmental conditions", ha='center', fontsize=16)
# fig.text(0.52, 0.5, "Comparison with Climatology ", ha='center', fontsize=16)

# # --- Overall figure title
# plt.suptitle("Krill Biomass on 30th April", fontsize=18, y=1.02, x=0.52)
# plt.show()




# # %% ====================================
# #             Initial Biomass
# #                 1st Nov
# # =======================================
# # output_file_ini = os.path.join(path_surrogates, "krill_density/initial_krill_biomass.nc")
# # if not os.path.exists(output_file_ini):
# #     # -- Initialisation
# #     initial_density = xr.zeros_like(clim_krillmass_SO['juvenile'].isel(days=0))  # shape: (eta_rho, xi_rho)

# #     # -- Calculate initial density and biomass
# #     # abundance_day0 = abundance_regridded.euphausia_abundance.isel(bootstrap=0, days=0)
# #     initial_density, initial_biomass = density_biomass(clim_krillmass_SO, proportion, mean_abundance, area_60S_SO, first_day=True, last_day=False)

# #     # -- To Dataset
# #     initial_density.name = "density"
# #     initial_density.attrs = {"description": "Initial krill biomass density on 1st November (common for all years and conditions).",
# #                             "units": "mg/m²",}
# #     initial_density_ds = initial_density.to_dataset()

# #     initial_biomass.name = "biomass"
# #     initial_biomass.attrs = {"description": "Initial krill biomass on 1st November (common for all years and conditions).",
# #                             "units": "mg",}
# #     initial_biomass_ds = initial_biomass.to_dataset()

# #     # -- Merge Datasets
# #     initial_ds = xr.merge([initial_density_ds, initial_biomass_ds])

# #     # -- Save to file
# #     initial_ds.to_netcdf(output_file_ini)
# #     print(f"Saved initial krill biomass dataset.'")
# # else:
# #     print(f"File '{output_file_ini}' already exists.")
# #     initial_ds=xr.open_dataset(output_file_ini)

# #  ================================================
# #             Final Density and Biomass
# #                    30th April
# # ===================================================
# # ======= 1. Climatology =======
# # output_file_clim = os.path.join(path_biomass, "fake_worlds/biomass_density/clim_krill_biomass.nc")
# # if not os.path.exists(output_file_clim):
# #     # Calcualte density and biomass
# #     density_clim, biomass_clim = density_biomass(clim_krillmass_SO, proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True) #shape (231, 1442)

# #     # -- To Dataset
# #     density_clim.name = "density"
# #     density_clim.attrs = {"description": "Krill biomass density on 30th April, under climatological conditions.",
# #                             "units": "mg/m²",}
# #     density_clim_ds = density_clim.to_dataset()

# #     biomass_clim.name = "biomass"
# #     biomass_clim.attrs = {"description": "Krill biomass on 30th April, under climatological conditions.",
# #                             "units": "mg",}
# #     biomass_clim_ds = biomass_clim.to_dataset()

# #     # -- Merge Datasets
# #     clim_ds = xr.merge([density_clim_ds, biomass_clim_ds])

# #     # -- Save to file
# #     clim_ds.to_netcdf(output_file_clim)
# #     print(f"Saved initial krill biomass dataset.'")
# # else:
# #     # -- Load Dataset
# #     print(f"File '{output_file_clim}' already exists. Load data")
# #     clim_ds=xr.open_dataset(output_file_clim)
    
# # # ======= 2. Actual =======
# # output_file_actual = os.path.join(path_biomass, "fake_worlds/biomass_density/actual_krill_biomass.nc")
# # if not os.path.exists(output_file_actual):

# #     # Initialisation
# #     years = np.arange(1980, 2019) 
# #     actual_density_list = []
# #     actual_biomass_list = []

# #     for yr_idx, yr in enumerate(years):
# #         print(f"\nYear: {yr}")
# #         density_actual_yearly, biomass_actual_yearly = density_biomass(actual_krillmass_SO.isel(years=yr_idx), proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True)

# #         # Store result to list 
# #         actual_density_list.append(density_actual_yearly)
# #         actual_biomass_list.append(biomass_actual_yearly)

# #     # Combine results
# #     density_actual = xr.concat(actual_density_list, dim="years")
# #     biomass_actual = xr.concat(actual_biomass_list, dim="years")

# #     # -- To Dataset
# #     density_actual.name = "density"
# #     density_actual.attrs = {"description": "Krill biomass density on 30th April.",
# #                             "units": "mg/m²",}
# #     density_actual_ds = density_actual.to_dataset()

# #     biomass_actual.name = "biomass"
# #     biomass_actual.attrs = {"description": "Krill biomass on 30th April.",
# #                             "units": "mg",}
# #     biomass_actual_ds = biomass_actual.to_dataset()
    
# #     # -- Merge Datasets
# #     actual_ds = xr.merge([density_actual_ds, biomass_actual_ds])

# #     # -- Save to file
# #     actual_ds.to_netcdf(output_file_actual)
# #     print(f"Saved actual krill biomass dataset.'")
# # else:
# #     # -- Load Dataset
# #     print(f"File '{output_file_actual}' already exists. Load data")
# #     actual_ds=xr.open_dataset(output_file_actual)
    
# # # ======= 3. No MHWs =======
# # output_file_noMHWs = os.path.join(path_biomass, "fake_worlds/biomass_density/noMHWs_krill_biomass.nc")
# # if not os.path.exists(output_file_noMHWs):
# #     years = np.arange(1980, 2019) 
# #     noMHWs_density_list = []
# #     noMHWs_biomass_list = []

# #     for yr_idx, yr in enumerate(years):
# #         print(f"\nYear: {yr}")
# #         density_noMHWs_yearly, biomass_noMHWs_yearly = density_biomass(noMHWs_krillmass_SO.isel(years=yr_idx), proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True)

# #         # Store result to list 
# #         noMHWs_density_list.append(density_noMHWs_yearly)
# #         noMHWs_biomass_list.append(biomass_noMHWs_yearly)

# #     # Combine results
# #     density_noMHWs = xr.concat(noMHWs_density_list, dim="years")
# #     biomass_noMHWs = xr.concat(noMHWs_biomass_list, dim="years")

# #     # -- To Dataset
# #     density_noMHWs.name = "density"
# #     density_noMHWs.attrs = {'description': 'Krill biomass density on 30th April each year. The 100m-avg temperatures under surface MHWs have been replaced by 100m-avg temperature climatology to simulate a world without MHWs influence.',
# #                             "units": "mg/m²",}
# #     density_noMHWs_ds = density_noMHWs.to_dataset()

# #     biomass_noMHWs.name = "biomass"
# #     biomass_noMHWs.attrs = {'description': 'Krill biomass on 30th April each year. The 100m-avg temperatures under surface MHWs have been replaced by 100m-avg temperature climatology to simulate a world without MHWs influence.',
# #                             "units": "mg",}
# #     biomass_noMHWs_ds = biomass_noMHWs.to_dataset()

# #     # -- Merge Datasets
# #     noMHWs_ds = xr.merge([density_noMHWs_ds, biomass_noMHWs_ds])

# #     # -- Save to file
# #     noMHWs_ds.to_netcdf(output_file_noMHWs)
# #     print(f"Saved actual krill biomass dataset.'")
# # else:
# #     # -- Load Dataset
# #     print(f"File '{output_file_noMHWs}' already exists. Load data")
# #     noMHWs_ds=xr.open_dataset(output_file_noMHWs)
    
# # # ======= 4. No Warming =======
# # output_file_nowarming = os.path.join(path_biomass, "fake_worlds/biomass_density/no_warming_krill_biomass.nc")
# # if not os.path.exists(output_file_noMHWs):
# #     years = np.arange(1980, 2019) 
# #     nowarming_density_list = []
# #     nowarming_biomass_list = []

# #     for yr_idx, yr in enumerate(years):
# #         print(f"\nYear: {yr}")
# #         density_nowarming_yearly, biomass_nowarming_yearly = density_biomass(nowarming_krillmass_SO.isel(years=yr_idx), proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True)

# #         # Store result to list 
# #         nowarming_density_list.append(density_nowarming_yearly)
# #         nowarming_biomass_list.append(biomass_nowarming_yearly)

# #     # Combine results
# #     density_nowarming = xr.concat(nowarming_density_list, dim="years")
# #     biomass_nowarming = xr.concat(nowarming_biomass_list, dim="years")

# #     # -- To Dataset
# #     density_nowarming.name = "density"
# #     density_nowarming.attrs = {'description': 'Krill biomass density on 30th April each year. '
# #                                               'The 100m-avg temperatures are detrended.',
# #                             "units": "mg/m²",}
# #     density_nowarming_ds = density_nowarming.to_dataset()

# #     biomass_nowarming.name = "biomass"
# #     biomass_nowarming.attrs = {'description': 'Krill biomass on 30th April each year. '
# #                                               'The 100m-avg temperatures are detrended.',
# #                             "units": "mg",}
# #     biomass_nowarming_ds = biomass_nowarming.to_dataset()

# #     # -- Merge Datasets
# #     nowarming_ds = xr.merge([density_nowarming_ds, biomass_nowarming_ds])

# #     # -- Save to file
# #     nowarming_ds.to_netcdf(output_file_nowarming)
# #     print(f"Saved 'no warming' krill biomass dataset.'")
# # else:
# #     # -- Load Dataset
# #     print(f"File '{output_file_nowarming}' already exists. Load data")
# #     nowarming_ds=xr.open_dataset(output_file_nowarming)
    
# # %% ================================= Plot biomass concentration =================================
# # --- Prepare data for visualisation
# clim_density_diff = clim_ds.density - initial_ds.initial_density
# density1989_actual_diff = actual_ds.density.isel(years=9) - initial_ds.initial_density
# density2000_actual_diff = actual_ds.density.isel(years=20) - initial_ds.initial_density
# density2016_actual_diff = actual_ds.density.isel(years=36) - initial_ds.initial_density

# density1989_diff_clim = actual_ds.density.isel(years=9) - clim_ds.density
# density2000_diff_clim = actual_ds.density.isel(years=20) - clim_ds.density
# density2016_diff_clim = actual_ds.density.isel(years=36) - clim_ds.density

# # --- Layout config 
# plot = 'slides' #report slides

# if plot == 'report':
#     fig_width = 6.3228348611
#     fig_height = fig_width/1.5

# else:
#     fig_width = 12
#     fig_height = 8

# fig, axs = plt.subplots(2, 4, figsize=(fig_width * 1.2, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})
# plt.subplots_adjust(hspace=0.05, wspace=0.05)

# # Circular boundary
# theta = np.linspace(0, 2 * np.pi, 200)
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * 0.5 + 0.5)

# # --- Color setup
# from matplotlib.colors import LinearSegmentedColormap
# colors = ["#D1105A", "#F24B04", "#FFFFFF", "#884AB2", "#471CA8"]  
# # colors = ["#FF9E1F", "#FF930A","#F24B04", "#D1105A", "#AD2D86", "#471CA8", "#371F6F", "#2A1E48"]
# cmap_biomass = LinearSegmentedColormap.from_list("biomass", colors, N=256)
# # cmap_biomass = 'Purples'
# norm_biomass = mcolors.Normalize(vmin=50, vmax=300)  # Tonnes/km² or custom
# cmap_diff = 'coolwarm'
# norm_diff = mcolors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)
# norm_diff_clim = mcolors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)

# # --- Data setup 
# plot_data = [
#     # Row 1: Differences year
#     (clim_density_diff, "Climatology Change", cmap_biomass, norm_diff),
#     (density1989_actual_diff, "Season Change 1989-1990", cmap_biomass, norm_diff),
#     (density2000_actual_diff, "Season Change 2000-2001", cmap_biomass, norm_diff),
#     (density2016_actual_diff, "Season Change 2016-2017", cmap_biomass, norm_diff),

#     # Row 2: Differences clim
#     (None, 'None',None, None), 
#     (density1989_diff_clim, "1989-1990 vs Climatology", cmap_diff, norm_diff_clim),
#     (density2000_diff_clim, "2000-2001 vs Climatology", cmap_diff, norm_diff_clim),
#     (density2016_diff_clim, "2016-2017 vs Climatology", cmap_diff, norm_diff_clim),
# ]

# # --- Plotting 
# ims = []
# for i, (data, title, cmap, norm) in enumerate(plot_data):
#     row = i // 4
#     col = i % 4
#     ax = axs[row, col]

#     if data is None:
#         ax.axis('off')
#         continue

#     im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
#                        transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
#                        shading='auto', rasterized=True)
#     ims.append(im)

#     ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
#     ax.set_boundary(circle, transform=ax.transAxes)
#     ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
#     ax.coastlines(color='black', linewidth=0.4, zorder=5)
#     ax.set_facecolor('#F6F6F3')

#     # Gridlines
#     import matplotlib.ticker as mticker
#     gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.7, linestyle='--', linewidth=0.4, zorder=7)
#     gl.xlocator = mticker.FixedLocator(np.arange(-80, 1, 20))
#     gl.xformatter = LongitudeFormatter(degree_symbol='°', number_format='.0f', dateline_direction_label=False)
#     gl.yformatter = LatitudeFormatter()
#     gl.xlabels_top = False
#     gl.xlabels_bottom = False
#     gl.ylabels_right = False
#     gl.xlabels_left = True
#     gl.xlabel_style = {'size': 6, 'rotation': 0}
#     gl.ylabel_style = {'size': 6, 'rotation': 0}

#     ax.set_title(title, fontsize=10)

# # --- Colorbars 
# # Biomass colorbar (top row)
# cbar_ax1 = fig.add_axes([0.92, 0.56, 0.015, 0.35])
# cbar1 = fig.colorbar(ims[0], cax=cbar_ax1, extend='both')
# cbar1.set_label("Density [mg/m2]", fontsize=10)
# cbar1.ax.tick_params(labelsize=8)

# # Changes Biomass colorbar (bottom row)
# cbar_ax2 = fig.add_axes([0.92, 0.12, 0.015, 0.35])
# cbar2 = fig.colorbar(ims[5], cax=cbar_ax2, extend='both')
# cbar2.set_label(r"$\Delta$ Density [mg/m2]", fontsize=10)
# cbar2.ax.tick_params(labelsize=8)



# # --- Output handling ---
# if plot == 'report':
#     outdir = os.path.join(os.getcwd(), 'Biomass/figures_outputs/Biomass/')
#     os.makedirs(outdir, exist_ok=True)
#     outfile = f"biomass_differences_{plot}.pdf"
#     # plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
#     plt.show()
# else:
#     # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
#     plt.show()







# # %%

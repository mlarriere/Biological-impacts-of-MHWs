"""
Created on Tue 30 July 17:04:45 2025

Calculating the Biomass of E. Superba

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
path_biomass_ts = os.path.join(path_surrogates, f'biomass_timeseries')
path_biomass_ts_SO = os.path.join(path_biomass_ts, f'SouthernOcean')
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
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') #volume first 100m

# --- Southern Ocean south of 60°S
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)

# %% ======================== Biomass from CEPHALOPOD ========================
# -- Load data
# Biomass are climatological products [mgC/m3]
biomass_regridded = xr.open_dataset(os.path.join(path_cephalopod, 'euphausia_biomass_SO_regridded.nc')) #shape (181, 50, 231, 1442)
biomass_regrid_interp = xr.open_dataset(os.path.join(path_cephalopod, 'euphausia_biomass_SO_regrid_interp.nc')) #shape (181, 50, 231, 1442)
original_product = xr.open_dataset(os.path.join(path_cephalopod, 'total_krill_biomass_SO.nc'))

# -- Convert biomass from [mgC/m3] to [mg/m3]
# Carbon fraction in Euphausia Superba -- Data from Farber-Lorda et al. (2009)
C_frac_stage = {'juvenile': (0.4989, 0.0250), 'males': (0.4756, 0.0266), 'mature females': (0.4756, 0.0297), 'spent females': (0.5299, 0.0267),} # [fraction C / mg biomass]
mean_C_fraction = np.mean(np.array([v[0] for v in C_frac_stage.values()]))
propagated_sd = np.sqrt(np.sum(np.array([v[1] for v in C_frac_stage.values()])**2) / len(np.array([v[1] for v in C_frac_stage.values()])))
# print(f"Mean C fraction: {mean_C_fraction:.4f}")
# print(f"Propagated SD: {propagated_sd:.4f}")

biomass_regridded_dry = biomass_regridded / mean_C_fraction #[mg/m3]
biomass_regrid_interp_dry = biomass_regrid_interp / mean_C_fraction #[mg/m3]
print(f'Before: {biomass_regridded.isel(days=0, algo_bootstrap=0, eta_rho=200, xi_rho=1000).euphausia_biomass.values:.3f} mgC/m3')
print(f'After: {biomass_regridded_dry.isel(days=0, algo_bootstrap=0, eta_rho=200, xi_rho=1000).euphausia_biomass.values:.3f} mg/m3')


# %% ======================== Load data ========================
# --- Load mass data [mg] for each maturity stage -- Southern Ocean  
# 1. Climatology
clim_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_mass_stages_SO.nc")) #shape (181, 231, 1442)

# 2. Actual
actual_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "actual_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
actual_krillmass_SO = actual_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 3. No MHWs (MHWs replaced by clim)
noMHWs_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "noMHWs_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
noMHWs_krillmass_SO = noMHWs_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})
clim_trended_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_trended_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
clim_trended_krillmass_SO = clim_trended_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 3. No warming (temperature signal detrended)
nowarming_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "nowarming_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
nowarming_krillmass_SO = nowarming_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})


# %% ======================== Defining Functions ========================
from functools import partial
from tqdm.contrib.concurrent import process_map
import gc

def biomass_k(k, growth_fact_pop, B0_interp=False):

    n_days = 181
    
    if B0_interp:
        B0 = biomass_regrid_interp_dry.euphausia_biomass.isel(days=0, algo_bootstrap=k).values
    else:
        B0 = biomass_regridded_dry.euphausia_biomass.isel(days=0, algo_bootstrap=k).values

    B_k = np.empty((n_days, *B0.shape), dtype=np.float32)
    B_k[0] = B0

    for t in range(1, n_days):
        B_k[t] = B_k[t-1] * growth_fact_pop[t - 1]

    return B_k


def evolution_biomass_yr(year_idx, ds_mass, proportion, B0_interp=False):
    # test
    # year_idx = 36
    # ds_mass = clim_krillmass_SO
    # B0_interp=False

    # -- Climatology case, i.e. no years dim
    has_years = 'years' in ds_mass.dims
    n_days = 181

    # -- Compute growth factor
    growth_fact_pop = np.zeros((n_days-1, ds_mass.eta_rho.size, ds_mass.xi_rho.size), dtype=np.float32) #shape (180, 231, 1442)
    for stage, p in proportion.items():
        if has_years:
            # Specific year
            ds_mass_yr = ds_mass[stage].isel(years=year_idx).values
        else:
            ds_mass_yr = ds_mass[stage].values # climatology  

        g_fact_stage = ds_mass_yr[1:] / ds_mass_yr[:-1] 
        growth_fact_pop += p * g_fact_stage # shape (180, 231, 1442)

    # -- Biomass timeseries for every models and algorithms
    # Loop over the algo and boostraps -- in parallel
    func = partial(biomass_k, growth_fact_pop=growth_fact_pop, B0_interp=B0_interp)
    results = process_map(func, range(50), max_workers=10, chunksize=1, desc='Algo')

    # Extract data
    B_all = np.stack(results, axis=0) #shape (50, 181, 231, 1442)
 
    # To Datsaset
    biomass_ds = xr.Dataset(data_vars=dict(biomass=(("algo_bootstrap", "days", "eta_rho", "xi_rho"), B_all)),
                            coords=dict(algo_bootstrap=np.arange(50), 
                                        days=ds_mass.days,
                                        lat_rho=ds_mass.lat_rho,
                                        lon_rho=ds_mass.lon_rho,))
    # -- Take median and std
    biomass_stats_ds = xr.Dataset({"biomass_median": biomass_ds.biomass.median(dim="algo_bootstrap"),
                                   "biomass_std":    biomass_ds.biomass.std(dim="algo_bootstrap"),})

    # Add attributes
    biomass_stats_ds.biomass_median.attrs.update({"description": "Median over the 5 models × 10 bootstraps of Cephalopod.",
                                                  "units": "mg m-3",})
    biomass_stats_ds.biomass_std.attrs.update({"description": "Spread over 5 models × 10 bootstraps of Cephalopod.",
                                               "units": "mg m-3",})

    # Clean memory
    del B_all, biomass_ds
    gc.collect()

    return biomass_stats_ds


def compute_biomass_surrogates(ds_mass, label, label_da, proportion, output_folder, max_workers=10, B0_interp=False):
    # test
    # max_workers=20
    # proportion=proportion
    # ds_mass=nowarming_krillmass_SO
    # label='No Warming'  
    # label_da='nowarming'  
    # B0_interp=True
    # output_folder= os.path.join(path_surrogates, f'biomass_timeseries/biomass_interpolated')
    
    print(f"  → {label}")
    
    output_file= os.path.join(output_folder, f'biomass_{label_da}.nc')

    if not os.path.exists(output_file):
        # Prepare function
        func = partial(evolution_biomass_yr, ds_mass=ds_mass, proportion=proportion, B0_interp=B0_interp)

        # -- Climatology case, i.e. no years dim
        has_years = 'years' in ds_mass.dims

        if has_years:
            # Run in parallel
            B_algo_median_da = process_map(func, range(39), max_workers=max_workers, chunksize=1, desc=f"{label} | Biomass ") #len = nyears and shape (181, 231, 1442)

            # Extract data
            B_algo_median_all_ds = xr.concat(B_algo_median_da, dim="years")  # shape: (39, 181, 231, 1442)

            if B0_interp:
                initial_biomass_description = "Cephalopod output for krill total (euphausia biomass ~80% of it).\nRegridded to ROMS resolution (0.25°) and interpolated."

            else:
                initial_biomass_description="Cephalopod output for krill total (euphausia biomass ~80% of it).\nRegridded to ROMS resolution, i.e. 0.25°."


            # Put together into dataset
            B_algo_median_all_ds.attrs.update({"surrogate": label,
                                               "Description": "Evolution of krill biomass weighted according to population proportions.\n"\
                                                              "Biomass timeseries computed for each models and boostraps.\n"\
                                                              "Then, we compute the median and std over the 50 timeseries.",
                                               "Population Proportions": ", ".join(f"{k} : {v*100:.0f}%" for k, v in proportion.items()),
                                               "Initial Biomass": initial_biomass_description,
                                               "Assumptions": "Fixed stage proportions, no mortality, no recruitment, no stage transitions within a growth season.",
                                               "Units": "mg/m3",})
            
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

            if B0_interp:
                initial_biomass_description = "Cephalopod output for krill total (euphausia biomass ~80% of it).\nRegridded to ROMS resolution (0.25°) and interpolated."

            else:
                initial_biomass_description="Cephalopod output for krill total (euphausia biomass ~80% of it).\nRegridded to ROMS resolution, i.e. 0.25°."

            # Put together into dataset
            B_algo_median_all_ds.attrs.update({"surrogate": label,
                                               "Description": "Evolution of krill biomass weighted according to population proportions.\n"\
                                                              "Biomass timeseries computed for each models and boostraps.\n"\
                                                              "Then, we compute the median and std over the 50 timeseries.",
                                               "Population Proportions": ", ".join(f"{k} : {v*100:.0f}%" for k, v in proportion.items()),
                                               "Initial Biomass": initial_biomass_description,
                                               "Assumptions": "Fixed stage proportions, no mortality, no recruitment, no stage transitions within a growth season.",
                                               "Units": "mg/m3",
                                               'Conversion to dry weight': f"Using mean C fraction of {mean_C_fraction:.4f} (SD: {propagated_sd:.4f}) from Farber-Lorda et al. (2009).",})

            # Save to file
            B_algo_median_all_ds.to_netcdf(output_file)

            # Clean memory
            del B_algo_median_da
            gc.collect()

    else:
        print(f"  → {label} already exists, skipping.")
        return

# %% ==================================== Compute Biomass for surrogates - Regridded ====================================
print('\nInitial biomass: Regridded')

output_folder_regrid=os.path.join(path_biomass_ts_SO, f'biomass_regridded')
files_regrid = [os.path.join(output_folder_regrid, "biomass_clim.nc"),
                os.path.join(output_folder_regrid, "biomass_actual.nc"),
                os.path.join(output_folder_regrid, "biomass_nomhws.nc"),
                os.path.join(output_folder_regrid, "biomass_clim_trend.nc"),
                os.path.join(output_folder_regrid, "biomass_nowarming.nc")]

if not all(os.path.exists(f) for f in files_regrid):
    print("Missing output files → running full processing pipeline...\n")

    # ---------- Surrogate 1: Climatology
    clim_biomass = compute_biomass_surrogates(clim_krillmass_SO, label="Climatology", label_da='clim', proportion=proportion, 
                                              output_folder=output_folder_regrid, max_workers=20, B0_interp=False)
    
    # ---------- Surrogate 2: Actual
    actual_biomass = compute_biomass_surrogates(actual_krillmass_SO, label="Actual", label_da='actual', proportion=proportion, 
                                                output_folder=output_folder_regrid, max_workers=20, B0_interp=False)
    
    # ---------- Surrogate 3: No MHWs
    nomhw_biomass = compute_biomass_surrogates(noMHWs_krillmass_SO, label="No MHWs", label_da='nomhws', proportion=proportion, 
                                               output_folder=output_folder_regrid, max_workers=20, B0_interp=False)
    
    clim_trended_biomass = compute_biomass_surrogates(clim_trended_krillmass_SO, label="Clim Trended", label_da='clim_trend', proportion=proportion, 
                                                             output_folder=output_folder_regrid, max_workers=20, B0_interp=False)
    # ---------- Surrogate 4: No Warming
    nowarming_biomass = compute_biomass_surrogates(nowarming_krillmass_SO, label="No Warming", label_da='nowarming', proportion=proportion, 
                                                   output_folder=output_folder_regrid, max_workers=20, B0_interp=False)

else:
    print("All files already written → Load data")
    # Load data
    clim_biomass = xr.open_dataset(files_regrid[0])
    actual_biomass = xr.open_dataset(files_regrid[1])
    nomhw_biomass = xr.open_dataset(files_regrid[2])
    clim_trended_biomass = xr.open_dataset(files_regrid[3])
    nowarming_biomass = xr.open_dataset(files_regrid[4])


# %% ==================================== Compute Biomass for surrogates - Regridded and Interpolated ====================================
print('\nInitial biomass: Regridded and Interpolated')

output_folder_interp=os.path.join(path_biomass_ts_SO, f'biomass_interpolated')
files_interp = [os.path.join(output_folder_interp, "biomass_clim.nc"),
                os.path.join(output_folder_interp, "biomass_actual.nc"),
                os.path.join(output_folder_interp, "biomass_nomhws.nc"),
                os.path.join(output_folder_interp, "biomass_clim_trend.nc"),
                os.path.join(output_folder_interp, "biomass_nowarming.nc")]

if not all(os.path.exists(f) for f in files_interp):
    print("Missing output files → running full processing pipeline...\n")

    # ---------- Surrogate 1: Climatology
    clim_biomass_interp = compute_biomass_surrogates(clim_krillmass_SO, label="Climatology", label_da='clim', proportion=proportion, 
                                                     output_folder=output_folder_interp, max_workers=20, B0_interp=True)
    
    # ---------- Surrogate 2: Actual
    actual_biomass_interp = compute_biomass_surrogates(actual_krillmass_SO, label="Actual", label_da='actual', proportion=proportion, 
                                                        output_folder=output_folder_interp, max_workers=20, B0_interp=True)

    # ---------- Surrogate 3: No MHWs
    nomhw_biomass_interp = compute_biomass_surrogates(noMHWs_krillmass_SO, label="No MHWs", label_da='nomhws', proportion=proportion, 
                                                      output_folder=output_folder_interp, max_workers=20, B0_interp=True)

    clim_trended_biomass_interp = compute_biomass_surrogates(clim_trended_krillmass_SO, label="Clim Trended", label_da='clim_trend', proportion=proportion, 
                                                             output_folder=output_folder_interp, max_workers=20, B0_interp=True)
    
    # ---------- Surrogate 4: No Warming
    nowarming_biomass_interp = compute_biomass_surrogates(nowarming_krillmass_SO, label="No Warming", label_da='nowarming', proportion=proportion, 
                                                   output_folder=output_folder_interp, max_workers=20, B0_interp=True)
    
    
else:
    print("All files already written → Load data")
    # Load data
    clim_biomass_interp = xr.open_dataset(files_interp[0])
    actual_biomass_interp = xr.open_dataset(files_interp[1])
    nomhw_biomass_interp = xr.open_dataset(files_interp[2])
    clim_trended_biomass_interp= xr.open_dataset(files_interp[3])
    nowarming_biomass_interp = xr.open_dataset(files_interp[4])

# %% ================================= Plot climatologcical biomass =================================
# Plot initial and final climatological biomass (2 columns, 1 row)
# --- Prepare data
B0 = clim_biomass.biomass_median.isel(days=0).isel(xi_rho=slice(0, -1))    # 1st Nov
B0_interp = clim_biomass_interp.biomass_median.isel(days=0).isel(xi_rho=slice(0, -1))    # 1st Nov
Bfinal = clim_biomass.biomass_median.isel(days=-1).isel(xi_rho=slice(0, -1)) # 30th Apr
Bfinal_interp = clim_biomass_interp.biomass_median.isel(days=-1).isel(xi_rho=slice(0, -1)) # 30th Apr

# Row 0: regridded only
# Row 1: regridded + interpolated
data_grid = [
    [B0,        Bfinal],        # regridded
    [B0_interp, Bfinal_interp]  # regridded + interpolated
]

row_labels = ["Regridded", "Regridded + interpolated"]
col_labels = ["Initial (1 Nov)", "Final (30 Apr)"]

# --- Figure setup
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.02, hspace=0.15)

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Color Setup
vmin, vmax = np.nanpercentile(np.stack([d.values for row in data_grid for d in row]), [5, 95])
cmap = "Reds"

# --- Plot panels
for r in range(2):
    for c in range(2):
        data = data_grid[r][c]

        ax = fig.add_subplot(gs[r, c], projection=ccrs.SouthPolarStereo())
        ax.set_boundary(circle, transform=ax.transAxes)

        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.7, zorder=3)

        im = ax.pcolormesh(data.lon_rho, data.lat_rho, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), zorder=1)

        gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 7, 'rotation': 0}
        gl.ylabel_style = {'size': 7, 'rotation': 0}
    
        # Column titles
        if r == 0:
            ax.set_title(col_labels[c], fontsize=12)

        # Row labels
        if c == 0:
            ax.text(
                -0.09, 0.5, row_labels[r],
                va="center", ha="right",
                rotation=90,
                transform=ax.transAxes,
                fontsize=12
            )

# --- Colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.74]) #(left, bottom, width, height)
plt.colorbar(im, cax=cax, extend='both').set_label("Biomass [mg m$^{-3}$]", fontsize=12)

# --- Title
plt.suptitle("Climatological Krill Biomass Concentration", fontsize=14, y=0.98, x=0.52)
plt.show()




# %% ================================= Plot Biomass concentration =================================
# --- Prepare data
dataset_interest =  nomhw_biomass_interp #actual_biomass_interp #nowarming_biomass_interp #clim_trended_biomass_interp 
# title = "Environmental conditions without MHWs\nClimatological signal trended"
title = "Environmental conditions without MHWs"
# title = "Environmental conditions without global warming"
# title = "Actual environmental conditions"

# Years to show
years_to_plot = [1980, 1989, 2000, 2010, 2016]

# Row 1: Biomass 
initial_biomass = dataset_interest.biomass_median.isel(years=0, days=0).isel(xi_rho=slice(0, -1))  # 1st Nov
biomass_actual_30Apr = [dataset_interest.biomass_median.isel(years=i, days=-1).isel(xi_rho=slice(0, -1)) for i in range(len(years_to_plot))]  

# Row 2: Difference (end of season)
diff_actual = [biomass_actual_30Apr[i] - clim_biomass_interp.biomass_median.isel(days=-1).isel(xi_rho=slice(0, -1)) for i in range(len(years_to_plot))]


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

    im2 = ax.pcolormesh(diff.lon_rho, diff.lat_rho, diff,
                        cmap=cmap_row2, vmin=vmin_row2, vmax=vmax_row2,
                        transform=ccrs.PlateCarree(), zorder=1)

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
fig.text(0.52, 0.48, "Comparison with Climatology ", ha='center', fontsize=16)

# --- Overall figure title
plt.suptitle("Krill Biomass on 30th April", fontsize=18, y=1.04, x=0.52)
plt.show()



# %%

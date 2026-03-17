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

# %% -------------------------------- Figure settings --------------------------------
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
biomass_regrid_interp = xr.open_dataset(os.path.join(path_cephalopod, 'regrid_interp/euphausia_biomass_no_eke.nc')) #shape (181, 10, 231, 1442)

# -- Convert biomass from [mgC/m3] to [mg/m3]
# Carbon fraction in Euphausia Superba -- Data from Farber-Lorda et al. (2009)
C_frac_stage = {'juvenile': (0.4989, 0.0250), 'males': (0.4756, 0.0266), 'mature females': (0.4756, 0.0297), 'spent females': (0.5299, 0.0267),} # [fraction C / mg biomass]
mean_C_fraction = np.mean(np.array([v[0] for v in C_frac_stage.values()]))
propagated_sd = np.sqrt(np.sum(np.array([v[1] for v in C_frac_stage.values()])**2) / len(np.array([v[1] for v in C_frac_stage.values()])))

biomass_regrid_interp_dry = biomass_regrid_interp / mean_C_fraction #[mg/m3]

# %% ======================== Load data ========================
# --- Load mass data [mg] for each maturity stage -- Southern Ocean  
# 1. Climatology
clim_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_mass_stages_SO.nc")) #shape (181, 231, 1442)
clim_krillmass_SO = clim_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 2. Actual
actual_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "actual_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
actual_krillmass_SO = actual_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 3. No MHWs (MHWs replaced by clim)
clim_trended_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_trended_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
clim_trended_krillmass_SO = clim_trended_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})

# 4. No warming (temperature signal detrended)
nowarming_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "nowarming_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)
nowarming_krillmass_SO = nowarming_krillmass_SO.rename({"lat": "lat_rho", "lon": "lon_rho"})


# %% ======================== Defining Functions ========================
from functools import partial
from tqdm.contrib.concurrent import process_map
import gc

def biomass_k(k, growth_fact_pop):
    # k=0
    n_days = 181
    
    B0 = biomass_regrid_interp_dry.euphausia_biomass.isel(days=0, bootstraps=k).values

    B_k = np.empty((n_days, *B0.shape), dtype=np.float32)
    B_k[0] = B0

    for t in range(1, n_days):
        B_k[t] = B_k[t-1] * growth_fact_pop[t - 1]

    return B_k


def evolution_biomass_yr(year_idx, ds_mass, proportion):
    # test
    # year_idx = 36
    # ds_mass = clim_krillmass_SO

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
    func = partial(biomass_k, growth_fact_pop=growth_fact_pop)
    results = process_map(func, range(10), max_workers=10, chunksize=1, desc='Bootstraps')

    # Extract data
    B_all = np.stack(results, axis=0) #shape (10, 181, 231, 1442)
    
    # To Datsaset
    biomass_ds = xr.Dataset(data_vars=dict(biomass=(("bootstraps", "days", "eta_rho", "xi_rho"), B_all)),
                            coords=dict(bootstraps=np.arange(10), 
                                        days=ds_mass.days,
                                        lat_rho=ds_mass.lat_rho,
                                        lon_rho=ds_mass.lon_rho,))
    
    biomass_ds.attrs.update({"Description": "Evolution of krill biomass weighted according to population proportions.\n"\
                                            "Biomass time series were computed for all 50 algorithm–bootstrap "\
                                            "realisations of the Cephalopod model. For each algorithm, the "\
                                            "bootstrap realisations were then averaged to obtain a single time series per algorithm.",
                            "Units": "mg/m3",})
    
    
    # Clean memory
    del B_all
    gc.collect()

    return biomass_ds


def compute_biomass_surrogates(ds_mass, label, label_da, proportion, output_folder, max_workers=10):
    # test
    # max_workers=10
    # proportion=proportion
    # ds_mass=clim_krillmass_SO_expanded
    # label='Climatology'  
    # label_da='clim'  
    # output_folder= os.path.join(path_biomass_ts_SO)
    
    print(f"  → {label}")
    
    output_file= os.path.join(output_folder, f'biomass_{label_da}.nc')

    if not os.path.exists(output_file):
        # Prepare function
        func = partial(evolution_biomass_yr, ds_mass=ds_mass, proportion=proportion)

        # -- Climatology case, i.e. no years dim
        if 'years' not in ds_mass.dims:
            raise ValueError(f"ds_mass for '{label}' has no 'years' dimension.")

        # Run in parallel
        B_algo_da = process_map(func, range(39), max_workers=max_workers, chunksize=1, desc=f"{label} | Biomass ") #len = nyears and shape (10, 181, 231, 1442)

        # Extract data
        B_algo_all_ds = xr.concat(B_algo_da, dim="years")  # shape: (39, 10, 181, 231, 1442)

        # Put together into dataset
        B_algo_all_ds.attrs.update({"Surrogate": label,
                                    "Description": "Evolution of krill biomass weighted according to population proportions.\n"\
                                                    "Biomass timeseries computed for each bootstrap of the CEPHALOPOD model output.",
                                    "Population Proportions": ", ".join(f"{k} : {v*100:.0f}%" for k, v in proportion.items()),
                                    "Initial Biomass": "Cephalopod output for krill total (euphausia biomass ~80% of it).\nRegridded to ROMS resolution (0.25°) and interpolated.",
                                    "Assumptions": "Fixed stage proportions, no mortality, no recruitment, no stage transitions within a growth season.",
                                    "Units": "mg/m3",
                                    'Conversion to dry weight': f"Using mean C fraction of {mean_C_fraction:.4f} (SD: {propagated_sd:.4f}) from Farber-Lorda et al. (2009).",})
            
        # Save to file
        B_algo_all_ds.to_netcdf(output_file, mode="w", engine="netcdf4")
        
        # Clean memory
        del B_algo_da
        gc.collect()


    else:
        print(f"  → {label} already exists, skipping.")
        return
# %% ==================================== Compute Biomass for surrogates - Regridded and Interpolated ====================================
files_interp = [os.path.join(path_biomass_ts_SO, "biomass_clim.nc"),
                os.path.join(path_biomass_ts_SO, "biomass_actual.nc"),
                os.path.join(path_biomass_ts_SO, "biomass_clim_trend.nc"),
                os.path.join(path_biomass_ts_SO, "biomass_nowarming.nc")]

if not all(os.path.exists(f) for f in files_interp):
    print("Missing output files → running full processing pipeline...\n")

    # ---------- Surrogate 1: Climatology
    clim_biomass_interp = compute_biomass_surrogates(clim_krillmass_SO, label="Climatology", label_da='clim', proportion=proportion, 
                                                     output_folder=path_biomass_ts_SO, max_workers=20)
    
    # ---------- Surrogate 2: Actual
    actual_biomass_interp = compute_biomass_surrogates(actual_krillmass_SO, label="Actual", label_da='actual', proportion=proportion, 
                                                        output_folder=path_biomass_ts_SO, max_workers=20)

    # ---------- Surrogate 3: No MHWs
    clim_trended_biomass_interp = compute_biomass_surrogates(clim_trended_krillmass_SO, label="Clim Trended", label_da='clim_trend', proportion=proportion, 
                                                             output_folder=path_biomass_ts_SO, max_workers=20)
    
    # ---------- Surrogate 4: No Warming
    nowarming_biomass_interp = compute_biomass_surrogates(nowarming_krillmass_SO, label="No Warming", label_da='nowarming', proportion=proportion, 
                                                   output_folder=path_biomass_ts_SO, max_workers=20)
    

else:
    print("All files already written → Load data")
    # Load data
    clim_biomass_interp = xr.open_dataset(files_interp[0])
    actual_biomass_interp = xr.open_dataset(files_interp[1])
    clim_trended_biomass_interp= xr.open_dataset(files_interp[2])
    nowarming_biomass_interp = xr.open_dataset(files_interp[3])


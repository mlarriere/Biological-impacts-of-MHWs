"""
Created on Wed 10 Dec 15:14:30 2025

Biomass and biomass changes inside MPAs

@author: Marguerite Larriere (mlarriere)
"""

# %% ======================== PACKAGES========================
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob
import collections

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


from datetime import datetime, timedelta
import time
from tqdm.contrib.concurrent import process_map

from joblib import Parallel, delayed

#%% ======================== Server ======================== 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% ======================== Figure settings ========================
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
# %% ======================== SETTINGS ========================
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
roms_bathymetry = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc').h
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
path_biomass= '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_cases= '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/fake_worlds'
path_biomass_cases= '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/fake_worlds/biomass_density'

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
doy_list = list(range(304, 364)) + list(range(0, 121)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)



# %% ======================== Defining MPAs ========================
# == Load data
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)

# == Fix extent 
# South of 60°S
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

# == Settings plot
mpa_dict = {
    "Ross Sea": (mpas_ds.mask_rs, "#5F0F40"),
    "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#FFBA08"),
    "East Antarctic": (mpas_ds.mask_ea, "#E36414"),
    "Weddell Sea": (mpas_ds.mask_ws, "#4F772D"),
    "Antarctic Peninsula": (mpas_ds.mask_ap, "#0A9396")
}


# %% ======================== Plot MPAs ========================
fig = plt.figure(figsize=(5, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

# Base map
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# == Plot MPA
lon = mpas_ds.lon_rho
lat = mpas_ds.lat_rho
for name, (mask, color) in mpa_dict.items():

    # Mask lon/lat
    lon_masked = lon.where(mask)
    lat_masked = lat.where(mask)

    ax.scatter(lon_masked.values, lat_masked.values,
               s=2, color=color, transform=ccrs.PlateCarree(), 
               label=name, alpha=0.8, zorder=1)

    # == Add name of the MPA
    # Center of the box
    lon_centroid = float(lon_masked.mean().values)+10
    lat_centroid = float(lat_masked.mean().values)

    # Add text box with colored border
    ax.text(
        lon_centroid,
        lat_centroid,
        name,
        transform=ccrs.PlateCarree(),
        fontsize=10,
        fontweight='bold',
        ha='center',
        va='center',
        bbox=dict(
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            boxstyle="round,pad=0.3",
            alpha=0.9
        ),
        zorder=5
    )

# Legend and title
# ax.legend(loc="lower left", fontsize=9)
ax.set_title("Southern Ocean MPAs", fontsize=12)

plt.tight_layout()
plt.show()

# %% ======================== Areas MPAs ========================
# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)

# --- Calcaulte area of each MPA
area_mpa_ap = area_60S_SO.where(mpas_south60S.mask_ap) #Antarctic Peninsula
area_mpa_rs = area_60S_SO.where(mpas_south60S.mask_rs) #Ross Sea
area_mpa_o = area_60S_SO.where(mpas_south60S.mask_o) #South Orkney Islands southern shelf
area_mpa_ea = area_60S_SO.where(mpas_south60S.mask_ea) #East Antarctic
area_mpa_ws = area_60S_SO.where(mpas_south60S.mask_ws) #Weddell Sea

# %% ======================== Load Biomass ========================
# ==== Load data ====
# -- Southern Ocean (from Biomass_calculations.py)
biomass_initial = xr.open_dataset(os.path.join(path_biomass_cases, "initial_krill_biomass.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_clim = xr.open_dataset(os.path.join(path_biomass_cases, "clim_krill_biomass.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_actual = xr.open_dataset(os.path.join(path_biomass_cases, "actual_krill_biomass.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_noMHWs = xr.open_dataset(os.path.join(path_biomass_cases, "noMHWs_krill_biomass.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_nowarming = xr.open_dataset(os.path.join(path_biomass_cases, "no_warming_krill_biomass.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

# ==== Mask MPAs ====
# 1. Initial Biomass (1st Nov)
biomass_initial_MPAs = biomass_initial.initial_biomass.where(mpas_south60S)
rename_dict = {var: var.replace("mask_", "biomass_") for var in biomass_initial_MPAs.data_vars if var.startswith("mask_")}
biomass_initial_MPAs = biomass_initial_MPAs.rename(rename_dict)

# 1. Climatological Biomass (30th April)
biomass_clim_MPAs = biomass_clim.biomass.where(mpas_south60S)
rename_dict = {var: var.replace("mask_", "biomass_") for var in biomass_clim_MPAs.data_vars if var.startswith("mask_")}
biomass_clim_MPAs = biomass_clim_MPAs.rename(rename_dict)

# 2. Actual Biomass (30th April)
biomass_actual_MPAs = biomass_actual.biomass.where(mpas_south60S)
rename_dict = {var: var.replace("mask_", "biomass_") for var in biomass_actual_MPAs.data_vars if var.startswith("mask_")}
biomass_actual_MPAs = biomass_actual_MPAs.rename(rename_dict)

# 3. No MHWs Biomass (30th April)
biomass_noMHWs_MPAs = biomass_noMHWs.biomass.where(mpas_south60S)
rename_dict = {var: var.replace("mask_", "biomass_") for var in biomass_noMHWs_MPAs.data_vars if var.startswith("mask_")}
biomass_noMHWs_MPAs = biomass_noMHWs_MPAs.rename(rename_dict)

# 4. No Warming Biomass (30th April)
biomass_nowarming_MPAs = biomass_nowarming.biomass.where(mpas_south60S)
rename_dict = {var: var.replace("mask_", "biomass_") for var in biomass_nowarming_MPAs.data_vars if var.startswith("mask_")}
biomass_nowarming_MPAs = biomass_nowarming_MPAs.rename(rename_dict)


# %% ================================= Total Biomass =================================
# Multiply by area to get biomass per grid cell and then sum
# ----- Initial Biomass 
initial_biomass_kg = biomass_initial_MPAs.sum(dim=['eta_rho', 'xi_rho'])*1e-6

# ----- 1. Final Biomass Climatology
final_biomass_clim_kg = biomass_clim_MPAs.sum(dim=['eta_rho', 'xi_rho'])*1e-6

# ----- 2. Final Biomass Actual 
final_biomass_actual_kg = biomass_actual_MPAs.sum(dim=['eta_rho', 'xi_rho'])*1e-6

# ----- 3. Final Biomass No MHWs 
final_biomass_noMHWs_kg = biomass_noMHWs_MPAs.sum(dim=['eta_rho', 'xi_rho'])*1e-6

# ----- 4. Final Biomass No Warming 
final_biomass_nowarming_kg = biomass_nowarming_MPAs.sum(dim=['eta_rho', 'xi_rho'])*1e-6

# %% ================== Seasonal Biomass Gains ==================
DeltaB_clim = final_biomass_clim_kg - initial_biomass_kg
DeltaB_actual = final_biomass_actual_kg - initial_biomass_kg
DeltaB_noMHWs = final_biomass_noMHWs_kg - initial_biomass_kg
DeltaB_nowarming = final_biomass_nowarming_kg - initial_biomass_kg

# %% ================== Differences in seasonal biomass gains ==================
# -- Percentage change in seasonal krill biomass gain relative to climatology:
# 1. In the 'real' world
perc_actual = (DeltaB_actual-DeltaB_clim)/DeltaB_clim *100

# 2. In a world without MHWs
perc_noMHWs = (DeltaB_noMHWs-DeltaB_clim)/DeltaB_clim *100

# 3. In a world without warming (variability-only)
perc_nowarming = (DeltaB_nowarming-DeltaB_clim)/DeltaB_clim *100

# -- Contributions [%]
perc_MHW = perc_actual - perc_noMHWs
perc_warming = perc_actual - perc_nowarming

# mean_MHW_influence = (perc_MHW.to_array(dim="MPA").mean(dim=["MPA", "years"]))
# mean_warming_influence = (perc_warming.to_array(dim="MPA").mean(dim=["MPA", "years"]))
# print(f"Mean MHW influence (over years and MPAs) on seasonal gain: {mean_MHW_influence:.3f} %")
# print(f"Mean warming influence (over years and MPAs) on seasonal gain: {mean_warming_influence:.3f} %")

# -- Normalize, i.e. actual gain = 100% and then reduce and add relative to it
perc_MHW_norm = (DeltaB_actual - DeltaB_noMHWs) / DeltaB_actual * 100
perc_warming_norm = (DeltaB_actual - DeltaB_nowarming) / DeltaB_actual * 100

# %% ================== Plots ==================
MPA_dict = {
    "biomass_rs": ("Ross Sea", "#5F0F40"),
    "biomass_o": ("South Orkney Islands", "#FFBA08"),
    "biomass_ea": ("East Antarctic", "#E36414"),
    "biomass_ws": ("Weddell Sea", "#4F772D"),
    "biomass_ap": ("Antarctic Peninsula", "#0A9396")
}

plt.figure(figsize=(12,6))

plt.plot(perc_MHW_norm.years, perc_MHW_norm.biomass_rs)
plt.plot(perc_MHW_norm.years, perc_warming_norm.biomass_rs)

plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
plt.title("Change in Krill Seasonal Biomass Gain: Actual vs Climatology", fontsize=14, weight='bold')
plt.xlabel("Year", fontsize=13)
plt.ylabel("$\Delta$ [\%]", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.35)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# %%

years = DeltaB_actual.years.values 
warming_arr = (DeltaB_actual - DeltaB_nowarming).biomass_rs
mhw_arr = (DeltaB_nowarming - DeltaB_noMHWs).biomass_rs

actual_arr = DeltaB_actual.biomass_rs  # total gain

# ---- Plot ----
fig, ax = plt.subplots(figsize=(14,6))

# Stacked bars: MHWs at bottom, Warming on top
ax.bar(years, mhw_arr, label="MHWs", color="skyblue")
ax.bar(years, warming_arr, bottom=mhw_arr, label="Warming", color="tomato")

ax.plot(years, actual_arr, color="black", lw=2, label="Actual seasonal gain")

ax.set_title("Southern Ocean Krill Seasonal Biomass Gain (Nov → Apr)\nMPA: Ross Sea", fontsize=14, weight='bold')
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Seasonal gain [kg]", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()



# %%

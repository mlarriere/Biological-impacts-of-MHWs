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
path_mass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/krill_mass'
path_length = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/krill_length'


# %% ======================== Parameters ========================
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP according to Tarling et al 2006 - graph under 0°C
# stage_IMP = {'juvenile': 1, 'immature': 1, 'mature': 1, 'gravid': 1} # Remove IMP as temperature dependent (not constant)
proportion = {'juvenile': 0.20, 'immature': 0.3, 'mature': 0.3, 'gravid':0.2}

# %% ======================== Area ========================
# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc')

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)

# %% ======================== Abundance from CEPHALOPOD ========================
# N = 17.85 # Abundance ind/m2

# Abundance and biomass are climatological products
output_file_regrid = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass/CEPHALOPOD/1st_run/euphausia_output/regridded_outputs'
output_file_abundance_regridd= os.path.join(output_file_regrid, "euphausia_abundance_SO_regridded.nc")
output_file_biomass_regridd= os.path.join(output_file_regrid, "euphausia_biomass_SO_regridded.nc")

abundance_regridded = xr.open_dataset(output_file_abundance_regridd)
print('Max value: ', abundance_regridded.euphausia_abundance.max())
mean_abundance=abundance_regridded.euphausia_abundance.mean(dim='days').isel(bootstrap=0)
print('max Mean value: ', mean_abundance.max().values)
# Max abundance = 0.28239554 ind/m2
biomass_regridded = xr.open_dataset(output_file_biomass_regridd)


# %% ======================== Load data ========================
# --- Load mass data for each maturity stage -- Southern Ocean  
# 1. Climatology
clim_krillmass_SO = xr.open_dataset(os.path.join(path_biomass, "fake_worlds/clim_mass_stages_SO.nc")) #shape (181, 231, 1442)

# 2. Actual
actual_krillmass_SO = xr.open_dataset(os.path.join(path_biomass, "fake_worlds/actual_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)

# 3. No MHWs (MHWs replaced by clim)
noMHWs_krillmass_SO = xr.open_dataset(os.path.join(path_biomass, "fake_worlds/noMHWs_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)

# 3. No warming (temperature signal detrended)
nowarming_krillmass_SO = xr.open_dataset(os.path.join(path_biomass, "fake_worlds/warming_mass_stages_SO.nc")) #shape (39, 181, 231, 1442)

# Check
print(f'Is initial mass identical? {(clim_krillmass_SO.immature.isel(days=0).values == actual_krillmass_SO.immature.isel(days=0, years=0).values).all()}')
print(f'Is initial mass identical? {(clim_krillmass_SO.immature.isel(days=0).values == noMHWs_krillmass_SO.immature.isel(days=0, years=0).values).all()}')

# %% ======================== Defining Functions ========================
def density_biomass(mass_ds, proportion, abundance, area, first_day=False, last_day=False):
    # Initialisation
    density_population = xr.zeros_like(mass_ds['juvenile'].isel(days=0))

    # -- Calculate density
    for stage, prop in proportion.items():
        if first_day:
            mass_day = mass_ds[stage].isel(days=0)  #(231, 1442)
        if last_day:
            mass_day = mass_ds[stage].isel(days=-1)  #(231, 1442)
        
        # Formula
        density_stage = prop * abundance * mass_day  # mg/m²

        # Sum on maturity stage
        density_population += density_stage

    # -- Calculate Biomass
    biomass_population = density_population * area  #mg/gridcell

    return density_population, biomass_population


# %% ================================================
#             Initial Density and Biomass
#                      1st Nov
# ===================================================
output_file_ini = os.path.join(path_biomass, "fake_worlds/biomass_density/initial_krill_biomass.nc")
if not os.path.exists(output_file_ini):
    # -- Initialisation
    initial_density = xr.zeros_like(clim_krillmass_SO['juvenile'].isel(days=0))  # shape: (eta_rho, xi_rho)

    # -- Calculate initial density and biomass
    # abundance_day0 = abundance_regridded.euphausia_abundance.isel(bootstrap=0, days=0)
    initial_density, initial_biomass = density_biomass(clim_krillmass_SO, proportion, mean_abundance, area_60S_SO, first_day=True, last_day=False)

    # -- To Dataset
    initial_density.name = "initial_density"
    initial_density.attrs = {"description": "Initial krill biomass density on 1st November (common for all years and conditions).",
                            "units": "mg/m²",}
    initial_density_ds = initial_density.to_dataset()

    initial_biomass.name = "initial_biomass"
    initial_biomass.attrs = {"description": "Initial krill biomass on 1st November (common for all years and conditions).",
                            "units": "mg",}
    initial_biomass_ds = initial_biomass.to_dataset()

    # -- Merge Datasets
    initial_ds = xr.merge([initial_density_ds, initial_biomass_ds])

    # -- Save to file
    initial_ds.to_netcdf(output_file_ini)
    print(f"Saved initial krill biomass dataset.'")
else:
    print(f"File '{output_file_ini}' already exists.")
    initial_ds=xr.open_dataset(output_file_ini)

# %% ================================================
#             Final Density and Biomass
#                    30th April
# ===================================================
# ======= 1. Climatology =======
output_file_clim = os.path.join(path_biomass, "fake_worlds/biomass_density/clim_krill_biomass.nc")
if not os.path.exists(output_file_clim):
    # Calcualte density and biomass
    density_clim, biomass_clim = density_biomass(clim_krillmass_SO, proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True) #shape (231, 1442)

    # -- To Dataset
    density_clim.name = "density"
    density_clim.attrs = {"description": "Krill biomass density on 30th April, under climatological conditions.",
                            "units": "mg/m²",}
    density_clim_ds = density_clim.to_dataset()

    biomass_clim.name = "biomass"
    biomass_clim.attrs = {"description": "Krill biomass on 30th April, under climatological conditions.",
                            "units": "mg",}
    biomass_clim_ds = biomass_clim.to_dataset()

    # -- Merge Datasets
    clim_ds = xr.merge([density_clim_ds, biomass_clim_ds])

    # -- Save to file
    clim_ds.to_netcdf(output_file_clim)
    print(f"Saved initial krill biomass dataset.'")
else:
    # -- Load Dataset
    print(f"File '{output_file_clim}' already exists. Load data")
    clim_ds=xr.open_dataset(output_file_clim)
    
# ======= 2. Actual =======
output_file_actual = os.path.join(path_biomass, "fake_worlds/biomass_density/actual_krill_biomass.nc")
if not os.path.exists(output_file_actual):

    # Initialisation
    years = np.arange(1980, 2019) 
    actual_density_list = []
    actual_biomass_list = []

    for yr_idx, yr in enumerate(years):
        print(f"\nYear: {yr}")
        density_actual_yearly, biomass_actual_yearly = density_biomass(actual_krillmass_SO.isel(years=yr_idx), proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True)

        # Store result to list 
        actual_density_list.append(density_actual_yearly)
        actual_biomass_list.append(biomass_actual_yearly)

    # Combine results
    density_actual = xr.concat(actual_density_list, dim="years")
    biomass_actual = xr.concat(actual_biomass_list, dim="years")

    # -- To Dataset
    density_actual.name = "density"
    density_actual.attrs = {"description": "Krill biomass density on 30th April.",
                            "units": "mg/m²",}
    density_actual_ds = density_actual.to_dataset()

    biomass_actual.name = "biomass"
    biomass_actual.attrs = {"description": "Krill biomass on 30th April.",
                            "units": "mg",}
    biomass_actual_ds = biomass_actual.to_dataset()
    
    # -- Merge Datasets
    actual_ds = xr.merge([density_actual_ds, biomass_actual_ds])

    # -- Save to file
    actual_ds.to_netcdf(output_file_actual)
    print(f"Saved actual krill biomass dataset.'")
else:
    # -- Load Dataset
    print(f"File '{output_file_actual}' already exists. Load data")
    actual_ds=xr.open_dataset(output_file_actual)
    
# ======= 3. No MHWs =======
output_file_noMHWs = os.path.join(path_biomass, "fake_worlds/biomass_density/noMHWs_krill_biomass.nc")
if not os.path.exists(output_file_noMHWs):
    years = np.arange(1980, 2019) 
    noMHWs_density_list = []
    noMHWs_biomass_list = []

    for yr_idx, yr in enumerate(years):
        print(f"\nYear: {yr}")
        density_noMHWs_yearly, biomass_noMHWs_yearly = density_biomass(noMHWs_krillmass_SO.isel(years=yr_idx), proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True)

        # Store result to list 
        noMHWs_density_list.append(density_noMHWs_yearly)
        noMHWs_biomass_list.append(biomass_noMHWs_yearly)

    # Combine results
    density_noMHWs = xr.concat(noMHWs_density_list, dim="years")
    biomass_noMHWs = xr.concat(noMHWs_biomass_list, dim="years")

    # -- To Dataset
    density_noMHWs.name = "density"
    density_noMHWs.attrs = {'description': 'Krill biomass density on 30th April each year. The 100m-avg temperatures under surface MHWs have been replaced by 100m-avg temperature climatology to simulate a world without MHWs influence.',
                            "units": "mg/m²",}
    density_noMHWs_ds = density_noMHWs.to_dataset()

    biomass_noMHWs.name = "biomass"
    biomass_noMHWs.attrs = {'description': 'Krill biomass on 30th April each year. The 100m-avg temperatures under surface MHWs have been replaced by 100m-avg temperature climatology to simulate a world without MHWs influence.',
                            "units": "mg",}
    biomass_noMHWs_ds = biomass_noMHWs.to_dataset()

    # -- Merge Datasets
    noMHWs_ds = xr.merge([density_noMHWs_ds, biomass_noMHWs_ds])

    # -- Save to file
    noMHWs_ds.to_netcdf(output_file_noMHWs)
    print(f"Saved actual krill biomass dataset.'")
else:
    # -- Load Dataset
    print(f"File '{output_file_noMHWs}' already exists. Load data")
    noMHWs_ds=xr.open_dataset(output_file_noMHWs)
    
# ======= 4. No Warming =======
output_file_nowarming = os.path.join(path_biomass, "fake_worlds/biomass_density/no_warming_krill_biomass.nc")
if not os.path.exists(output_file_noMHWs):
    years = np.arange(1980, 2019) 
    nowarming_density_list = []
    nowarming_biomass_list = []

    for yr_idx, yr in enumerate(years):
        print(f"\nYear: {yr}")
        density_nowarming_yearly, biomass_nowarming_yearly = density_biomass(nowarming_krillmass_SO.isel(years=yr_idx), proportion, mean_abundance, area_60S_SO, first_day=False, last_day=True)

        # Store result to list 
        nowarming_density_list.append(density_nowarming_yearly)
        nowarming_biomass_list.append(biomass_nowarming_yearly)

    # Combine results
    density_nowarming = xr.concat(nowarming_density_list, dim="years")
    biomass_nowarming = xr.concat(nowarming_biomass_list, dim="years")

    # -- To Dataset
    density_nowarming.name = "density"
    density_nowarming.attrs = {'description': 'Krill biomass density on 30th April each year. '
                                              'The 100m-avg temperatures are detrended.',
                            "units": "mg/m²",}
    density_nowarming_ds = density_nowarming.to_dataset()

    biomass_nowarming.name = "biomass"
    biomass_nowarming.attrs = {'description': 'Krill biomass on 30th April each year. '
                                              'The 100m-avg temperatures are detrended.',
                            "units": "mg",}
    biomass_nowarming_ds = biomass_nowarming.to_dataset()

    # -- Merge Datasets
    nowarming_ds = xr.merge([density_nowarming_ds, biomass_nowarming_ds])

    # -- Save to file
    nowarming_ds.to_netcdf(output_file_nowarming)
    print(f"Saved 'no warming' krill biomass dataset.'")
else:
    # -- Load Dataset
    print(f"File '{output_file_nowarming}' already exists. Load data")
    nowarming_ds=xr.open_dataset(output_file_nowarming)
    
# %% ================================= Plot density =================================
# --- Prepare data for visualisation
clim_density_diff = clim_ds.density - initial_ds.initial_density
density1989_actual_diff = actual_ds.density.isel(years=9) - initial_ds.initial_density
density2000_actual_diff = actual_ds.density.isel(years=20) - initial_ds.initial_density
density2016_actual_diff = actual_ds.density.isel(years=36) - initial_ds.initial_density

density1989_diff_clim = actual_ds.density.isel(years=9) - clim_ds.density
density2000_diff_clim = actual_ds.density.isel(years=20) - clim_ds.density
density2016_diff_clim = actual_ds.density.isel(years=36) - clim_ds.density

# --- Layout config 
plot = 'slides' #report slides

if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width/1.5
    
else:
    fig_width = 12
    fig_height = 8

fig, axs = plt.subplots(2, 4, figsize=(fig_width * 1.2, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Color setup
from matplotlib.colors import LinearSegmentedColormap
colors = ["#D1105A", "#F24B04", "#FFFFFF", "#884AB2", "#471CA8"]  
# colors = ["#FF9E1F", "#FF930A","#F24B04", "#D1105A", "#AD2D86", "#471CA8", "#371F6F", "#2A1E48"]
cmap_biomass = LinearSegmentedColormap.from_list("biomass", colors, N=256)
# cmap_biomass = 'Purples'
norm_biomass = mcolors.Normalize(vmin=50, vmax=300)  # Tonnes/km² or custom
cmap_diff = 'coolwarm'
norm_diff = mcolors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)
norm_diff_clim = mcolors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)

# --- Data setup 
plot_data = [
    # Row 1: Differences year
    (clim_density_diff, "Climatology Change", cmap_biomass, norm_diff),
    (density1989_actual_diff, "Season Change 1989-1990", cmap_biomass, norm_diff),
    (density2000_actual_diff, "Season Change 2000-2001", cmap_biomass, norm_diff),
    (density2016_actual_diff, "Season Change 2016-2017", cmap_biomass, norm_diff),

    # Row 2: Differences clim
    (None, 'None',None, None), 
    (density1989_diff_clim, "1989-1990 vs Climatology", cmap_diff, norm_diff_clim),
    (density2000_diff_clim, "2000-2001 vs Climatology", cmap_diff, norm_diff_clim),
    (density2016_diff_clim, "2016-2017 vs Climatology", cmap_diff, norm_diff_clim),
]

# --- Plotting 
ims = []
for i, (data, title, cmap, norm) in enumerate(plot_data):
    row = i // 4
    col = i % 4
    ax = axs[row, col]

    if data is None:
        ax.axis('off')
        continue

    im = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                       transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                       shading='auto', rasterized=True)
    ims.append(im)

    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=4)
    ax.coastlines(color='black', linewidth=0.4, zorder=5)
    ax.set_facecolor('#F6F6F3')

    # Gridlines
    import matplotlib.ticker as mticker
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.7, linestyle='--', linewidth=0.4, zorder=7)
    gl.xlocator = mticker.FixedLocator(np.arange(-80, 1, 20))
    gl.xformatter = LongitudeFormatter(degree_symbol='°', number_format='.0f', dateline_direction_label=False)
    gl.yformatter = LatitudeFormatter()
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.ylabels_right = False
    gl.xlabels_left = True
    gl.xlabel_style = {'size': 6, 'rotation': 0}
    gl.ylabel_style = {'size': 6, 'rotation': 0}

    ax.set_title(title, fontsize=10)

# --- Colorbars 
# Biomass colorbar (top row)
cbar_ax1 = fig.add_axes([0.92, 0.56, 0.015, 0.35])
cbar1 = fig.colorbar(ims[0], cax=cbar_ax1, extend='both')
cbar1.set_label("Density [mg/m2]", fontsize=10)
cbar1.ax.tick_params(labelsize=8)

# Changes Biomass colorbar (bottom row)
cbar_ax2 = fig.add_axes([0.92, 0.12, 0.015, 0.35])
cbar2 = fig.colorbar(ims[5], cax=cbar_ax2, extend='both')
cbar2.set_label(r"$\Delta$ Density [mg/m2]", fontsize=10)
cbar2.ax.tick_params(labelsize=8)



# --- Output handling ---
if plot == 'report':
    outdir = os.path.join(os.getcwd(), 'Biomass/figures_outputs/Biomass/')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"biomass_differences_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()







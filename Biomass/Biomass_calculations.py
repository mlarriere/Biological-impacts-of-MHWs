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
stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP accorindg to Tarling et al 2006 - graph under 0°C
proportion = {'juvenile': 0.20, 'immature': 0.3, 'mature': 0.3, 'gravid':0.2}

# %% ======================== Area ========================
# Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc')

# --- 1. Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)

# Mask latitudes south of 60°S (lat_rho <= -60)
area_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)

# --- 2. Atlantic Sector
def subset_atlantic_sector(ds, lat_range=(-80, -60), lon_range=(270, 360)):
    """
    Subset dataset ds to given latitude and longitude ranges.
    lon_range: tuple with (min_lon, max_lon) in degrees [0, 360]
    """
    lat_mask = (ds['lat_rho'] >= lat_range[0]) & (ds['lat_rho'] <= lat_range[1])
    lon_mask = (ds['lon_rho'] >= lon_range[0]) & (ds['lon_rho'] <= lon_range[1])

    combined_mask = lat_mask & lon_mask
    return ds.where(combined_mask, drop=True)


# Apply subset
area_Atl_Sect = subset_atlantic_sector(area_roms)

# Select surface layer
area_Atl_surf = area_Atl_Sect['area'].isel(z_t=0)
area_Atl_surf_m2 = area_Atl_Sect['area'].isel(z_t=0) * 1e6

# Sum area
total_area_Atl_km2 = area_Atl_surf.sum().item()
print(f'Total Atlantic Sector area south of 60°S: {total_area_Atl_km2:.2f} km²') #5707377.50 km²

# %%
N = 17.85 # Abundance ind/m2



# %% ======================== Load data ========================
# Load mass data for each maturity stage
krill_mass = xr.open_dataset(os.path.join(path_mass, 'mass_traj_mat_stages.nc'))
krill_mass_clim = xr.open_dataset(os.path.join(path_mass, 'clim_mass_traj_mat_stages.nc'))


# %% ======================== Initial Density ========================
initial_density = xr.zeros_like(krill_mass_clim['juvenile'].isel(days=0))  # shape: (eta_rho, xi_rho)

# Calculate initial biomass - common for all years (initial length equal)
for stage, prop in proportion.items():
    print(f"Adding initial biomass for stage: {stage}")
    stage_mass_day0 = krill_mass_clim[stage].isel(days=0)  # (eta_rho, xi_rho)
    biomass_stage = prop * N * stage_mass_day0  # mg/m²
    initial_density += biomass_stage

# # Multiply by area to get biomass per grid cell
# initial_biomass_mg = initial_density * area_Atl_surf_m2  # mg/gridcell
# initial_biomass_t = initial_biomass_mg# * 1e-9# Convert from mg to t

# %% ======================== Final Density climatology ========================
final_density_clim = xr.zeros_like(krill_mass_clim['juvenile'].isel(days=0))  # shape: (eta_rho, xi_rho)

# Calculate initial biomass - common for all years (initial length equal)
for stage, prop in proportion.items():
    print(f"Adding initial biomass for stage: {stage}")
    stage_mass_day0 = krill_mass_clim[stage].isel(days=-1)  # (eta_rho, xi_rho)
    biomass_stage = prop * N * stage_mass_day0  # mg/m²
    final_density_clim += biomass_stage

# # Multiply by area to get biomass per grid cell
# final_biomass_clim_mg = final_density_clim * area_Atl_surf_m2  # mg/gridcell
# final_biomass_clim_t = final_biomass_clim_mg# * 1e-9 # Convert from mg to t

# %% ======================== Final Density Yearly ========================
selected_years = [1989, 2000, 2016]
selected_years_idx = np.array(selected_years) - 1980  # [9, 20, 36]

final_density_datasets = {}

for i, yr_idx in enumerate(selected_years_idx):
    yr_label = selected_years[i]
    print(f"\nYear: {yr_label}")
    
    # Initialize
    final_density = xr.zeros_like(krill_mass['juvenile'].isel(years=yr_idx, days=-1))

    # Calcualte biomass on the last day
    for stage, prop in proportion.items():
        print(f"  Adding final biomass for stage: {stage}")
        
        # Select final day of this year for this stage
        stage_mass_day_final = krill_mass[stage].isel(years=yr_idx, days=-1)  # (eta_rho, xi_rho)
        biomass_stage = prop * N * stage_mass_day_final  # mg/m²
        final_density += biomass_stage

    # final_biomass_gridcell = final_density * area_Atl_surf_m2 #mg
    # final_biomass_t = final_biomass_gridcell# * 1e-9

    # To Dataset
    final_density.attrs = {
        'description': f'Final krill biomass',
        'year': int(yr_label),
        'units': 'tonnes',
        'abundance': '17.85 ind/m²'
    }

    # Store
    final_density.name = 'biomass'
    final_density_datasets[yr_label] = final_density.to_dataset()


# %% ================================= Differences (30thApril - 1stNov )
clim_density_diff = final_density_clim - initial_density
density1989_diff = final_density_datasets[1989].biomass - initial_density
density2000_diff = final_density_datasets[2000].biomass - initial_density
density2016_diff = final_density_datasets[2016].biomass - initial_density

# %% ================================= Differences (30thApril season - Clim)
density1989_diff_clim = final_density_datasets[1989].biomass - final_density_clim
density2000_diff_clim = final_density_datasets[2000].biomass - final_density_clim
density2016_diff_clim = final_density_datasets[2016].biomass - final_density_clim

# %% ================================= Plot density =================================
# === Layout config ===
plot = 'report' #report slides

if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width/1.5
    
else:
    fig_width = 12
    fig_height = 8

fig, axs = plt.subplots(2, 4, figsize=(fig_width * 1.2, fig_height), subplot_kw={'projection': ccrs.SouthPolarStereo()})
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# === Path boundary ===
theta = np.linspace(np.pi / 2, np.pi, 100)
center, radius = [0.5, 0.51], 0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

# === Color setup ===
from matplotlib.colors import LinearSegmentedColormap
colors = ["#D1105A", "#F24B04", "#FFFFFF", "#884AB2", "#471CA8"]  
# colors = ["#FF9E1F", "#FF930A","#F24B04", "#D1105A", "#AD2D86", "#471CA8", "#371F6F", "#2A1E48"]
cmap_biomass = LinearSegmentedColormap.from_list("biomass", colors, N=256)
# cmap_biomass = 'Purples'
norm_biomass = mcolors.Normalize(vmin=50, vmax=300)  # Tonnes/km² or custom
cmap_diff = 'coolwarm'
norm_diff = mcolors.TwoSlopeNorm(vmin=-300, vcenter=0, vmax=300)
norm_diff_clim = mcolors.TwoSlopeNorm(vmin=-150, vcenter=0, vmax=150)

# === Data setup ===
plot_data = [
    # Row 1: Differences year
    (clim_density_diff, "Climatology Change", cmap_biomass, norm_diff),
    (density1989_diff, "Season Change 1989-1990", cmap_biomass, norm_diff),
    (density2000_diff, "Season Change 2000-2001", cmap_biomass, norm_diff),
    (density2016_diff, "Season Change 2016-2017", cmap_biomass, norm_diff),

    # Row 2: Differences clim
    (None, 'None',None, None), 
    (density1989_diff_clim, "1989-1990 vs Climatology", cmap_diff, norm_diff_clim),
    (density2000_diff_clim, "2000-2001 vs Climatology", cmap_diff, norm_diff_clim),
    (density2016_diff_clim, "2016-2017 vs Climatology", cmap_diff, norm_diff_clim),
]

# === Plotting ===
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

# === Colorbars ===
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
    plt.savefig(os.path.join(outdir, outfile), dpi=300, format='pdf', bbox_inches='tight')
    # plt.show()
else:
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()




# %% ================================= Biomass =================================
# Multiply by area to get biomass per grid cell and then sum

# ----- Initial Biomass 
initial_biomass_mg = initial_density * area_Atl_surf_m2  # mg
initial_biomass_Mt = initial_biomass_mg.sum(dim=['eta_rho', 'xi_rho'])* 1e-15

# ----- Final Density climatology
final_biomass_clim_mg = final_density_clim * area_Atl_surf_m2  # mg
final_biomass_clim_Mt = final_biomass_clim_mg.sum(dim=['eta_rho', 'xi_rho'])* 1e-15

# ----- Final Density Yearly 
selected_years = [1989, 2000, 2016]
selected_years_idx = np.array(selected_years) - 1980  # [9, 20, 36]

final_biomass_datasets = {}

for i, yr_idx in enumerate(selected_years_idx):
    yr_label = selected_years[i]
    print(f"\nYear: {yr_label}")
    
    # Initialize
    final_density = xr.zeros_like(krill_mass['juvenile'].isel(years=yr_idx, days=-1))

    # Calcualte biomass on the last day
    for stage, prop in proportion.items():
        print(f"  Adding final biomass for stage: {stage}")
        
        # Select final day of this year for this stage
        stage_mass_day_final = krill_mass[stage].isel(years=yr_idx, days=-1)  # (eta_rho, xi_rho)
        biomass_stage = prop * N * stage_mass_day_final  # mg/m²
        final_density += biomass_stage

    final_biomass_gridcell = final_density * area_Atl_surf_m2 #mg

    # To Dataset
    final_density.attrs = {
        'description': f'Final krill biomass',
        'year': int(yr_label),
        'units': 'tonnes',
        'abundance': '17.85 ind/m²'
    }

    # Store
    final_density.name = 'biomass'
    final_biomass_datasets[yr_label] = final_density.to_dataset()

final_biomass1989 = final_biomass_datasets[1989].biomass * area_Atl_surf_m2  # mg
final_biomass1989_Mt = final_biomass1989.sum(dim=['eta_rho', 'xi_rho'])* 1e-15

final_biomass2000 = final_biomass_datasets[2000].biomass * area_Atl_surf_m2  # mg
final_biomass2000_Mt = final_biomass2000.sum(dim=['eta_rho', 'xi_rho'])* 1e-15

final_biomass2016 = final_biomass_datasets[2016].biomass * area_Atl_surf_m2  # mg
final_biomass2016_Mt = final_biomass1989.sum(dim=['eta_rho', 'xi_rho'])* 1e-15

# %% ================================= Differences (30thApril - 1stNov )
clim_biomass_diff = final_biomass_clim_Mt - initial_biomass_Mt
biomass1989_diff = final_biomass1989_Mt - initial_biomass_Mt
biomass2000_diff = final_biomass2000_Mt - initial_biomass_Mt
biomass2016_diff = final_biomass2016_Mt - initial_biomass_Mt

# %% ================================= Differences (30thApril season - Clim)
biomass1989_diff_clim = final_biomass1989_Mt - final_biomass_clim_Mt
biomass2000_diff_clim = final_biomass2000_Mt - final_biomass_clim_Mt
biomass2016_diff_clim = final_biomass2016_Mt - final_biomass_clim_Mt

# %% ================================= Print Results

print(f"Initial biomass total: {initial_biomass_Mt.item():.2f} Mt")
print(f"Final biomass climatology total: {final_biomass_clim_Mt.item():.2f} Mt\n")

for year in selected_years:
    final_biomass_Mt = final_biomass_datasets[year].biomass * area_Atl_surf_m2
    final_biomass_Mt_sum = final_biomass_Mt.sum(dim=['eta_rho', 'xi_rho']) * 1e-15
    print(f"Final biomass {year} total: {final_biomass_Mt_sum.item():.2f} Mt")

print("\nDifferences (Final - Initial):")
print(f"Climatology: {clim_biomass_diff.item():.2f} Mt")
print(f"1989: {biomass1989_diff.item():.2f} Mt")
print(f"2000: {biomass2000_diff.item():.2f} Mt")
print(f"2016: {biomass2016_diff.item():.2f} Mt")

print("\nDifferences (Year - Climatology):")
print(f"1989: {biomass1989_diff_clim.item():.2f} Mt")
print(f"2000: {biomass2000_diff_clim.item():.2f} Mt")
print(f"2016: {biomass2016_diff_clim.item():.2f} Mt")

# %%
# # Extract final mass for each maturity stage, MHW conditions and year
# mass_final = {stage: [] for stage in stage_lengths}

# for stage, ds in zip(['juvenile', 'immature', 'mature', 'gravid']):
#     # Select the last day for each year, assuming 'days' is sorted
#     final_mass = ds.isel(days=-1)
    
#     # Keep only the mass variables (non_mhw and mhw_*)
#     mass_vars = [v for v in ds.data_vars]
#     final_mass_vars = final_mass[mass_vars]
    
#     mass_final[stage] = final_mass_vars #shape: (39, )



# # %% ======================== Biomass Calculation ========================
# # Calculate total biomass (mg) for each year
# biomass_total = {}
# for var in ['mass_cat0', 'mass_cat1', 'mass_cat2', 'mass_cat3', 'mass_cat4']:
#     B_j = area_atl_sect * sum(proportion[stage] * N * mass_final[stage][var] for stage in proportion)
#     B_j_Mt = B_j * 1e-15  # Convert from mg to Mt
#     biomass_total[var] = B_j_Mt


# # %% ================ Relative Chnages ================
# delta_biomass_rel = {}
# for i in range(1, 5):  # Categories 1 to 4
#     delta_percent = ((biomass_total[f'mass_cat{i}'] - biomass_total['mass_cat0']) / biomass_total['mass_cat0']) * 100
#     delta_biomass_rel[f'delta_cat{i}'] = delta_percent

# # %% ======================== Area fraction of the MHWs categories ========================
# # --- Total area
# def subset_spatial_domain(ds, lat_range=(-80, -60), lon_range=(270, 360)): #, (0, 30)
#     lat_min, lat_max = lat_range
#     lon_range1, lon_range2 = lon_range

#     lat_mask = (ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max)
#     lon_mask = ((ds['lon_rho'] >= lon_range1) & (ds['lon_rho'] <= lon_range2)) #| ((ds['lon_rho'] >= lon_range2[0]) & (ds['lon_rho'] <= lon_range2[1]))

#     return ds.where(lat_mask & lon_mask, drop=True)

# # Load data
# ROMS_area = xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc') # do not contain land cells
# area_Atl_Sect = subset_spatial_domain(ROMS_area)
# area_Atl_surf = area_Atl_Sect['area'].isel(z_t=0) # Select surface layer

# # --- Area under MHWs
# temp_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/temp_avg100m_daily_mhw.nc")
# temp_mhw = xr.open_dataset(temp_mhw_file)

# mhw_area_km2 = {}
# for deg in range(1, 5):
#     varname = f'temp_{deg}deg'
    
#     # MHW presence mask: True where MHW temp data exists (ocean+land)
#     mhw_mask = np.isfinite(temp_mhw[varname])  # (years, days, eta_rho, xi_rho)
    
#     # Cells that have experienced MHWs
#     mhw_any = mhw_mask.any(dim='days')  # (years, eta_rho, xi_rho)
    
#     # Multiply boolean mask by area of each cell 
#     area_cells = mhw_any * area_Atl_surf # (years, eta_rho, xi_rho)

#     # Total area affected per year
#     area_sum = area_cells.sum(dim=['eta_rho', 'xi_rho'])
    
#     # Count affected ocean cells
#     mhw_area_km2[f'cat{deg}'] = area_sum

# # Total ocean area
# total_ocean_area = area_Atl_surf.sum().item()

# # Sum all MHW areas per year
# total_mhw_area = sum(mhw_area_km2[f'cat{deg}'] for deg in range(1, 5))

# # Non-MHW ocean area per year
# mhw_area_km2['cat0'] = total_ocean_area - total_mhw_area

# # --- Calculating area fraction
# mhw_area_frac = {k: v / total_ocean_area for k, v in mhw_area_km2.items()}
# print(f"Fraction of ocean under 1°C MHW in 2000: {mhw_area_frac['cat4'][36].values*100:.3f}%")



# # %% ======================== PLOT ========================
# years_to_plot = [1989, 2000, 2016]
# year_indices = [year - 1980 for year in years_to_plot]

# fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# for ax, year, idx in zip(axs, years_to_plot, year_indices):
#     biomass_year = {var: biomass_total[var].isel(years=idx).item() for var in biomass_total}
#     bars = ax.bar(biomass_year.keys(), biomass_year.values(), color='teal')
#     ax.set_title(f"Biomass in {year}")
#     ax.set_xlabel("MHW Scenario")
#     ax.set_xticklabels(biomass_year.keys(), rotation=45)
#     if ax == axs[0]:
#         ax.set_ylabel("Biomass (Mt)")

#     # Add text labels on top of each bar
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height,
#                 f"{height:.3e}", ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.show()

# # %% 

# categories = ['Non MHWs', 'MHWs 1°C', 'MHWs 2°C', 'MHWs 3°C', 'MHWs 4°C']
# # x_pos = np.arange(len(categories))
# x_pos = np.array([1, 2, 3, 4, 5])  # wider spacing between bars

# years_to_plot = [1989, 2000, 2016]
# year_indices = [year - 1980 for year in years_to_plot]


# # === Layout config ===
# plot = 'report' #report slides

# if plot == 'report':
#     fig_width = 6.3228348611
#     fig_height = fig_width
    
# else:
#     fig_width = 12
#     fig_height = 8
  
# fig, axs = plt.subplots(len(years_to_plot), 3, figsize=(fig_width, fig_height), sharex=True)
# fig.subplots_adjust(hspace=0.4, wspace = 0.45)  

# title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
# label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
# tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
# suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {'fontsize': 10, 'fontweight': 'bold'}

# for row_idx, year in enumerate(years_to_plot):
#     bar_width = 0.6 if plot == 'report' else 0.8

#     idx = year - 1980  # year index

#     # ------- Panel 1: Absolute Biomass
#     biomass_vals = [biomass_total[f'mass_cat{i}'].isel(years=idx).item() for i in range(5)]
#     bars1 = axs[row_idx, 0].bar(x_pos, biomass_vals, color='#168AAD', width=bar_width)

#     # Axis and title
#     axs[row_idx, 0].set_ylabel("Biomass [Mt]", **label_kwargs)
#     axs[row_idx, 0].set_title("Biomass" if row_idx == 0 else "", **suptitle_kwargs)
#     axs[row_idx, 0].set_ylim(0,10)
#     axs[row_idx, 0].set_xlim(0.4, 5.6)

#     # Add text with the values
#     for bar in bars1:
#         h = bar.get_height()
#         axs[row_idx, 0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}", ha='center', color='#1A759F', va='bottom', **tick_kwargs)

#     # ------- Panel 2: Relative Biomass Change (% vs cat0)
#     delta_vals = [0.0] + [delta_biomass_rel[f'delta_cat{i}'].isel(years=idx).item() for i in range(1, 5)]
#     bars2 = axs[row_idx, 1].bar(x_pos, delta_vals, color='#FFA200', width=bar_width)

#     # Axis and title
#     axs[row_idx, 1].set_ylabel(r"$\Delta$ [\%]",  **label_kwargs)
#     axs[row_idx, 1].set_title("Relative\nBiomass Change" if row_idx == 0 else "", **suptitle_kwargs)
#     axs[row_idx, 1].axhline(0, color='gray', linestyle='--')
    
#     # Ylim with some padding
#     heights = [bar.get_height() for bar in bars2]
#     ymin = min(heights) - 3  # minimum
#     ymax = max(heights) + 5  # maximum
#     axs[row_idx, 1].set_ylim(ymin, ymax)
#     axs[row_idx, 1].set_xlim(0.4, 5.6)

#     # Add text with the values
#     for bar in bars2:
#         h = bar.get_height()
#         va = 'bottom' if h >= 0 else 'top'
#         axs[row_idx, 1].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}%", ha='center', color = '#FF9500' ,va=va, **tick_kwargs)

#     # ------- Panel 3: MHW Area Fraction (%)
#     area_frac_vals = [mhw_area_frac[f'cat{i}'][idx].values * 100 for i in range(5)]
#     bars3 = axs[row_idx, 2].bar(x_pos, area_frac_vals, color='#55A630', width=bar_width)

#     # Axis and title
#     axs[row_idx, 2].set_ylabel(r"[\%]", **label_kwargs)
#     axs[row_idx, 2].set_title("Area Fraction" if row_idx == 0 else "", **suptitle_kwargs)

#     # Ylim with some padding
#     heights = [bar.get_height() for bar in bars3]
#     ymin = 0 # minimum
#     ymax = max(heights) + 10  # maximum
#     axs[row_idx, 2].set_ylim(ymin, ymax)
#     axs[row_idx, 2].set_xlim(0.4, 5.6)

#     # Add text with the values
#     for bar in bars3:
#         h = bar.get_height()
#         axs[row_idx, 2].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}%", ha='center', color='#2B9348', va='bottom', **tick_kwargs)
    
#     # Set x-ticks & labels only on bottom row for clarity
#     if row_idx == len(years_to_plot) - 1:
#         for ax in axs[row_idx, :]:
#             ax.set_xticks(x_pos)
#             # ax.set_xticklabels(categories)
#             ax.set_xticklabels(categories, rotation=45, ha='right', **tick_kwargs)

#     else:
#         for ax in axs[row_idx, :]:
#             ax.set_xticks(x_pos)
#             ax.set_xticklabels([])

#     # Add year label on the left of each row (y-axis label style)
#     axs[row_idx, 0].annotate(f'{year} - {year+1}' , xy=(-0.5, 0.5), xycoords='axes fraction',
#                              ha='right', va='center', rotation=90, **suptitle_kwargs)

# # Add a big title if plotting for slides
# if plot == 'slides':
#     fig.suptitle("Krill Biomass under MHWs", **suptitle_kwargs, y=1.02)

# # --- Output handling ---
# if plot == 'report':
#     outdir = os.path.join(os.getcwd(), 'Biomass/figures_outputs/Biomass/')
#     os.makedirs(outdir, exist_ok=True)
#     outfile = f"biomass_3years_{plot}.pdf"
#     # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
#     plt.show()
# else:    
#     # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_AtlanticSector/atlantic_sector{selected_years[yr_chosen]}_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
#     plt.show()


# # %%

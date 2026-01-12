"""
Created on Tue 22 July 10:22:14 2025

Length distribution in 1 population of krill

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
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
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

# %% ======================== Selected years ========================
selected_years = [1989, 2000, 2016]
selected_years_idx = np.array(selected_years) - 1980  # [9, 20, 36]


# %% ======================== Load drivers ========================
# Year of interest
yr_chosen=2
year_index = selected_years_idx[yr_chosen] 

def subset_spatial_domain(ds, lat_range=(-80, -60), lon_range=(270, 360)): #, (0, 30)
    lat_min, lat_max = lat_range
    lon_range1, lon_range2 = lon_range

    lat_mask = (ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max)
    lon_mask = ((ds['lon_rho'] >= lon_range1) & (ds['lon_rho'] <= lon_range2)) #| ((ds['lon_rho'] >= lon_range2[0]) & (ds['lon_rho'] <= lon_range2[1]))

    return ds.where(lat_mask & lon_mask, drop=True)

# ==== Temperature [°C] 
# Weighted averaged temperature of the first 100m - Austral summer - 60S - years = seasonal (i.e. ranging from 1980 to 2018 with days 304-119)
temp_avg_100m_SO_allyrs = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc')) #shape (39, 181, 231, 1442)
temp_avg_100m_study_area_allyrs = subset_spatial_domain(temp_avg_100m_SO_allyrs) #select spatial extent -- shape (39, 181, 231, 360)
temp_avg_100m_study_area_1season = temp_avg_100m_study_area_allyrs.isel(years=year_index) #select temporal extent for 1 year of interest -- shape (181, 231, 360)

# ==== Chla [mh Chla/m3] 
# Weighted averaged chla of the first 100m - Austral summer - 60S - years = seasonal (i.e. ranging from 1980 to 2018 with days 304-119)
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 
chla_surf_study_area_allyrs = subset_spatial_domain(chla_surf_SO_allyrs) #select spatial extent
chla_surf_study_area_1season = chla_surf_study_area_allyrs.isel(years=year_index) #select temporal extent for 1 year of interest -- shape (181, 231, 360)

#%% ============== Calculating length for each maturity stage ==============
# Atlantic Sector for 1 season of interest
from Growth_Model.Atkinson2006_model import length_Atkison2006 

# -- Parameters
stage_lengths = {'juvenile': 25, 'immature': 30, 'mature': 40, 'gravid': 45}
# stage_IMP = {'juvenile': 12, 'immature': 24, 'mature': 13, 'gravid': 13} # IMP accorindg to Tarling et al 2006 - graph under 0°C

# ==== Calculate length on a daily basis
# -- Juvenile
length_juv_daily = length_Atkison2006(chla=chla_surf_study_area_1season.chla, temp=temp_avg_100m_study_area_1season.avg_temp,
                                         initial_length=stage_lengths['juvenile'], maturity_stage='juvenile')

# -- Immature
length_imm_daily = length_Atkison2006(chla=chla_surf_study_area_1season.chla, temp=temp_avg_100m_study_area_1season.avg_temp,
                                         initial_length=stage_lengths['immature'], maturity_stage='immature')

# -- Mature
length_mat_daily = length_Atkison2006(chla=chla_surf_study_area_1season.chla, temp=temp_avg_100m_study_area_1season.avg_temp,
                                         initial_length=stage_lengths['mature'], maturity_stage='mature')

# -- Gravid
length_gra_daily = length_Atkison2006(chla=chla_surf_study_area_1season.chla, temp=temp_avg_100m_study_area_1season.avg_temp,
                                         initial_length=stage_lengths['gravid'], maturity_stage='gravid')


# ==== One step back - extracting growth rate (mm/d) between each days
daily_growth_juv = length_juv_daily.diff(dim='days') #shape (180, 231, 360)
daily_growth_imm = length_imm_daily.diff(dim='days')
daily_growth_mat = length_mat_daily.diff(dim='days')
daily_growth_gra = length_gra_daily.diff(dim='days')

# ==== Create masks for the different MHW scenarios (Use Temp Mask - Boolean: where MHW occurred)
temp_mhw_file = os.path.join(path_growth_inputs, f"atlantic_sector/temp_avg100m_daily_mhw.nc")
temp_mhw = xr.open_dataset(temp_mhw_file)
mhw_1 = xr.where(~np.isnan(temp_mhw.temp_1deg.isel(years=year_index)), 1, 0)
mhw_2 = xr.where(~np.isnan(temp_mhw.temp_2deg.isel(years=year_index)), 1, 0)
mhw_3 = xr.where(~np.isnan(temp_mhw.temp_3deg.isel(years=year_index)), 1, 0)
mhw_4 = xr.where(~np.isnan(temp_mhw.temp_4deg.isel(years=year_index)), 1, 0)

# Non non-exclusive masks, i.e. a 4°C MHW also counted as a 1°C
growth_by_stage_and_mhw = {}
stage_daily_growths = {
    'juvenile': daily_growth_juv,
    'immature': daily_growth_imm,
    'mature': daily_growth_mat,
    'gravid': daily_growth_gra
}
for stage, daily_growth in stage_daily_growths.items():
    growth_by_mhw = {
        1: daily_growth.where(mhw_1 == 1),
        2: daily_growth.where(mhw_2 == 1),
        3: daily_growth.where(mhw_3 == 1),
        4: daily_growth.where(mhw_4 == 1),
        0: daily_growth.where((mhw_1 + mhw_2 + mhw_3 + mhw_4) == 0)
    }
    growth_by_stage_and_mhw[stage] = growth_by_mhw


# ==== Length trajectories under the different scenarios
n_days = daily_growth.sizes["days"]+1
length_by_stage_and_mhw = {}

for stage in stage_lengths:
    initial_length = stage_lengths[stage]
    intermoult_period = stage_IMP[stage]
    length_by_mhw_level = {}

    for level in range(5):
        growth = growth_by_stage_and_mhw[stage][level]  # (days, eta_rho, xi_rho)

        # 1. Mean growth across space for each day
        daily_mean_growth = growth.mean(dim=["eta_rho", "xi_rho"], skipna=True)

        # 2. Intermoult-block logic with last valid step growth
        growth_blocks = []
        for i in range(0, n_days, intermoult_period):
            block = daily_mean_growth.isel(days=slice(i, min(i + intermoult_period, n_days)))
            
            # Take last valid growth value in the block, or 0 if none
            valid_growth = block.dropna(dim='days')
            if valid_growth.size > 0:
                block_growth = valid_growth[-1].item()
            else:
                block_growth = 0.0
            growth_blocks.extend([block_growth] * len(block))

        # 3. Length trajectory
        length_series = [initial_length]
        current_length = initial_length
        for i in range(1, n_days):
            if i % intermoult_period == 0:
                current_length += growth_blocks[i - 1]
            length_series.append(current_length)

        # 4. Store as DataArray
        length_by_mhw_level[level] = xr.DataArray(
            data=length_series,
            dims=["days"],
            coords={"days": length_gra_daily["days"]}
        ) #shape: (181,)

    # Save all MHW levels for this stage
    length_by_stage_and_mhw[stage] = length_by_mhw_level


# %% ============== Converting length to mass ==============
def length_to_mass(p, length_array, r):
    mass_array = p*length_array**r
    return mass_array

# --- Defining constants
# Accroding to mass length coefficient of Atkison et al (2006)
p = 10**(-4.19)
r = 3.89
print(f'Coefficients: p={p:.5f}, r={r:.3f}')

# ==== Mean mass trajectories from avg length trajectories
mass_trajectories_by_stage_and_mhw = {}
for stage, length_trajectory_dict in length_by_stage_and_mhw.items():    
    mass_trajectories_by_mhw = {}

    for level, length_da in length_trajectory_dict.items():
        # Convert length trajectory to mass trajectory directly
        mass_da = length_to_mass(p, length_da, r)

        mass_trajectories_by_mhw[level] = mass_da

    mass_trajectories_by_stage_and_mhw[stage] = mass_trajectories_by_mhw

# cannot apply the same method as before as the IMP is applied to length and not to mass and the relationship between the 2 is not linear

# %% ============== Plot length and mass ==============
# --- Setup ---
stages = ['juvenile', 'immature', 'mature', 'gravid']
stage_labels = ['Juvenile', 'Immature', 'Mature', 'Gravid']
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']
non_mhw_color = 'black'

# --- Time axis (Nov 1 start) ---
from datetime import datetime, timedelta
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)
days_xaxis = np.arange(181)

# --- Plotting mode ---
plot = 'report'  # 'report' or 'slides'
if plot == 'report':
    fig_height = 9.3656988889
    fig_width = fig_height/2 #6.3228348611  # inches

else:
    fig_width = 15
    fig_height = 6.5

# --- Font & layout settings ---
# --- Font settings ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 10}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}

fig, axs = plt.subplots(4, 2, figsize=(fig_width, fig_height), 
                        gridspec_kw={'wspace': 0.6, 'hspace': 0.25}, sharex=True)
axs = axs.reshape(4, 2)

scenarios = [0, 1, 2, 3, 4]
scenario_labels = ['Non-MHWs', '1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']
scenario_colors = [non_mhw_color] + threshold_colors

last_day = days_xaxis[-1]
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []

for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)


for row, stage in enumerate(stages):
    ax_len = axs[row, 0]
    ax_mass = axs[row, 1]

    for scen, label, color in zip(scenarios, scenario_labels, scenario_colors):
        # Get mean length and mass time series for each scenario
        y_len = length_by_stage_and_mhw[stage][scen]
        y_mass = mass_trajectories_by_stage_and_mhw[stage][scen]
       
        # Convert to numpy arrays if necessary
        if hasattr(y_len, 'values'):
            y_len = y_len.values
        if hasattr(y_mass, 'values'):
            y_mass = y_mass.values

        # Plot
        lw = 1 if plot == 'report' else 2
        ax_len.plot(days_xaxis, y_len, color=color, label=label, linewidth=lw)
        ax_mass.plot(days_xaxis, y_mass, color=color, label=label, linewidth=lw)

        # Text annotation for final LENGTH
        text_fotsize = 9 if plot == 'report' else 12
        ax_len.text(last_day + 7, y_len[-1], f"{y_len[-1]:.1f}",
                    color=color, fontsize=text_fotsize, verticalalignment='center', horizontalalignment='left')
        # Text annotation for final MASS
        ax_mass.text(last_day + 7, y_mass[-1], f"{y_mass[-1]:.1f}",
                    color=color, fontsize=text_fotsize, verticalalignment='center', horizontalalignment='left')

        
    # --- Labels
    ax_len.set_ylabel("Length [mm]", **label_kwargs)
    # ax_len.grid(True, linestyle=':', alpha=0.4)

    ax_mass.set_ylabel("Mass [mg]", **label_kwargs)
    # ax_mass.grid(True, linestyle=':', alpha=0.4)
    if row == 3:
        # X-axis ticks and labels for bottom row only
        ax_len.set_xlabel("Date", **label_kwargs)
        ax_len.set_xticks(tick_positions)
        ax_len.set_xticklabels(tick_labels, rotation=45)
        ax_len.tick_params(axis='x', labelsize=tick_kwargs.get('labelsize', None))

        ax_mass.set_xlabel("Date", **label_kwargs)
        ax_mass.set_xticks(tick_positions)
        ax_mass.set_xticklabels(tick_labels, rotation=45)
        ax_mass.tick_params(axis='x', labelsize=tick_kwargs.get('labelsize', None))
    else:
        ax_len.set_xticks([])
        ax_mass.set_xticks([])

    # Always apply y-tick label size (optional, if needed)
    ax_len.tick_params(axis='y', labelsize=tick_kwargs.get('labelsize', None))
    ax_mass.tick_params(axis='y', labelsize=tick_kwargs.get('labelsize', None))

    # --- Title
    # Add centered title above both subplots in this row:
    x_mid = (ax_len.get_position().x0 + ax_mass.get_position().x1) / 2
    y_top = ax_len.get_position().y1 + 0.01
    fig.text(x_mid, y_top, f' --------------------- {stage_labels[row]} --------------------- ', ha='center', va='bottom', fontsize=subtitle_kwargs.get('fontsize', 10))

    
# Increase xlim to see the texts
for row in range(4):
    axs[row, 0].set_xlim(-2, last_day+5)
    axs[row, 1].set_xlim(-2, last_day+5)

# --- Legend 
if plot == 'report':
    legend_box = (0.5, 0.99)
    ncol=2
else:
    legend_box = (0.5, 0.93)
    ncol=5
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=legend_box, ncol=ncol, frameon=True, **legend_kwargs)

# --- Main Title
y = 1.03 if plot == 'report' else 0.96
fig.suptitle(f"Krill Length and Mass Trajectories by Maturity Stage\nGrowth Season {selected_years[yr_chosen]}", y=y, **maintitle_kwargs)

# --- Output handling ---
outdir = os.path.join(os.getcwd(), 'Biomass/figures_outputs/Population')
os.makedirs(outdir, exist_ok=True)

outfile = f"mass_length_{plot}.{'png' if plot == 'slides' else 'pdf'}"
savepath = os.path.join(outdir, outfile)

# --- Save or show ---
if plot == 'report':
    # plt.savefig(savepath, dpi=200, format='pdf', bbox_inches='tight')
    plt.show()  
else:
    # plt.savefig(savepath, dpi=500, format='png', bbox_inches='tight')
    plt.show()




# %%

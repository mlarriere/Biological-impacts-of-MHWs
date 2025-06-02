"""
Created on Friday 30 Mai 14:18:37 2025

code to show global warming in the southern ocean - report (intro) 

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,   
    "text.latex.preamble": r"\usepackage{mathptmx} \usepackage[x11names, dvipsnames, table]{xcolor}",
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

# Handling time
from datetime import datetime, timedelta
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0, 121)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)


# %%
def extract_temperature(ieta):
    start_time = time.time()
    # ieta =200
    fn = f"{path_temp}{file_var}eta{ieta}.nc"
    ds = xr.open_dataset(fn)[var][1:, 0:365, :, :]  # Shape: (30, 365, 35, 1442)
    # Select periods
    ds_clim1 = ds.isel(year=slice(0,10)).mean(dim=('year','day'), skipna=True)
    ds_clim2 = ds.isel(year=slice(10,20)).mean(dim=('year','day'), skipna=True)
    ds_clim3 = ds.isel(year=slice(20,30)).mean(dim=('year','day'), skipna=True)
    ds_warm = ds.isel(year=slice(30,40)).mean(dim=('year','day'), skipna=True)

    print(f"Finished eta {ieta} in {time.time() - start_time:.2f}s")

    return ieta, ds_clim1, ds_clim2, ds_clim3, ds_warm #ds_surf, ds_mid

# Calling function to combine eta -- with process_map
det_clim1 = np.empty((neta, nz, nxi), dtype=np.float32)
det_clim2 = np.empty((neta, nz, nxi), dtype=np.float32)
det_clim3 = np.empty((neta, nz, nxi), dtype=np.float32)
det_warm = np.empty((neta, nz, nxi), dtype=np.float32)

for ieta, ds_clim1, ds_clim2, ds_clim3, ds_warm in process_map(extract_temperature, range(0, neta), max_workers=30, desc="Processing eta"):
        det_clim1[ieta] = ds_clim1
        det_clim2[ieta] = ds_clim2
        det_clim3[ieta] = ds_clim3
        det_warm[ieta] = ds_warm
       
# Flip eta position
det_clim1_transposed = det_clim1.transpose(1, 0, 2) #shape (35, 434, 1442)
det_clim2_transposed = det_clim2.transpose(1, 0, 2) #shape (35, 434, 1442)
det_clim3_transposed = det_clim3.transpose(1, 0, 2) #shape (35, 434, 1442)
det_warm_transposed = det_warm.transpose(1, 0, 2)


#%% To dataset
ds_clim1 = xr.Dataset(
    data_vars=dict(temp = (["depth", "eta_rho", "xi_rho"], det_clim1_transposed)),
    coords=dict(
        years= range(1980, 1980+nyears),
        days= range(0, ndays),
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Mean temperature - period (1980-1989)'),
        ) 

ds_clim2 = xr.Dataset(
    data_vars=dict(temp = (["depth", "eta_rho", "xi_rho"], det_clim2_transposed)),
    coords=dict(
        years= range(1980, 1980+nyears),
        days= range(0, ndays),
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Mean temperature - period (1990-1999)'),
        ) 

ds_clim3 = xr.Dataset(
    data_vars=dict(temp = (["depth", "eta_rho", "xi_rho"], det_clim3_transposed)),
    coords=dict(
        years= range(1980, 1980+nyears),
        days= range(0, ndays),
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Mean temperature - period (2000-2009)'),
        ) 

ds_warm = xr.Dataset(
    data_vars=dict(temp = (["depth", "eta_rho", "xi_rho"], det_warm_transposed)),
    coords=dict(
        years= range(1980, 1980+nyears),
        days= range(0, ndays),
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Mean temperature - period (2010-2019)'),
        ) 
#%% === Create Figure with 2 Subplots ===
# --- Zonal means ---
# Mean temperature calculation for different latitude bands and datasets
datasets = {
    'clim1': ds_clim1,
    'clim2': ds_clim2,
    'clim3': ds_clim3,
    'warm': ds_warm
}

# Depth levels to select (0 to 13 inclusive)
# South of 60°S
mean_temps_south_60 = {}
for name, ds in datasets.items():
    mask = ds['lat_rho'] <= -60
    temp_masked = ds['temp'].where(mask).isel(depth=slice(0, 14))
    mean_temp = temp_masked.mean(dim=['eta_rho', 'xi_rho', 'depth'], skipna=True)
    mean_temps_south_60[name] = mean_temp.item()

overall_mean_south_60 = np.mean(list(mean_temps_south_60.values()))

# Between 50°S and 60°S
mean_temps_50_60 = {}
for name, ds in datasets.items():
    mask = (ds['lat_rho'] > -60) & (ds['lat_rho'] <= -50)
    temp_masked = ds['temp'].where(mask).isel(depth=slice(0, 14))
    mean_temp = temp_masked.mean(dim=['eta_rho', 'xi_rho', 'depth'], skipna=True)
    mean_temps_50_60[name] = mean_temp.item()

overall_mean_50_60 = np.mean(list(mean_temps_50_60.values()))

print("Overall mean south of 60°S:", overall_mean_south_60)
print("Overall mean between 50°S and 60°S:", overall_mean_50_60)


#%%
# zonal1 = ds_clim1.temp.mean(dim='xi_rho')  # 1980–1989
zonal2 = ds_clim2.temp.mean(dim='xi_rho')  # 1990–1999
zonal3 = ds_clim3.temp.mean(dim='xi_rho')  # 2000–2009
zonal4 = ds_warm.temp.mean(dim='xi_rho')   # 2010–2019
temp_avg_zonal = (zonal1+zonal2+zonal3+zonal4)/4

# --- Coordinates ---
lat = ds_clim3.lat_rho.mean(dim='xi_rho').values    # (eta_rho,)
depth = xr.open_dataset(f"{path_temp}{file_var}eta200.nc")[var][1:, 0:365, :, :].z_rho.values 

# --- Meshgrid for contourf ---
LAT, DEPTH = np.meshgrid(lat, depth)

# --- Plot ---
from matplotlib.colors import BoundaryNorm

# --- Define 21 levels between -3 and 3 ---
bounds = np.linspace(-3, 7, 35)  # 35 depth → 20 color bins
cmap = plt.get_cmap('coolwarm', len(bounds) - 1)  # Discrete colormap with 20 bins
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

# --- Plot ---
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width / 2  
fig, ax = plt.subplots(figsize=(fig_width,fig_height))

cf = ax.contourf(
    LAT, DEPTH, temp_avg_zonal.values,
    levels=bounds, cmap=cmap, norm=norm)

# ax.set_title("Average Zonal Temperature (1980–2019)", fontsize=18)
ax.set_xlabel("Latitude (°N)",fontsize=14)
ax.set_ylabel("Depth (m)",fontsize=14)
ax.set_xlim(-78, -50)
yticks = np.array([-5, -100, -200, -300, -400, -500])
ax.set_yticks(yticks)
ax.set_yticklabels([str(t) for t in yticks])  # shows depth as positive numbers

ax.tick_params(labelsize=12)

# --- Colorbar ---
cbar = fig.colorbar(cf, ax=ax, orientation='vertical', label='Temperature (°C)', extend='both')
cbar.set_label('Temperature (°C)', fontsize=14)  # bigger label font
cbar.set_ticks(np.arange(-3, 7+1, 1))
cbar.ax.tick_params(labelsize=12, which='major', length=2)  # Larger ticks, no minors

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/zonal_avg_temperature.pdf'), dpi =150, format='pdf', bbox_inches='tight')


# %%

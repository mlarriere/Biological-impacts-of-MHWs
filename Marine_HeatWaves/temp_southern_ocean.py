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
from matplotlib.colors import ListedColormap

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


# %% ================== Extract Temperature for different periods ==================
def extract_temperature_fullcolumn(ieta, period):
    start_time = time.time()
    # ieta =200
    fn = f"{path_temp}{file_var}eta{ieta}.nc"
    ds = xr.open_dataset(fn)[var][1:, 0:365, 0:35, :]  # Shape: (40, 365, 35, 1442)

    if period == True: 
        # --- Average over time (full period by 10yrs bin) ---
        ds_clim1 = ds.isel(year=slice(0,10)).mean(dim=('year','day'), skipna=True)
        ds_clim2 = ds.isel(year=slice(10,20)).mean(dim=('year','day'), skipna=True)
        ds_clim3 = ds.isel(year=slice(20,30)).mean(dim=('year','day'), skipna=True)
        ds_warm  = ds.isel(year=slice(30,40)).mean(dim=('year','day'), skipna=True)

        print(f"Finished eta {ieta} in {time.time() - start_time:.2f}s")
        return ieta, ds_clim1, ds_clim2, ds_clim3, ds_warm

    else:
        # --- Average over time (full period) ---
        ds_avg = ds.mean(dim=('year', 'day'), skipna=True)
        print(f"Finished eta {ieta} in {time.time() - start_time:.2f}s")
        return ieta, ds_avg


def create_temp_dataset(data, description):
    return xr.Dataset(data_vars=dict(temp=(["depth", "eta_rho", "xi_rho"], data)),
                      coords=dict(
                          depth=range(nz),
                          lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values),
                          lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values),
                          ),
                      attrs=dict(description=description),
                      )

from functools import partial
period = False  # True False
print(f'Extracting T°C for different periods: {period}')
extract_temp = partial(extract_temperature_fullcolumn, period=period)

if period:
    # --- Temperature of the first 100m in the water column (AVERAGE for each 10yrs period) ---
    det_clim1 = np.empty((neta, nz, nxi), dtype=np.float32)
    det_clim2 = np.empty((neta, nz, nxi), dtype=np.float32)
    det_clim3 = np.empty((neta, nz, nxi), dtype=np.float32)
    det_warm  = np.empty((neta, nz, nxi), dtype=np.float32)

    for ieta, ds_clim1, ds_clim2, ds_clim3, ds_warm in process_map(extract_temp, range(neta), max_workers=30, desc="Processing eta"):
        det_clim1[ieta] = ds_clim1
        det_clim2[ieta] = ds_clim2
        det_clim3[ieta] = ds_clim3
        det_warm[ieta]  = ds_warm
        
    # Flip eta position
    det_clim1_transposed = det_clim1.transpose(1, 0, 2) #shape (35, 434, 1442)
    det_clim2_transposed = det_clim2.transpose(1, 0, 2) #shape (35, 434, 1442)
    det_clim3_transposed = det_clim3.transpose(1, 0, 2) #shape (35, 434, 1442)
    det_warm_transposed = det_warm.transpose(1, 0, 2)

    # To dataset
    ds_clim1 = create_temp_dataset(det_clim1_transposed, "Mean temperature - period (1980–1989)")
    ds_clim2 = create_temp_dataset(det_clim2_transposed, "Mean temperature - period (1990–1999)")
    ds_clim3 = create_temp_dataset(det_clim3_transposed, "Mean temperature - period (2000–2009)")
    ds_warm  = create_temp_dataset(det_warm_transposed,  "Mean temperature - period (2010–2019)")

else:
    # --- Full column average temperature over all years (1980–2019) ---
    det_avg = np.empty((neta, nz, nxi), dtype=np.float32)

    for ieta, ds_avg in process_map(extract_temp, range(neta), max_workers=30, desc="Processing eta"):
        det_avg[ieta] = ds_avg # shape (35, 1442)

    # Flip eta dimension
    det_avg_transposed = det_avg.transpose(1, 0, 2)  # shape: (depth, eta_rho, xi_rho)

    # To dataset
    ds_avg_temp = create_temp_dataset(det_avg_transposed, "Mean temperature over all years (1980–2019)")


# %% ================== Temperature of the first 100m in the water column (NOT TIME AVERAGE) ==================
def extract_temperature_100m(ieta):
    start_time = time.time()
    fn = f"{path_temp}{file_var}eta{ieta}.nc"
    ds = xr.open_dataset(fn)[var][1:, 0:365, 0:14, :]  # Shape: (40, 365, 14, 1442)
    
    # Mean temperature in the first 100m of the water column
    ds_avg = ds.mean(dim='z_rho', skipna=True)
    print(f"Finished eta {ieta} in {time.time() - start_time:.2f}s")

    return ieta, ds_avg

det_avg_temp = np.empty((neta, nyears, ndays, nxi), dtype=np.float32)
for ieta, ds_avg_val in process_map(extract_temperature_100m, range(neta), max_workers=30, desc="Processing eta"):
    det_avg_temp[ieta] = ds_avg_val #shape (40, 365, 1442)

# Flip eta position
det_avg_temp_transposed = det_avg_temp.transpose(1, 2, 0, 3) #shape (40, 365, 434, 1442)

# To dataset
ds_avg_temp_100m = xr.Dataset(
    data_vars=dict(temp = (["years", "days", "eta_rho", "xi_rho"], det_avg_temp_transposed)),
    coords=dict(
        years= range(1980, 1980+nyears),
        days= range(0, ndays),
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Averaged temperature over the full water column (5-100m)'),
        ) 

# Write to NCDF
file_path=os.path.join(path_clim, 'avg_temp_watercolumn.nc')
if not os.path.exists(file_path):
    ds_avg_temp_100m.to_netcdf(file_path)
    print(f"Saved: {file_path}")
else:
    print(f"File already exists: {file_path}")

#%% ================== Calculate Zonal and Hotspots Mean Temperature ==================
def compute_zonal_mean(ds, lat_cond, lon_cond=None, depth_range=slice(0, 14)):
    mask = lat_cond(ds['lat_rho'])
    if lon_cond is not None:
        mask = mask & lon_cond(ds['lon_rho'])
    temp_masked = ds.temp.where(mask).isel(depth=depth_range)
    mean_temp = temp_masked.mean(dim=['eta_rho', 'xi_rho', 'depth'], skipna=True)
    return mean_temp.item()

if period == False:
    print("=== Mean temperatures (1980–2019) ===")

    temp_south_60 = compute_zonal_mean(ds=ds_avg_temp, lat_cond=lambda lat: lat <= -60)
    print(f"Southern Ocean - South of 60°S: {temp_south_60:.1f}°C")

    temp_50_60 = compute_zonal_mean(ds=ds_avg_temp, lat_cond=lambda lat: (lat > -60) & (lat <= -50))
    print(f"Southern Ocean - Between 50°S and 60°S: {temp_50_60:.1f}°C")

    temp_hotspot1 = compute_zonal_mean(ds=ds_avg_temp, lat_cond=lambda lat: (lat > -65) & (lat <= -60), lon_cond=lambda lon: (lon <= 292) & (lon > 240))
    print(f"Hotspot 1 (60-65°S 240-292°E): {temp_hotspot1:.1f}°C")

    temp_hotspot2 = compute_zonal_mean(ds=ds_avg_temp, lat_cond=lambda lat: (lat > -65) & (lat <= -60), lon_cond=lambda lon: (lon <= 195) & (lon > 160))
    print(f"Hotspot 2 (60-65°S 160-195°E): {temp_hotspot2:.1f}°C")


# %% ================== Plot Hotspots Regions ==================
mhw_duration = xr.open_dataset(os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc')))
lon = mhw_duration['lon_rho'].values % 360
lat = mhw_duration['lat_rho'].values

# -- Parameters --
year_idx = 37 
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
variables = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
titles = ['1°C', '2°C', '3°C', '4°C']

# Hotspot masks
hotspot_mask1 = (lat > -65) & (lat <= -60) & (lon > 240) & (lon <= 292)
hotspot_mask2 = (lat > -65) & (lat <= -60) & (lon > 160) & (lon <= 195)

# Plot setup
fig = plt.figure(figsize=(6.5, 6.5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Map base
ax.coastlines(color='black', linewidth=1, zorder=4)
ax.add_feature(cfeature.LAND, zorder=2, facecolor='#F6F6F3')
ax.set_facecolor('lightgrey')

for lon_line in [-90, 0, 120]:
    ax.plot([lon_line, lon_line], [-90, -60], transform=ccrs.PlateCarree(),
            color="#080808", linestyle='--', linewidth=1, zorder=5)

# Gridlines
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7, zorder=3)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()

# Plot each threshold detection overlayed
for i, var in enumerate(variables):
    data = mhw_duration[var].isel(years=year_idx).mean(dim={'days'})
    masked_data = np.where(data, 1, np.nan)  # mask False values
    ax.pcolormesh(lon, lat, masked_data, transform=ccrs.PlateCarree(),
                  cmap=ListedColormap([threshold_colors[i]]), shading='auto',
                  zorder=1, rasterized=True)

ax.contour(lon, lat, hotspot_mask1, levels=[0.5], colors='red', linewidths=2, transform=ccrs.PlateCarree(), zorder=5)
ax.contour(lon, lat, hotspot_mask2, levels=[0.5], colors='blue', linewidths=2, transform=ccrs.PlateCarree(), zorder=5)

# Title
ax.set_title(f"MHW Detected Events\nOn Average in {1980 + year_idx} (5m depth)", fontsize=16)

# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=threshold_colors[i], edgecolor='k', label=f'MHWs $>${titles[i]}') for i in range(4)]
legend_patches.append(Patch(facecolor='none', edgecolor='red', label=f'Hotspot Region 1', linewidth=1))
legend_patches.append(Patch(facecolor='none', edgecolor='blue', label=f'Hotspot Region 2', linewidth=1))
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, frameon=True,
          bbox_to_anchor=(0.01, -0.2), borderaxespad=0)

plt.tight_layout()
plt.show()

#%% ================== Plot Zonal Mean Southern Ocean for introduction ==================
# --- Load data ---
temp_avg_zonal = ds_avg_temp.temp.mean(dim='xi_rho')

# --- Coordinates for the axis ---
lat = ds_avg_temp.lat_rho.mean(dim='xi_rho').values    # (eta_rho,)
depth = xr.open_dataset(f"{path_temp}{file_var}eta200.nc")['temp'][1:, 0:365, :, :].z_rho.values 

# --- Meshgrid for contourf ---
LAT, DEPTH = np.meshgrid(lat, depth)

# --- Plot ---
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap

# --- Define 21 levels between -3 and 3 ---
vmin, vmax = -3, 7
color_positions = np.array([vmin, -1.5, 0, 1.5, 3.5, vmax])  # Accelerated warm transition
normalized_positions = (color_positions - vmin) / (vmax - vmin)

# Ensure white is at 0.3 normalized position
colors = ["#001219", "#669BBC", "#FFFFFF", "#CA6702", "#AE2012", "#5C0101"]
cmap = LinearSegmentedColormap.from_list("blue_green_yellow_buffered", list(zip(normalized_positions, colors)), N=256)
bounds = np.linspace(vmin, vmax, 35)
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

# --- Plot ---
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width / 2  
fig, ax = plt.subplots(figsize=(fig_width,fig_height))

cf = ax.contourf(
    LAT, DEPTH, temp_avg_zonal.values,
    levels=bounds, cmap=cmap, norm=norm)

# ax.set_title("Average Zonal Temperature (1980–2019)", fontsize=18)
ax.set_xlabel("Latitude [°N]")
ax.set_ylabel("Depth [m]")
ax.set_xlim(-78, -50)
yticks = np.array([-5, -100, -200, -300, -400, -500])
ax.set_yticks(yticks)
ax.set_yticklabels([str(t) for t in yticks])  # shows depth as positive numbers

ax.set_xlim(-78, -50)
yticks = np.array([-5, -100, -200, -300, -400, -500])
ax.set_yticks(yticks)
ax.set_yticklabels([str(t) for t in yticks])  # shows depth as positive numbers

ax.axvline(x=-60, color='black', linestyle=':', linewidth=0.8, zorder=10)
ax.axhline(y=-100, color='black', linestyle=':', linewidth=0.8, zorder=10)
# ax.tick_params(labelsize=12)

# --- Colorbar ---
cbar = fig.colorbar(cf, ax=ax, orientation='vertical', label='Temperature [°C]', extend='both')
cbar.set_label('Temperature [°C]')
cbar.set_ticks(np.arange(vmin, vmax+1, 1))  # round integer ticks only
cbar.minorticks_off()
cbar.ax.tick_params( which='major', length=5)

plt.tight_layout()
plt.show() 
# plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/zonal_avg_temperature.pdf'), dpi =200, format='pdf', bbox_inches='tight')


# %% --------------------------- PLOT SECTORS  ---------------------------
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# === Layout config ===
plot = 'report'
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width/2
else:
    fig_width = 16
    fig_height = 8

fig = plt.figure(figsize=(fig_width, fig_height))
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

title_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'} if plot == 'slides' else {}
gridlabel_kwargs = {'size': 9, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}

# Set circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Set map extent
lw = 1 if plot == 'slides' else 0.5
ax.set_extent([-180, 180, -90, -40], ccrs.PlateCarree())  # or -25/-20 as needed
ax.set_facecolor("#D6D6D6")
ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)
ax.coastlines(color='black', linewidth=lw)


# --- Fill Pacific Sector (wraps around date line) ---
# 120°E to 180°
lons1 = np.linspace(120, 180, 100)
lats1 = np.full_like(lons1, -60)
ax.fill(np.concatenate(([180], lons1[::-1], [120])), np.concatenate(([-90], lats1, [-90])), transform=ccrs.PlateCarree(), color='#CA6702', alpha=0.75, zorder=0)

# -180° to -90°
lons2 = np.linspace(-180, -90, 100)
lats2 = np.full_like(lons2, -60)
ax.fill(np.concatenate(([-90], lons2[::-1], [-180])), np.concatenate(([-90], lats2, [-90])), transform=ccrs.PlateCarree(), color='#CA6702', alpha=0.75, zorder=0)

# --- Fill Atlantic Sector: -90° to 0°
lons = np.linspace(-90, 0, 100)
lats = np.full_like(lons, -60)
ax.fill(np.concatenate(([0], lons[::-1], [-90])), np.concatenate(([-90], lats, [-90])), transform=ccrs.PlateCarree(), color='#057985', alpha=0.75, zorder=0)

# --- Fill Indian Sector: 0° to 120°
lons = np.linspace(0, 120, 100)
lats = np.full_like(lons, -60)
ax.fill(np.concatenate(([120], lons[::-1], [0])), np.concatenate(([-90], lats, [-90])), transform=ccrs.PlateCarree(), color='#9B2226', alpha=0.75, zorder=0)

# --- Sector boundaries ---
lw_sector = 1.2 if plot == 'slides' else 0.8
# Draw circle at 60°S
circle_lat = -60
circle_lons = np.linspace(-180, 180, 500)
circle_lats = np.full_like(circle_lons, circle_lat)

ax.plot(circle_lons, circle_lats, transform=ccrs.PlateCarree(),
        color='black', linestyle='--', linewidth=lw_sector, zorder=3)

sector_lons = [-90, 0, 120]
for lon in sector_lons:
    ax.plot([lon, lon], [-90, -60], transform=ccrs.PlateCarree(),
            color='#080808', linestyle='--', linewidth=lw_sector)

# --- Sector labels ---
label_kwargs = suptitle_kwargs.copy()
label_kwargs.update({
    'ha': 'center',
    'va': 'center',
    'bbox': dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
})
ax.text(-135, -60, r'{Pacific Sector}', transform=ccrs.PlateCarree(), **label_kwargs)
ax.text(-45, -60, r'{Atlantic Sector}', transform=ccrs.PlateCarree(), **label_kwargs)
ax.text(50, -55, r'{Indian Sector}', transform=ccrs.PlateCarree(), **label_kwargs)

# --- Gridlines ---
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = gridlabel_kwargs
gl.ylabel_style = gridlabel_kwargs
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# --- Output handling ---
outdir = os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/')
os.makedirs(outdir, exist_ok=True)

if plot == 'report':
    outfile = "sectors_report.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    # plt.show()
else:
    outfile = "sectors_slides.png"
    # plt.savefig(os.path.join(outdir, outfile), dpi=500, format='png', bbox_inches='tight')
    plt.show()


# %%

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
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_biomass_surrogates = os.path.join(path_surrogates, f'biomass_timeseries')


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


# # %% ======================== Plot MPAs ========================
# fig = plt.figure(figsize=(5, 8))
# gs = gridspec.GridSpec(nrows=1, ncols=1)
# ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())

# # Circular boundary
# theta = np.linspace(0, 2 * np.pi, 200)
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * 0.5 + 0.5)
# ax.set_boundary(circle, transform=ax.transAxes)

# # Base map
# ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
# ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

# ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# # == Plot MPA
# lon = mpas_ds.lon_rho
# lat = mpas_ds.lat_rho
# for name, (mask, color) in mpa_dict.items():

#     # Mask lon/lat
#     lon_masked = lon.where(mask)
#     lat_masked = lat.where(mask)

#     ax.scatter(lon_masked.values, lat_masked.values,
#                s=2, color=color, transform=ccrs.PlateCarree(), 
#                label=name, alpha=0.8, zorder=1)

#     # == Add name of the MPA
#     # Center of the box
#     lon_centroid = float(lon_masked.mean().values)+10
#     lat_centroid = float(lat_masked.mean().values)

#     # Add text box with colored border
#     ax.text(
#         lon_centroid,
#         lat_centroid,
#         name,
#         transform=ccrs.PlateCarree(),
#         fontsize=10,
#         fontweight='bold',
#         ha='center',
#         va='center',
#         bbox=dict(
#             facecolor="white",
#             edgecolor=color,
#             linewidth=1.5,
#             boxstyle="round,pad=0.3",
#             alpha=0.9
#         ),
#         zorder=5
#     )

# # Legend and title
# # ax.legend(loc="lower left", fontsize=9)
# ax.set_title("Southern Ocean MPAs", fontsize=12)

# plt.tight_layout()
# plt.show()

# %% ======================== Areas and volume MPAs ========================
# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km2
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km3

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)

# --- Calculate area and volume of each MPA
mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}

area_mpa = {}
volume_mpa = {}

for abbrv, (name, mask) in mpa_masks.items():
    area_mpa[abbrv] = area_60S_SO.where(mask)
    volume_mpa[abbrv] = volume_60S_SO_100m.where(mask)
    volume_mpa[name] = volume_60S_SO_100m.where(mask)

# %% ======================== Biomass data ========================
mass = xr.open_dataset(os.path.join(path_surrogates, "mass_length/clim_mass_stages_SO.nc"))
# ==== Load data ====
# -- Southern Ocean (from Biomass_calculations.py)
biomass_clim = xr.open_dataset(os.path.join(path_biomass_surrogates, "biomass_clim.nc"))
biomass_clim=biomass_clim.assign_coords(lon_rho=(("eta_rho", "xi_rho"), mass.lon_rho.data), lat_rho=(("eta_rho", "xi_rho"), mass.lat_rho.data),)
biomass_clim=biomass_clim.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_actual = xr.open_dataset(os.path.join(path_biomass_surrogates, "biomass_actual.nc"))
biomass_actual=biomass_actual.assign_coords(lon_rho=(("eta_rho", "xi_rho"), mass.lon_rho.data), lat_rho=(("eta_rho", "xi_rho"), mass.lat_rho.data),)
biomass_actual=biomass_actual.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_noMHWs = xr.open_dataset(os.path.join(path_biomass_surrogates, "biomass_nomhws.nc"))
biomass_noMHWs=biomass_noMHWs.assign_coords(lon_rho=(("eta_rho", "xi_rho"), mass.lon_rho.data), lat_rho=(("eta_rho", "xi_rho"), mass.lat_rho.data),)
biomass_noMHWs=biomass_noMHWs.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_nowarming = xr.open_dataset(os.path.join(path_biomass_surrogates, "biomass_nowarming.nc"))
biomass_nowarming=biomass_nowarming.assign_coords(lon_rho=(("eta_rho", "xi_rho"), mass.lon_rho.data), lat_rho=(("eta_rho", "xi_rho"), mass.lat_rho.data),)
biomass_nowarming=biomass_nowarming.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_ds = {"clim": biomass_clim, "actual": biomass_actual, "nomhws": biomass_noMHWs, "nowarming": biomass_nowarming,}
surrogate_names = {"clim": "Climatology", "actual": "Actual Conditions", "nomhws": "No Marine Heatwaves", "nowarming": "No Warming"}
mpa_names = list(mpa_dict.keys())
mpa_abbrs = list(mpa_masks.keys())

# %% ======================== Mask MPAs ========================
from functools import partial
from tqdm.contrib.concurrent import process_map 

# --- Function to mask one surrogate for one MPA
def mask_one_surrogate(args):
    surrog, ds, mpa_name, mpa_abbrv, mpa_mask = args

    # Expand mask dimensions
    mask = mpa_mask
    if "years" in ds.dims:
        mask = mask.expand_dims({"years": ds.years.size}, axis=1)
    if "days" in ds.dims:
        mask = mask.expand_dims({"days": ds.days.size}, axis=2)
    
    # Mask biomass to MPAs
    biomass_masked = ds.biomass_median.where(mask)
    std_masked = ds.biomass_std.where(mask)
    
    # To Dataset
    ds_mpa = xr.Dataset({"biomass_median": biomass_masked, "biomass_std": std_masked},
                        coords={"years": ds.years if "years" in ds.dims else None,
                                "days": ds.days,
                                "lon_rho": (("eta_rho", "xi_rho"), ds.lon_rho.data),
                                "lat_rho": (("eta_rho", "xi_rho"), ds.lat_rho.data),},
                        attrs={"mpa_name": mpa_name})
    
    # Save
    file_path = os.path.join(path_biomass_surrogates, f"mpas/{surrog}_biomass_{mpa_abbrv}.nc")
    ds_mpa.to_netcdf(file_path)

    return f"Saved {mpa_name}"

# %% ======================== Mask MPAs ========================
output_folder=os.path.join(path_surrogates, f'biomass_timeseries/mpas')
files = [os.path.join(output_folder, f"{sur}_biomass_{abbrv}.nc") for sur in surrogate_names.keys() for abbrv in mpa_masks.keys()]

if not all(os.path.exists(f) for f in files):
    print('Masking biomass to MPAs and writing to file...')

    # --- Loop over MPAs
    for abbrv, (mpa_name, mpa_mask) in mpa_masks.items():
        # Test
        # abbrv='RS'
        # mpa_name='Ross Sea'
        # mpa_mask=mpa_masks['RS'][1]

        print(f"\nProcessing MPA: {mpa_name} ({abbrv})")

        # Prepare arguments for function
        args_list = [(surrog, ds, mpa_name, abbrv, mpa_mask) for surrog, ds in biomass_ds.items()]
        
        # Run in parallel
        results = process_map(mask_one_surrogate, args_list, max_workers=4,  desc=f"{abbrv} | Mask ")

else:
    print('Loading files...')
    biomass_mpas = {}

    for abbrv, (mpa_name, _) in mpa_masks.items():
        biomass_mpas[abbrv] = {}

        for surrog in surrogate_names.keys():
            fname = os.path.join(output_folder, f"{surrog}_biomass_{abbrv}.nc")
            biomass_mpas[abbrv][surrog] = xr.open_dataset(fname)

    

# # --- Apply for all surrogates
# biomass_mpa_ds = {}
# for surrog, ds in biomass_ds.items():
#     print(f"Masking {surrogate_names[surrog]}")
#     biomass_mpa_ds[surrog] = mask_one_surrogate(ds, mpa_mask)

# # -- To Dataset
# biomass_mpa_combined = {}
# for surrog, masked_ds in biomass_mpa_ds.items():
#     biomass_mpa_combined[surrog] = xr.Dataset({"biomass_median": masked_ds.biomass_median,
#                                             "biomass_std": masked_ds.biomass_std},
#                                             coords={"years": masked_ds.years if "years" in masked_ds.dims else None,
#                                                     "days": masked_ds.days,
#                                                     "lon_rho": (("eta_rho", "xi_rho"), masked_ds.lon_rho.data),
#                                                     "lat_rho": (("eta_rho", "xi_rho"), masked_ds.lat_rho.data)},
#                                             attrs={"mpa_name": mpa_name})
# # -- Save to file
# save_dir = os.path.join(path_biomass_surrogates, "mpas")
# for surrog, ds in biomass_mpa_combined.items():
#     file_path = os.path.join(save_dir, f"{surrog}_biomass_AP.nc")
#     print(f"Saving {surrog} biomass dataset to {file_path}")
#     ds.to_netcdf(file_path)

# print(biomass_mpa_ds["actual"])

# %% ================================= Plot Biomass concentration =================================
# --- Prepare data
mpa="Antarctic Peninsula"
dataset_interest =  biomass_mpa_ds['nowarming'] #actual, nomhws, nowarming
dataset_ref = biomass_mpa_ds['clim']
# title = "Actual environmental conditions"
# title = "Environmental conditions without surface MHWs"
title = "Environmental conditions without global warming"

# Years to show
years_to_plot = [1980, 1989, 2000, 2010, 2016]

# Row 1: Biomass 
biomass_actual_30Apr = [dataset_interest.biomass_median.isel(years=i, days=-1) for i in range(len(years_to_plot))]  #biomass_actual_30Apr[0].shape: (231, 1440)

# Row 2: Difference (end of season)
diff_actual = [biomass_actual_30Apr[i] - dataset_ref.biomass_median.isel(days=-1) for i in range(len(years_to_plot))] #diff_actual[0].shape: (231, 1440)

# --- Figure setup
ncols = len(biomass_actual_30Apr)
fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(nrows=2, ncols=ncols, wspace=0.08, hspace=0.3)

# --- Circular boundary
theta = np.linspace(np.pi/2, np.pi , 100)  # from 0° to -90° clockwise - Quarter-circle sector boundary
center, radius = [0.5, 0.51], 0.5 # centered at 0.5,0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

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
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(color='black', linewidth=0.7, zorder=3)

    im1 = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                        cmap=cmap_row1, vmin=vmin_row1, vmax=vmax_row1,
                        transform=ccrs.PlateCarree(), zorder=1)

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
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
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
fig.text(0.52, 0.5, "Comparison with Climatology ", ha='center', fontsize=16)

# --- Overall figure title
plt.suptitle(f"Krill Biomass on 30th April\n{mpa}", fontsize=18, y=1.1, x=0.52)
plt.show()

# 
# %% ================================= MAsk MHW to match Biomass =================================
# Biomass and MHW datasets
biomass_data = biomass_mpa_ds['actual'].biomass_median  # shape (39, 181, 231, 1440)
mhw_ap = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/duration_AND_thresh_AP.nc')) #shape (39, 181, 231, 1440)

# --- Create mask: True where at least one day has biomass
biomass_valid_mask = biomass_data.notnull().any(dim='days')  # shape (39, 231, 1440)
biomass_valid_mask_expanded = biomass_valid_mask.expand_dims({'days': mhw_ap.sizes['days']}, axis=1) #shape (39, 181, 231, 1440)

# --- Mask MHW variables
mhw_vars = ['duration', 'det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
mhw_masked_vars = {}
for var in mhw_vars:
    mhw_masked_vars[var] = mhw_ap[var].where(biomass_valid_mask_expanded)

# --- To Dataset
mhw_masked_ds = xr.Dataset(mhw_masked_vars,
                           coords={'years': mhw_ap.years, 'days': mhw_ap.days,
                                   'lon_rho': (('eta_rho', 'xi_rho'), mhw_ap.lon_rho.data),
                                   'lat_rho': (('eta_rho', 'xi_rho'), mhw_ap.lat_rho.data),},
                                   attrs=mhw_ap.attrs)
mhw_masked_ds = mhw_masked_ds.assign_attrs(mpa_name=mhw_ap.attrs.get('mpa_name', 'AP'))

# Save
mhw_masked_ds.to_netcdf(os.path.join(path_combined_thesh, 'mpas/mhws_AP_masked.nc')) 

mhw_masked_ds = mhw_masked_ds.assign_attrs(mpa_name=mhw_data.attrs.get('mpa_name', 'Unknown'))
# %% ================================= Location with longest MHWs events =================================
# Select threshold
det3deg = False #True #False
det4deg = True #False #True 
det1deg = False #False #True 
if det3deg:
    threshold= 'det_3deg'
if det4deg:
    threshold= 'det_4deg'
if det1deg:
    threshold= 'det_1deg'

# Threshold mask
duration_filtered = mhw_ap['duration'].where(mhw_ap[threshold]) #shape (39, 181, 231, 1440)

# Find when and where
index = np.unravel_index(np.nanargmax(duration_filtered.values), duration_filtered.shape)
year_idx, day_idx, eta_idx, xi_idx = index
year = ds['years'].values[year_idx]
day = ds['days'].values[day_idx]
lat = ds['lat_rho'].values[eta_idx, xi_idx]
lon = ds['lon_rho'].values[eta_idx, xi_idx]
duration = duration_filtered.values[index]

print(f"Longest {threshold} MHW lasted {duration:.1f} days")
print(f"Year: {year}, Day-of-year: {day}")
print(f"Location: lat={lat:.2f}, lon={lon:.2f}")

# ==== 1°C ====
# Longest det_1deg MHW lasted 142.0 days
# Year: 20, Day-of-year: 38
# Location: lat=-65.33, lon=295.38

# ==== 3°C ====
# Longest det_3deg MHW lasted 142.0 days
# Year: 20, Day-of-year: 59
# Location: lat=-65.33, lon=295.38

# ==== 4°C ====
# Longest det_4deg MHW lasted 142.0 days
# Year: 20, Day-of-year: 71
# Location: lat=-65.33, lon=295.38

# %% ================================= Biomass timeseries =================================
# Look at the shap of biomass evlution -- see if biomass influence the progretion of biomass increase over the season
# Flatten the grid and compute the distance
lat_grid = biomass_mpa_ds['clim'].lat_rho.values
lon_grid = biomass_mpa_ds['clim'].lon_rho.values

# Compute squared distance to target location
dist2 = (lat_grid - lat)**2 + (lon_grid - lon)**2

# Find index of minimum distance
eta_idx, xi_idx = np.unravel_index(dist2.argmin(), dist2.shape)

print("Grid indices:", eta_idx, xi_idx)
print("Grid location:", lat_grid[eta_idx, xi_idx], lon_grid[eta_idx, xi_idx])

# Extract biomass
biomass_mhw = biomass_mpa_ds['actual'].biomass_median.isel(years=year_idx, eta_rho=eta_idx, xi_rho=xi_idx)

# Find nearest eta_rho/xi_rho index
eta_idx = np.abs(biomass_mpa_ds['clim'].lat_rho[:,0] - lat).argmin()
xi_idx  = np.abs(biomass_mpa_ds['clim'].lon_rho[0,:] - lon).argmin()

biomass_mhw = biomass_mpa_ds['actual'].biomass_median.isel(years=20, eta_rho=eta_idx, xi_rho=xi_idx)


# Mean number of MHWs days per year
mhw_duration_avg = mhw_event_masked_ds.duration.mean(dim='years')

# %% 
# --- Figure setup
fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1, wspace=0.08, hspace=0.3)

# --- Circular boundary
theta = np.linspace(np.pi/2, np.pi , 100)  # from 0° to -90° clockwise - Quarter-circle sector boundary
center, radius = [0.5, 0.51], 0.5 # centered at 0.5,0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

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
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(color='black', linewidth=0.7, zorder=3)

    im1 = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                        cmap=cmap_row1, vmin=vmin_row1, vmax=vmax_row1,
                        transform=ccrs.PlateCarree(), zorder=1)

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
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
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
fig.text(0.52, 0.5, "Comparison with Climatology ", ha='center', fontsize=16)

# --- Overall figure title
plt.suptitle(f"Krill Biomass on 30th April\n{mpa}", fontsize=18, y=1.1, x=0.52)
plt.show()

# %% Look at the biomass total over MPAs evolution in time (season line and color=years)

# %%
# ==== MPAs mask for biomass ====
mpa_names = list(mpa_dict.keys())
mpa_abbrs = list(mpa_masks.keys())
# --- Stack MPA masks into 1 DataArray
mpa_masks_stack = xr.concat([mpas_south60S[mpa_dict[name][0].name] for name in mpa_dict.keys()], dim="mpa")
mpa_masks_stack = mpa_masks_stack.assign_coords(mpa=list(mpa_dict.keys()))

# --- MPAs mask biomass
biomass_mpa_ds = {}
for surrog, ds in biomass_ds.items():
    # test
    # surrog = 'actual'
    # ds = biomass_ds[surrog]

    print(f'Computing {surrog}...')

    # Create mpa dimension and expand biomass along this dimension
    biomass_expanded = ds.biomass_median.expand_dims({"mpa": list(mpa_dict.keys())}, axis=0) #shape (5, 39, 181, 231, 1440)
    std_expanded = ds.biomass_std.expand_dims({"mpa": list(mpa_dict.keys())}, axis=0)
    
    # Broadcast MPA mask over years/days
    mask_broadcast = mpa_masks_stack
    if 'years' in ds.dims:
        mask_broadcast = mask_broadcast.expand_dims({"years": ds.years.size}, axis=1)
    if 'days' in ds.dims:
        mask_broadcast = mask_broadcast.expand_dims({"days": ds.days.size}, axis=2) #shape (5, 39, 181, 231, 1440)
    
    # Apply mask
    biomass_masked = biomass_expanded.where(mask_broadcast)
    std_masked = std_expanded.where(mask_broadcast)
    
    # To dataset
    biomass_mpa_ds[surrog] = xr.Dataset({"biomass_median": biomass_masked,
                                         "biomass_std": std_masked},
                                         coords={"mpa": list(mpa_dict.keys()),
                                                 "years": ds.years,
                                                 "days": ds.days,
                                                 "lon_rho": (("eta_rho", "xi_rho"), ds.lon_rho.data),
                                                 "lat_rho": (("eta_rho", "xi_rho"), ds.lat_rho.data),})
    
biomass_mpa_ds = {}
for surrog, ds in biomass_ds.items():
    print(f'Computing {surrog}...')
    # test
    surrog = 'actual'
    ds = biomass_actual

    # Add mpa dimension
    ds = ds.assign_coords(
        lon_rho=(("eta_rho", "xi_rho"), area_60S_SO.lon_rho.data),
        lat_rho=(("eta_rho", "xi_rho"), area_60S_SO.lat_rho.data)
    )
    biomass_expanded = ds.biomass_median.expand_dims({"mpa": mpa_names}, axis=0) #shape (5, 181, 231, 1440) for clim, (5, 39, 181, 231, 1440) for others
    mask_expanded = mpa_masks_stack
    
    # Apply mask -- shape (5, 181, 231, 1440) for clim, (5, 39, 181, 231, 1440) for others
    biomass_masked = biomass_expanded.where(mask_expanded)
    std_masked = ds.biomass_std.expand_dims({"mpa": mpa_names}, axis=0).where(mask_expanded) 
    
    # To Dataset
    biomass_mpa_ds[surrog] = xr.Dataset({"biomass_median": biomass_masked,
                                         "biomass_std": std_masked})


# Check
print(biomass_mpa_ds["actual"]["biomass_median"].sel(mpa="Antarctic Peninsula"))





# %%


# %% ==== Total Biomass ====
for mpa_name, mpa_abbr in zip(mpa_names, mpa_abbrs):
    mpa_name=mpa_names[0]
    mpa_abbr=mpa_abbrs[0]
    print(f"Processing MPA: {mpa_name}")

    # --- Geometry masked to this MPA
    mask = mpa_masks[mpa_abbr]  # (eta_rho, xi_rho)

    area_mpa = area_60S_SO.where(mask)
    volume_mpa = volume_60S_SO_100m.where(mask)

    # Dataset that will contain ALL surrogates for this MPA
    ds_mpa = xr.Dataset()

    for surrog, ds in biomass_mpa_ds.items():
        # surrog='clim'
        # ds=biomass_mpa_ds[surrog]
        print(f"  Surrogate: {surrog}")

        # --- Select this MPA
        biomass_med = ds["biomass_median"].sel(mpa=mpa_name)
        biomass_std = ds["biomass_std"].sel(mpa=mpa_name)

        # --- Biomass per grid cell - shape (39, 181, 231, 1440)
        total_med = biomass_med * (volume_mpa*1e9)/ (area_mpa*1e6)
        total_std = biomass_std * (volume_mpa*1e9)/ (area_mpa*1e6)

        # --- Sum - shape (39, 181)
        med_sum = total_med.sum(dim=("eta_rho", "xi_rho"), skipna=True)
        std_sum = total_std.sum(dim=("eta_rho", "xi_rho"), skipna=True)

        # --- Store with surrogate-specific names
        ds_mpa[f"tot_{surrog}_median"] = med_sum
        ds_mpa[f"tot_{surrog}_std"] = std_sum

    # --- Dataset-level attributes
    ds_mpa.attrs = {
        "MPA": mpa_name,
        "description": (
            "Total biomass over the MPA.\n "
            "Biomass time series were calculated for five models with ten bootstrap runs each.\n "
            "Initial biomass conditions were provided by CEPHALOPOD.\n"
            "Biomass was propagated using the growth model of Atkinson et al. (2006).\n "
            "Median and standard deviation were calculated across all models and bootstraps "
            "after the biomass time series had been computed."
        ),
        "surrogates": ", ".join([surrogate_names[s] for s in biomass_mpa_ds.keys()]),
    }


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

%%

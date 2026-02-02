"""
Created on Tues 20 Jan 10:49:30 2025

MHWs and their characteristics in the MPAs

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

# %% ======================== Mask ========================
mhw_events_surface = xr.open_dataset(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc')) #shape (39, 181, 231, 1442)
mhw_events_surface = mhw_events_surface.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))  #shape (39, 181, 231, 1440)
mhw_vars = ["duration", "det_1deg", "det_2deg", "det_3deg", "det_4deg"]

for abbrv, (name, mask_2d) in mpa_masks.items():
    output_file = os.path.join(path_combined_thesh, f'mpas/duration_AND_thresh_{abbrv}.nc')

    if not os.path.exists(output_file):
        # Test
        # abbrv='RS'
        # name='Ross Sea'
        # mask_2d = mpas_south60S.mask_rs
        print(f"Masking MHW events for {name} ({abbrv})")
        # Expand dimensions
        mask = mask_2d
        if "days" in mhw_events_surface.dims:
            mask = mask.expand_dims(days=mhw_events_surface.days)
        
        # Mask MPA 
        biomass_mpa_actual = xr.open_dataset(os.path.join(path_biomass_surrogates, f"mpas/actual_biomass_{abbrv}.nc")).biomass_median
        biomass_valid_mask = ~np.all(np.isnan(biomass_mpa_actual), axis=1)  # shape (39, 231, 1440) -- all loc with timeseries valid
        
        masked_vars = {var: [] for var in mhw_vars}
        for yr in range(mhw_events_surface.sizes["years"]):
            # print(f"  Year {1980 + yr}")
            for var in mhw_vars:
                ds_yr = mhw_events_surface[var].isel(years=yr) #(181, 231, 1440)
                biomass_valid_mask_yr = biomass_valid_mask.isel(years=yr) #(231, 1440)
                
                mpa_biomass_masked = ds_yr.where(mask)

                # Mask on grid CEPHALOPOD, i.e. when biomass timeseries is Nan for the whole season, disregard MHWs for that location.
                mpa_biomass_masked_valid = mpa_biomass_masked.where(biomass_valid_mask_yr) #shape (181, 231, 1440)

                masked_vars[var].append(mpa_biomass_masked_valid)
    
        # Put together 
        mhw_event_masked_all = {var: xr.concat(masked_vars[var], dim=mhw_events_surface.years) for var in mhw_vars}
        
        # To Dataset
        mhw_event_masked_ds = xr.Dataset(mhw_event_masked_all,
                                        coords={"years": mhw_events_surface.years,
                                                "days": mhw_events_surface.days,
                                                "lon_rho": (("eta_rho", "xi_rho"), mhw_events_surface.lon_rho.data),
                                                "lat_rho": (("eta_rho", "xi_rho"), mhw_events_surface.lat_rho.data),},
                                        attrs={**mhw_events_surface.attrs,
                                               "mpa_name": f"{name} ({abbrv})",
                                               "masking": "All variables are masked to include only grid cells where biomass timeseries is not all NaN for the full season."})
       
        # Save to file
        mhw_event_masked_ds.to_netcdf(output_file) #shape (39, 181, 231, 1442)
    else:
        print(f'MHWs in {name} already saved to file')

# -- Load data
datasets = {}
for region in mpa_masks.keys():
    datasets[region] = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/duration_AND_thresh_{region}.nc'))


# Test
# test = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/duration_AND_thresh_RS.nc'))
# mean_duration_test = test['duration'].median(dim=['years', 'days'])
# print(mean_duration_test.max().values)
# print(mean_duration_test.min().values)

# %% ======================== Duration > 5days ========================
# According to Hobday et al. 2016, MHWs are defined as events lasting at least 5 days
# ---- For each MPA
def mask_mhw_5days(region):

    # Paths and files
    input_file = os.path.join(path_combined_thesh, f'mpas/duration_AND_thresh_{region}.nc')
    output_dir = os.path.join(path_combined_thesh, 'mpas', 'duration5days')
    output_file = os.path.join(output_dir, f'duration_AND_thresh_5days_{region}.nc')

    if os.path.exists(output_file):
        return f'{region}: already exists'

    ds = xr.open_dataset(input_file)

    mhw_5days = ds['duration'] >= 5

    for v in ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']:
        ds[v] = ds[v].where(mhw_5days)

    ds['duration'] = ds['duration'].where(mhw_5days)

    # Save to file
    ds.to_netcdf(output_file)

    ds.close()

    return f'{region}: saved'

# Run in parallel over the different MPAs
regions = list(mpa_masks.keys())
results = process_map(mask_mhw_5days, regions, max_workers=6, chunksize=1, desc='Duration ≥ 5 days')

# Put results together
datasets_5days = {}
for region in mpa_masks.keys():
    datasets_5days[region] = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/duration5days/duration_AND_thresh_5days_{region}.nc'))

# ---- For the whole Southern Ocean (south of 60°S)
# Save to file
output_file_SO_5days = os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON_5days.nc')
if not os.path.exists(output_file_SO_5days):    
    mhw_events_surface_5days = mhw_events_surface.copy()
    mhw_events_surface_5days = mhw_events_surface_5days.where(mhw_events_surface_5days['duration'] >= 5)
    mhw_events_surface_5days.to_netcdf(output_file_SO_5days)
else:
    # Load data
    mhw_events_surface_5days = xr.open_dataset(output_file_SO_5days)


# %% ======================== Median duration ========================
# Median duration (not mean to avoid extreme values/skewed distributions)
median_duration_mpas = {region: ds['duration'].median(dim=['years', 'days']) for region, ds in datasets_5days.items()}
median_duration_SO = mhw_events_surface_5days.duration.median(dim=['years', 'days'])


# %% ======================== Plot Median duration ========================
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

# Colormap for median duration
colors = ["#F4E9CD", "#9DBEBB", "#5F9796", "#468189", "#254D58", "#031926"]
# Bin depending on data range
values = median_duration_SO.values.flatten()
values = values[~np.isnan(values)]
bins = np.percentile(values, [0, 20, 40, 60, 80, 90, 100])

cmap = LinearSegmentedColormap.from_list("duration", colors)
norm = BoundaryNorm(boundaries=bins, ncolors=cmap.N)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

# Circular boundary for polar projection
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

# Base map
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

# Extent: Antarctic South of 60°S
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Plot median duration for the full SO
pcm = ax.pcolormesh(
    mhw_events_surface_5days['lon_rho'],
    mhw_events_surface_5days['lat_rho'],
    median_duration_SO,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    zorder=3
)

# Colorbar
cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Median MHW Duration (days)")
import matplotlib.patches as mpatches

for region, ds in datasets_5days.items():
    # Get min/max lon/lat of the MPA mask
    lon = ds['lon_rho'].values
    lat = ds['lat_rho'].values
    
    # Mask where there are no events (optional, to get bounds)
    mask = ~np.isnan(ds['duration'].values)
    if mask.any():
        lon_min, lon_max = lon[mask].min(), lon[mask].max()
        lat_min, lat_max = lat[mask].min(), lat[mask].max()
        
        # Draw rectangle
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=1.5,
            edgecolor='black',
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=4
        )
        ax.add_patch(rect)

ax.set_title("Median Marine Heatwave Duration (days) - Southern Ocean", fontsize=14)
plt.tight_layout()
plt.show()



# %%

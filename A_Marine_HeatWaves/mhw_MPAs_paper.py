"""
Created on Tues 20 Jan 10:49:30 2026

MHWs and their characteristics in the MPAs

@author: Marguerite Larriere (mlarriere)
"""

# %% ======================== PACKAGES ========================
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
# -- Load data
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)

# Fix extent - South of 60°S
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

# Settings plot
mpa_dict = {
    "Ross Sea": (mpas_ds.mask_rs, "#c77c27"),
    "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#e05c8a"),
    "East Antarctic": (mpas_ds.mask_ea, "#C00225"),
    "Weddell Sea": (mpas_ds.mask_ws, "#5f0f40"),
    "Antarctic Peninsula": (mpas_ds.mask_ap, "#867308")
}

# --- Areas and volume MPAs
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
tot_area_mpa = {}

for abbrv, (name, mask) in mpa_masks.items():
    area_mpa[abbrv] = area_60S_SO.where(mask)
    volume_mpa[abbrv] = volume_60S_SO_100m.where(mask)
    
    # Total area per MPAs
    tot_area_mpa[abbrv] = area_mpa[abbrv].sum(dim=['eta_rho', 'xi_rho'])

# Print results of total area
total_all_mpas = sum(tot_area_mpa.values())
print(f"{'Total across all MPAs':48s}: {total_all_mpas:>12,.0f} km²")
print("\n" + "=" * 80)
print(f"{'MPA':<5} {'Name':<45} {'Area (km²)':<15} {'% of Total':<10}")
print("=" * 80)
for abbrv, (name, mask) in mpa_masks.items():
    area_km2 = tot_area_mpa[abbrv].values
    percentage = (area_km2 / total_all_mpas) * 100
    print(f"{abbrv:<5} {name:<45} {area_km2:>12,.0f}   {percentage:>6.1f}%")
print("=" * 80)

# %% ======================== Median duration Southern Ocean ========================
# duration_mhw_90th = xr.open_dataset(os.path.join(path_duration, 'mhw_duration_90th_5m.nc')) #shape (40, 365, 434, 1442)
# file_median_duration_SO_fullyr = os.path.join(path_duration, f'median_duration_90th.nc') 

# if not os.path.exists(file_median_duration_SO_fullyr):
#     print('Southern Ocean')
#     duration_mhw_90th_SO = duration_mhw_90th.where(duration_mhw_90th['lat_rho'] <= -60, drop=True) #shape: (40, 365, 231, 1442)
    
#     # Save to file
#     encoding = {"duration": {"dtype": "float32"}}
#     duration_mhw_90th_SO.to_netcdf( os.path.join(path_duration, f'mhw_duration_90th_5m_SO.nc'), encoding=encoding)

#     print("Computing median MHW duration in the Southern Ocean...")
#     mhw_dur = duration_mhw_90th_SO['duration']
#     mhw_dur_mhw = mhw_dur.where(mhw_dur > 0) #shape (40, 365, 231, 1442)
#     median_duration = mhw_dur_mhw.median(dim=['years', 'days']) #min: 7days, max: 283days
#     # mean_duration = mhw_dur_mhw.mean(dim=['years', 'days']) #min: 9.59640103days, max: 126.48611111days

#     # Add attributes
#     median_duration.attrs = {"description": ("Median MHW duration in the Southern Ocean (south of 60°S), computed only over days with MHW occurrence."\
#                                                 "MHW events are defined using Hobday et al. (2016) definition (90th percentile only).")}
    
#     # To dataset
#     median_duration_ds = xr.Dataset({"duration": median_duration.astype("float32")},
#                                                 attrs=median_duration.attrs) 
    
#     # Save to file
#     encoding = {"duration": {"dtype": "float32"}}
#     median_duration_ds.to_netcdf(file_median_duration_SO_fullyr, encoding=encoding)


# else:
#     median_duration_90thperc = xr.open_dataset(file_median_duration_SO_fullyr)


# %% ======================== Median duration Absolute thresholds ========================
mhw_events_surface = xr.open_dataset(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc')) #shape (39, 181, 231, 1442)
mhw_events_surface = mhw_events_surface.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))  #shape (39, 181, 231, 1440)

mhw_dur = mhw_events_surface['duration']
# -- MHWs >= 90th percentile
mhw_dur_mhw = mhw_dur.where(mhw_dur > 0) 

# -- MHWs >= 90th percentile and 1°C
duration_1deg = mhw_dur_mhw.where(mhw_events_surface['det_1deg'] == 1) #shape (39, 181, 231, 1440)
# -- MHWs >= 90th percentile and 3°C
duration_3deg = mhw_dur_mhw.where(mhw_events_surface['det_3deg'] == 1) #shape (39, 181, 231, 1440)

# Calculate mean duration across time dimension
mean_duration_1deg = duration_1deg.mean(dim=['years', 'days']) #min=5days, max: 293days
mean_duration_3deg = duration_3deg.mean(dim=['years', 'days']) #min=7days, max: 386days

# %% ======================== Plot Paper ========================
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from skimage import measure
bins = [5, 15, 30, 60, 90, 120]

# Colormap 
colors = [
    "#F2E5C7",  # 5–15 d
    "#CFE1DA",  # 15–30 d
    "#86B6B3",  # 1–2 months
    "#1E5F74",  # 2–3 months
    "#0B1F2A",  # 3–4 months
]

cmap = LinearSegmentedColormap.from_list("duration", colors)
cmap.set_under("white")  # for MHWs < 5 days
norm = BoundaryNorm(boundaries=bins, ncolors=cmap.N)

# Figure settings
plot = 'report'  # slides report

# Define figure size based on output type
if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width * 0.5  # Two subplots side by side
else:
    fig_width = 14  # inches
    fig_height = 7

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 10}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 9}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}


# Create figure with two subplots
fig = plt.figure(figsize=(fig_width, fig_height))

# Data and titles for each subplot
data_list = [mean_duration_1deg, mean_duration_3deg]
titles = [r'MHWs $\ge$ 1$^\circ$C and 90th percentile', 
          r'MHWs $\ge$ 3$^\circ$C and 90th percentile']

axes = []  # collect axes
for i, (data, title) in enumerate(zip(data_list, titles)):
    ax = fig.add_subplot(1, 2, i+1, projection=ccrs.SouthPolarStereo())
    axes.append(ax)    

    # Circular boundary for polar projection
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    
    # Base map
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    
    # Features
    lw = 1 if plot == 'slides' else 0.5
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')
    
    # Plot median duration
    pcm = ax.pcolormesh(
        data.lon_rho, data.lat_rho, data,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=1, rasterized=True
    )
    
    # Gridlines
    lw_grid = 0.7 if plot == 'slides' else 0.3
    gl = ax.gridlines(
        draw_labels=True, color='gray', alpha=0.5,
        linestyle='--', linewidth=lw_grid, zorder=7
    )
    gl.xlabels_top = False
    gl.ylabels_right = False
    gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    # MPA boundaries
    lon = mpas_ds.lon_rho
    lat = mpas_ds.lat_rho
    for name, (mask, color) in mpa_dict.items():
        mask_2d = mask.values if hasattr(mask, "values") else mask
        lon_np  = lon.values
        lat_np  = lat.values

        contours = measure.find_contours(mask_2d.astype(float), 0.5)
        for contour in contours:
            eta_idx = contour[:, 0].astype(int)
            xi_idx  = contour[:, 1].astype(int)
            ax.plot(lon_np[eta_idx, xi_idx], lat_np[eta_idx, xi_idx],
                    color=color, linewidth=1,
                    transform=ccrs.PlateCarree(), zorder=2)
        
        # Centroid
        lon_masked_np = lon_np[mask_2d]
        lat_masked_np = lat_np[mask_2d]
        lon_centroid = lon_masked_np.mean()
        lat_centroid = lat_masked_np.mean()
        if ax==axes[0]:
            ax.text(
                lon_centroid, lat_centroid, name,
                transform=ccrs.PlateCarree(), zorder=5,
                fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor="white", edgecolor=color, linewidth=1.5, boxstyle="round,pad=0.3", alpha=0.9)
            )
    # Subplot title
    ax.set_title(title, **subtitle_kwargs, pad=10)

# Shared colorbar
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
# Shared colorbar — manually positioned
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
cbar_ax = fig.add_axes([0.1, -0.025, 0.8, 0.035])  # [left, bottom, width, height]
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Mean Duration", **label_kwargs)
cbar.set_ticks(bin_centers)
cbar.set_ticklabels([
    "5–15 d",
    "15–30 d",
    "1–2 months",
    "2–3 months",
    "3–4 months",
])
cbar.ax.tick_params(**tick_kwargs)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'A_Marine_HeatWaves/figures_outputs/MPAs/mhw_duration_1_3deg.pdf'), dpi=200, format='pdf', bbox_inches='tight')


# # %% ======================== Mask ========================
# # Note: all the MHW events have a duration > 5days.
# mhw_events_surface = xr.open_dataset(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc')) #shape (39, 181, 231, 1442)
# mhw_events_surface = mhw_events_surface.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))  #shape (39, 181, 231, 1440)
# duration_mhw_90th = duration_mhw_90th.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
# mhw_vars = ["duration", "det_1deg", "det_2deg", "det_3deg", "det_4deg"]


# for abbrv, (name, mask_2d) in mpa_masks.items():
#     output_file_abs = os.path.join(path_combined_thesh, f'mpas/interpolated/duration_AND_thresh_{abbrv}.nc')
#     output_file_90th = os.path.join(path_combined_thesh, f'mpas/interpolated/duration_90th_{abbrv}.nc')

#     if not os.path.exists(output_file_abs) and not os.path.exists(output_file_90th):
#         # Test
#         # abbrv='RS'
#         # name='Ross Sea'
#         # mask_2d = mpas_south60S.mask_rs
#         print(f"Masking MHW events for {name} ({abbrv})")
        
#         # Expand dimensions
#         # mask = mask_2d
#         mask_mpa = mask_2d.astype(bool)
#         # if "days" in mhw_events_surface.dims:
#         #     mask = mask.expand_dims(days=mhw_events_surface.days)

#         # Mask MPA 
#         biomass_mpa_actual = xr.open_dataset(os.path.join(path_biomass_surrogates, f"mpas/biomass_interpolated/actual_biomass_{abbrv}.nc")).biomass.isel(algo=0) #(39, 181, 231, 1440)
#         biomass_valid_mask = ~np.all(np.isnan(biomass_mpa_actual), axis=1)  # shape (39, 231, 1440) -- all loc with timeseries valid
        
#         # Initialize storage
#         masked_vars = {var: [] for var in mhw_vars}
#         duration_90th_masked_years = []

#         for yr in range(mhw_events_surface.sizes["years"]):
#             # yr=0
#             print(f"  Year {1980 + yr}")

#             # Yearly masks
#             biomass_mask_yr = biomass_valid_mask.isel(years=yr)  # (eta_rho, xi_rho)
#             mask_yr = mask_mpa  # (eta_rho, xi_rho)
            
#             # Biomass and MPA mask 
#             final_mask_yr = xr.broadcast(mask_yr & biomass_mask_yr, mhw_events_surface.isel(years=yr)[mhw_vars[0]])[0]

#             # --- MHWs events 90th percntile
#             duration_yr = duration_mhw_90th.duration.isel(years=yr)
#             duration_yr_masked = duration_yr.where(final_mask_yr)
#             duration_90th_masked_years.append(duration_yr_masked)
            
#             # --- MHWs events 90th percntile and absolute
#             for var in mhw_vars:
#                 # var=mhw_vars[1]
#                 ds_yr = mhw_events_surface[var].isel(years=yr)  # (days, eta_rho, xi_rho)

#                 # Apply mask
#                 ds_masked = ds_yr.where(final_mask_yr)

#                 masked_vars[var].append(ds_masked)

#         # Put together 
#         mhw_event_masked_all = {var: xr.concat(masked_vars[var], dim="years") for var in mhw_vars}
#         duration_90th_masked = xr.concat(duration_90th_masked_years, dim="years")

#         # To Dataset
#         mhw_event_masked_ds = xr.Dataset(mhw_event_masked_all,
#                                         coords={"years": mhw_events_surface.years,
#                                                 "days": mhw_events_surface.days,
#                                                 "lon_rho": (("eta_rho", "xi_rho"), mhw_events_surface.lon_rho.data),
#                                                 "lat_rho": (("eta_rho", "xi_rho"), mhw_events_surface.lat_rho.data),},
#                                         attrs={**mhw_events_surface.attrs,
#                                                "mpa_name": f"{name} ({abbrv})",
#                                                "masking": "All variables are masked to include only grid cells where biomass timeseries is not all NaN for the full season."})
        
#         duration_90th_masked_ds = duration_90th_masked.to_dataset(name="duration")
#         duration_90th_masked_ds.attrs = {"description": "Duration of MHWs defined as T°C exceeding the 90th percentile only.",
#                                          "mpa_name": f"{name} ({abbrv})",
#                                          "masking": "All variables are masked to include only grid cells where biomass timeseries (interpolated) is not all NaN for the full season."}
                
        
#         # Save to file
#         mhw_event_masked_ds.to_netcdf(output_file_abs) #shape (39, 181, 231, 1442)
#         duration_90th_masked_ds.to_netcdf(output_file_90th) #shape (39, 181, 231, 1442)
#     else:
#         print(f'MHWs in {name} already saved to file') 

# # -- Load data
# datasets_abs = {}
# datasets_rel = {}
# for region in mpa_masks.keys():
#     datasets_abs[region] = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/interpolated/duration_AND_thresh_{region}.nc'))
#     datasets_rel[region] = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/interpolated/duration_90th_{region}.nc'))

# median_duration_90thperc_seasonal = mhw_dur_mhw.median(dim=['years','days'])
# # %% ======================== Median duration MPAs ========================
# # Median duration (not mean to avoid extreme values/skewed distributions)
# threshold_vars = ["det_1deg", "det_2deg", "det_3deg", "det_4deg"]

# def compute_median_duration_region(region):
#     # test
#     # region = "AP"

#     print(f"Computing median durations for {region}")

#     output_file = os.path.join(path_combined_thesh, f"mpas/median_duration_{region}.nc")

#     if os.path.exists(output_file):
#         return region 
    
#     ds = datasets_abs[region]
#     dur = ds["duration"]

#     median_mpas = {}

#     # -- Overall median duration (any MHW)
#     dur_mhw = dur.where(dur > 0) #shape (39, 181, 231, 1440)
#     median_mpas["median_duration"] = dur_mhw.median(dim=["years", "days"]) #in AP -- min value: 5days, max value: 90days

#     # Add attributes
#     median_mpas["median_duration"].attrs = {"description": ("Median MHW duration, computed only over days with MHW occurrence."\
#                                                             "MHW events are defined using Hobday et al. (2016) definition (90th percentile and duration ≥5 days)."),
#                                                 "units": "days"}

#     # -- Median duration by absolute threshold
#     for thresh in threshold_vars:
#         print(f" Threshold: {thresh}")
#         # test
#         # thresh = "det_2deg"
        
#         dur_thresh = dur.where((ds[thresh] > 0) & (dur > 0)) #shape (39, 181, 231, 1440)
#         varname = f"median_duration_{thresh.replace('det_', '')}"

#         median_mpas[varname] = dur_thresh.median(dim=["years", "days"]) #in AP (2°C) -- min value: 7.5days , max value: 114days

#         # Add attributes
#         median_mpas[varname].attrs = {"description": (f"Median duration of MHWs exceeding {thresh.replace('det_', '').replace('deg','°C')}, computed only over days with MHW occurrence."),
#                                       "units": "days"}

#     # -- Combine to Dataset
#     median_duration_ds = xr.Dataset(median_mpas,
#                                     coords={"lon_rho": ds.lon_rho, "lat_rho": ds.lat_rho},
#                                     attrs={"mpa": region})

#     # Save to file
#     median_duration_ds.to_netcdf(output_file)

#     return region

# # Run in parallel
# regions = list(mpa_masks.keys())
# results = process_map(compute_median_duration_region, regions, max_workers=5, chunksize=1, desc='Median durations MPAs')

# # Load data
# median_duration_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/median_duration_{region}.nc")) for region in regions}






# # %% ======================== Plot Median duration ========================
# from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
# import matplotlib.patches as mpatches
# from skimage import measure

# # Bin 
# data = median_duration_90thperc_seasonal
# bins = [5, 15, 30, 60, 90, 120, 283]


# # Colormap 
# colors = [
#     "#F2E5C7",  # 5–15 d
#     "#CFE1DA",  # 15–30 d
#     "#86B6B3",  # 1–2 months
#     "#3F8F9C",  # 2–3 months
#     "#1E5F74",  # 3–4 months
#     "#0B1F2A",  # growth season
# ]

# cmap = LinearSegmentedColormap.from_list("duration", colors)
# norm = BoundaryNorm(bins, cmap.N)


# # Colormap settings
# cmap = LinearSegmentedColormap.from_list("duration", colors)
# cmap.set_under("white") #for MHWs < 5 days
# norm = BoundaryNorm(boundaries=bins, ncolors=cmap.N)

# # Figure settings
# plot='report' #slides report

# # Define figure size based on output type
# if plot == 'report':
#     fig_width = 6.3228348611
#     fig_height = fig_width 
    
# else:
#     fig_width = 7  # inches = \textwidth
#     fig_height = 7 

# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

# # Font size settings
# maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
# subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
# label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
# tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}

# # Circular boundary for polar projection
# theta = np.linspace(0, 2 * np.pi, 200)
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * 0.5 + 0.5)
# ax.set_boundary(circle, transform=ax.transAxes)

# # Base map
# ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# # Features
# lw = 1 if plot == 'slides' else 0.5
# ax.coastlines(color='black', linewidth=lw, zorder=5)
# ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
# ax.set_facecolor('lightgrey')

# # Plot median duration for the full SO
# # data = median_duration_90thperc.duration
# pcm = ax.pcolormesh(data.lon_rho, data.lat_rho, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1, rasterized=True)

# # Gridlines
# lw_grid = 0.7 if plot == 'slides' else 0.3
# gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=lw_grid, zorder=20)    
# gl.xlabels_top = False
# gl.ylabels_right = False
# gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
# gl.xlabel_style = gridlabel_kwargs
# gl.ylabel_style = gridlabel_kwargs
# gl.xformatter = LongitudeFormatter()
# gl.yformatter = LatitudeFormatter()

# # Colorbar
# # cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", fraction=0.04, pad=0.03, shrink=0.75)
# # cbar.set_label("Median MHW Duration (days)", **label_kwargs)
# # cbar.set_ticks(bins)
# # cbar.set_ticklabels([
# #     "5",
# #     "15",
# #     "30",
# #     "60",
# #     "90",
# #     "120",
# #     "$\ge$ 181"
# # ])

# # Bin centers for ticks
# bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# cbar = plt.colorbar(
#     pcm, ax=ax,
#     orientation="vertical",
#     fraction=0.04, pad=0.03, shrink=0.75
# )

# cbar.set_label("Median MHW Duration", **label_kwargs)

# cbar.set_ticks(bin_centers)
# cbar.set_ticklabels([
#     "5–15 d",
#     "15–30 d",
#     "1–2 months",
#     "2–3 months",
#     "3–4 months",
#     "Growth season"
# ])

# # MPAs boundaries
# lon = mpas_ds.lon_rho
# lat = mpas_ds.lat_rho
# for name, (mask, color) in mpa_dict.items():
#     mask_2d = mask.values if hasattr(mask, "values") else mask

#     # Convert lon/lat to numpy
#     lon_np = lon.values
#     lat_np = lat.values

#     # Contours
#     contours = measure.find_contours(mask_2d.astype(float), 0.5)
#     for contour in contours:
#         eta_idx = contour[:, 0].astype(int)
#         xi_idx  = contour[:, 1].astype(int)
#         lon_contour = lon_np[eta_idx, xi_idx]
#         lat_contour = lat_np[eta_idx, xi_idx]
#         ax.plot(lon_contour, lat_contour, color=color, linewidth=1.5, transform=ccrs.PlateCarree(), zorder=2)

#     # Centroid
#     lon_masked_np = lon_np[mask_2d]
#     lat_masked_np = lat_np[mask_2d]
#     lon_centroid = lon_masked_np.mean()
#     lat_centroid = lat_masked_np.mean()

#     ax.text(
#         lon_centroid, lat_centroid, name,
#         transform=ccrs.PlateCarree(), zorder=5,
#         fontsize=10, fontweight='bold', ha='center', va='center',
#         bbox=dict(facecolor="white", edgecolor=color, linewidth=1.5, boxstyle="round,pad=0.3", alpha=0.9)
#     )

# ax.set_title("90th percentile", **maintitle_kwargs)

# # --- Output handling ---
# plt.tight_layout()
# if plot == 'report':    
#     # plt.savefig(os.path.join(os.getcwd(), f'A_Marine_HeatWaves/figures_outputs/mhw_duration.pdf'), dpi=200, format='pdf', bbox_inches='tight')
#     plt.show()
# else:
#     plt.show()


# # %% ======================== MHW area affected ========================
# # For each MPA and each year, calculate the area affected by MHWs
# # --- Calculate area and volume of each MPA
# mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
#              "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
#              "EA": ("East Antarctic", mpas_south60S.mask_ea),
#              "WS": ("Weddell Sea", mpas_south60S.mask_ws),
#              "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}

# area_mpa = {}
# volume_mpa = {}

# for abbrv, (name, mask) in mpa_masks.items():
#     area_mpa[abbrv] = area_60S_SO.where(mask)
#     volume_mpa[abbrv] = volume_60S_SO_100m.where(mask)

# # ======= Absolute thresholds
# thresholds = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
# mhw_area_affected = {}

# for region, ds in datasets_abs.items():
#     # region = 'RS'
#     # ds = datasets_abs[region]

#     file_area = os.path.join(path_combined_thesh, f'mpas/mhw_daily_area_{region}.nc')

#     if not os.path.exists(file_area):
#         print(f"Calculating daily MHW area affected for {region}")
        
#         n_years = ds.sizes["years"]
#         n_days = ds.sizes["days"]
#         n_thresh = len(thresholds)
        
#         # Storage: years x thresholds x days
#         area_array = np.full((n_years, n_thresh, n_days), np.nan)
        
#         for i, thresh in enumerate(thresholds):
#             # i = 0
#             # thresh = thresholds[i]
#             deg = thresh.split("_")[1].replace("deg", "")
#             print(f"\nProcessing threshold: {deg}°C")
#             for yr in range(n_years):
#                 # yr = 0
#                 # print(f'    {yr+1980}')
#                 ds_yr = ds.isel(years=yr)
                
#                 # MHW for this threshold
#                 mhw_valid = ds_yr['duration']>0
#                 mhw_mask = mhw_valid & (ds_yr[thresh] == 1)  # True where MHW occurs

#                 # area % 
#                 total_area_mpa = area_mpa[region].sum(dim=["eta_rho", "xi_rho"]) #km2
                
#                 # Daily area affected
#                 daily_area = area_mpa[region].where(mhw_mask).sum(dim=["eta_rho", "xi_rho"])
                
#                 daily_area_perc = (daily_area / total_area_mpa) * 100

#                 # Store data
#                 area_array[yr, i, :] = daily_area_perc.values
        
#         # To DataArray
#         da = xr.DataArray(area_array, dims=["years", "threshold", "days"],
#                         coords={"years": ds.years, "threshold": thresholds, "days": ds.days},
#                         name="mhw_area_affected")

#         # Add attributes (metadata)
#         da.attrs["units"] = "% of MPA area"
#         da.attrs["description"] = (
#             "Daily area of each region affected by MHWs ≥5 days, "
#             "for each threshold."
#         )
        
#         mhw_area_affected[region] = da
        
#         # Save to file
#         # file_area = os.path.join(path_combined_thesh, f'mpas/mhw_daily_area_{region}.nc')
#         da.to_netcdf(file_area)
#         print(f"Saved daily area affected for {region} to {file_area}")
#     else:
#         print(f"MHWs area in {region} already saved to file")

#         # load data
#         mhw_area_affected_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/mhw_daily_area_{region}.nc")) for region in regions}

# # %% Maximum area affacted each year
# mhw_area_affected_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/mhw_daily_area_{region}.nc")) for region in regions}
# # ======= Absolute thresholds
# thresholds = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
# max_area_affected = {}

# for region, mhw_mpa in mhw_area_affected_mpas.items():
#     # region='AP'
#     # mhw_mpa=mhw_area_affected_mpas[region]
#     max_annual = mhw_mpa.max(dim="days")
#     max_area_affected[region] = max_annual
    
# # %% ======================== Plot max MHW area timeseries ========================
# from datetime import datetime, timedelta
# from matplotlib.patches import Patch
# plot='report'
# # Font size settings
# maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
# subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
# label_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
# tick_kwargs = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}

# # ------------------------
# # Choose region
# # ------------------------
# region = "WS"   # RS, SO, EA, WS, AP

# # ------------------------
# # Region: full MPA name
# # ------------------------
# region_name_map = {
#     "RS": "Ross Sea",
#     "SO": "South Orkney Islands southern shelf",
#     "EA": "East Antarctic",
#     "WS": "Weddell Sea",
#     "AP": "Antarctic Peninsula",
# }

# mpa_name = region_name_map[region]
# _, mpa_color = mpa_dict[mpa_name]

# # ------------------------
# # Data
# # ------------------------
# da = max_area_affected[region].mhw_area_affected

# # Mean over years → daily climatology
# # da_mean = da.mean(dim="years")

# # ------------------------
# # Time axis
# # ------------------------
# year_indices = da.years.values  # 0 to 39
# actual_years = year_indices + 1980  # 1980 to 2019

# # days_xaxis = np.arange(181)
# # base_date = datetime(2021, 11, 1)
# # date_dict = {
# #     int(i): (base_date + timedelta(days=int(i))).strftime("%b %d")
# #     for i in days_xaxis
# # }
# # tick_positions = np.arange(0, 181, 15)
# # tick_labels = [date_dict[d] for d in tick_positions]

# # ------------------------
# # Thresholds to plot
# # ------------------------
# thresholds_to_plot = {
#     "det_1deg": (r"$\ge$ 1$^\circ$C and 90th perc", "#5A7854"),
#     "det_3deg": (r"$\ge$ 3$^\circ$C and 90th perc", "#E07800"),
# }

# # ------------------------
# # Figure
# # ------------------------
# fig, ax = plt.subplots(figsize=(7, 4))

# for thresh, (label, color) in thresholds_to_plot.items():
#     y = da.sel(threshold=thresh).values
#     ax.plot(actual_years, y, color=color, lw=2)
#     ax.fill_between(actual_years, 0, y, color=color)

# # ------------------------
# # Axes styling
# # ------------------------
# ax.set_xlim(1980, 2018)
# ax.set_ylim(0, None)

# ax.set_ylabel("\% MPA Area", **label_kwargs)
# ax.set_xlabel("Years", **label_kwargs)

# # ax.set_xticks(tick_positions)
# # ax.set_xticklabels(tick_labels, rotation=45, ha="right")
# ax.xaxis.set_major_locator(plt.MultipleLocator(5))  # Every 5 years


# # ------------------------
# # Color plot box by MPA
# # ------------------------
# for spine in ax.spines.values():
#     spine.set_edgecolor(mpa_color)
#     spine.set_linewidth(1)

# ax.tick_params(axis="both", colors=mpa_color)
# ax.xaxis.label.set_color(mpa_color)
# ax.yaxis.label.set_color(mpa_color)

# # ------------------------
# # MPA label inside plot
# # ------------------------
# ax.text(
#     0.025, 0.95, mpa_name,
#     transform=ax.transAxes,
#     ha="left", va="top",
#     fontweight="bold",
#     color=mpa_color,
#     bbox=dict(
#         facecolor="white",
#         edgecolor=mpa_color,
#         lw=1,
#         boxstyle="round,pad=0.3"
#     ), **subtitle_kwargs)

# # ------------------------
# # Legend
# # ------------------------
# handles = [
#     Patch(facecolor=color, edgecolor="black", lw=0.5, label=label)
#     for label, color in [(v[0], v[1]) for v in thresholds_to_plot.values()]
# ]

# fig.legend(
#     handles=handles,
#     loc="upper center",
#     ncol=2,
#     frameon=True,
#     bbox_to_anchor=(0.52, 0.85), **label_kwargs)

# plt.tight_layout()
# # plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'A_Marine_HeatWaves/figures_outputs/MPAs/mhw_max_area_timeseries_{region}.pdf'), dpi=200, format='pdf', bbox_inches='tight')


# # %%

# %%

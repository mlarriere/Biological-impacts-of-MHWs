"""
Created on Thurs 08 March 09:02:03 2025

Attribution of MHW to change in biomass in the Southern Ocean

@author: Marguerite Larriere (mlarriere)
"""

# %% ======================== PACKAGES========================
import os

from attr import attrib
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
from skimage import measure
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

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
path_biomass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_biomass_ts = os.path.join(path_surrogates, f'biomass_timeseries')
path_biomass_ts_SO = os.path.join(path_biomass_ts, f'SouthernOcean')
path_biomass_ts_MPAs = os.path.join(path_biomass_ts, f'mpas')
path_masslength = os.path.join(path_surrogates, f'mass_length')
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')

# %% ======================== Load Biomass data ========================
surrogate_names = {"clim": "Climatology", "actual": "Actual Conditions", "clim_trend":"Climatology wih trend", "nowarming": "No Warming"}
files_interp = [os.path.join(path_biomass_ts_SO, f"biomass_{sur}.nc") for sur in surrogate_names.keys()]
biomass = {}

for surrog in surrogate_names.keys():
    fname = os.path.join(path_biomass_ts_SO, f"biomass_{surrog}.nc")
    biomass[surrog] = xr.open_dataset(fname) #shape (39, 10, 181, 231, 1442)

# %% ========================================================================
#           Attribution of seasonal biomass gain to MHWs and warming
#    ========================================================================
 # %% ======================== Step1. Seasonal gains for the different surrogates ========================
seasonal_gain_interp = {}
for surrog in surrogate_names.keys():
    seasonal_gain_interp[surrog] = biomass[surrog].isel(days=-1) - biomass[surrog].isel(days=0) #shape (39, 10, 231, 1442)

# %% ======================== Step2. Change relative to climatology per grid cell (fraction) ========================
change_biomass_cell = {}
for surrog in [s for s in surrogate_names if s != "clim"]:
    change_biomass_cell[surrog] = (seasonal_gain_interp[surrog] - seasonal_gain_interp['clim'])/seasonal_gain_interp['clim'] #shape (39, 10, 231, 1442)

# %% ======================== Step3. Impact of the warming and MHWs per grid cell and spatial average (Step4)
impact_mhws = change_biomass_cell['actual'] - change_biomass_cell['clim_trend']  # shape (39, 10, 231, 1440)
impact_warming = change_biomass_cell['actual'] - change_biomass_cell['nowarming']  # shape (39, 10, 231, 1440)
residual = change_biomass_cell['actual'] - impact_mhws - impact_warming #residuals are mainly the chla 'contribution' according to the Atkinson model

# Quick check 
# check_sum = impact_mhws.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200) + impact_warming.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200) + residual.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200)
# print((check_sum-change_biomass_cell['actual'].isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200)).values) #should be close to 0

# To common dataset
impact_ds = xr.Dataset({
    "attrib_mhws": impact_mhws["biomass"],
    "attrib_warming": impact_warming["biomass"],
    "residual": residual["biomass"],
})

# Save to file
fpath = os.path.join(path_surrogates, 'attributions/attributions_SO.nc')
impact_ds.to_netcdf(fpath)

# %% ======================== Attributions under MHWs ========================
mhw_events_surface = xr.open_dataset(os.path.join(path_combined_thesh, f"duration_AND_thresh_5mSEASON.nc"))

# --- Base mask: 90th percentile and 5days duration (Hobday)
duration_mask = mhw_events_surface['duration'] > 0  # (39, 181, 231, 1442)

# --- Detection mask
det = {i: mhw_events_surface[f'det_{i}deg'] == 1 for i in [1, 2, 3, 4]}

# --- 4 mutually exclusive intensity bins ---
# Select cells that have experienced MHWs at least once in the season
intensity_masks = {'90perc': duration_mask.any(dim="days"),
                    '1deg': (duration_mask & det[1]).any(dim="days"),
                    '2deg': (duration_mask & det[2]).any(dim="days"),
                    '3deg': (duration_mask & det[3]).any(dim="days"),
                    '4deg': (duration_mask & det[4]).any(dim="days"),
                    }

def compute_mask(args):
    suffix, mask = args
    out = {}

    # attribution variables
    for varname in ["attrib_mhws", "attrib_warming", "residual"]:
        out[f"{varname}_{suffix}"] = impact_ds[varname].where(mask)

    # biomass variable
    out[f"biomass_{suffix}"] = change_biomass_cell["actual"]["biomass"].where(mask)

    return out

attribution_path = os.path.join(path_surrogates, 'attributions')
fpath_bio = os.path.join(path_surrogates, f'biomass_change_wrt_clim_mhws_cells.nc')
fpath_attrib_mhws = os.path.join(attribution_path, f'attrib_mhws_mhws_cells.nc')
fpath_attrib_warming = os.path.join(attribution_path, f'attrib_warming_mhws_cells.nc')
fpath_residuals = os.path.join(attribution_path, f'residuals_attrib_mhws_cells.nc')

if not os.path.exists(fpath_bio) and not os.path.exists(fpath_attrib_mhws) and not os.path.exists(fpath_attrib_warming) and not os.path.exists(fpath_residuals):
    print('Processing...')
    # Call function
    args = list(intensity_masks.items())

    results = process_map(compute_mask, args, max_workers=5, chunksize=1, desc="Select only MHWs cells")

    combined_dict = {}
    for r in results:
        combined_dict.update(r)

    attrib_dict = {k: v for k, v in combined_dict.items() if not k.startswith("biomass_")}
    mhws_dict = {k: v for k, v in attrib_dict.items() if k.startswith("attrib_mhws_")}
    warming_dict = {k: v for k, v in attrib_dict.items() if k.startswith("attrib_warming_")}
    res_dict = {k: v for k, v in attrib_dict.items() if k.startswith("residual_")}
    biomass_dict = {k: v for k, v in combined_dict.items() if k.startswith("biomass_")}

    # To Dataset (1 for each attribution and one for biomass)
    mhws_ds = xr.Dataset(
    data_vars=mhws_dict,
    coords={
        "years": impact_ds["attrib_mhws"].years,
        "bootstraps": impact_ds["attrib_mhws"].bootstraps,
        "eta_rho": impact_ds["attrib_mhws"].eta_rho,
        "xi_rho": impact_ds["attrib_mhws"].xi_rho,
    }, 
    attrs={"description": "Contribution of MHWs to change in biomass -- cells selected have experienced MHWs of ideg intensity"}
    )
    
    warming_ds = xr.Dataset(
    data_vars=warming_dict,
    coords={
        "years": impact_ds["attrib_warming"].years,
        "bootstraps": impact_ds["attrib_warming"].bootstraps,
        "eta_rho": impact_ds["attrib_warming"].eta_rho,
        "xi_rho": impact_ds["attrib_warming"].xi_rho,
    }, 
    attrs={"description": "Contribution of long term temp to change in biomass -- cells selected have experienced MHWs of ideg intensity"}
    )

    residual_ds = xr.Dataset(
    data_vars=res_dict,
    coords={
        "years": impact_ds["residual"].years,
        "bootstraps": impact_ds["residual"].bootstraps,
        "eta_rho": impact_ds["residual"].eta_rho,
        "xi_rho": impact_ds["residual"].xi_rho,
    }, 
    attrs={"description": "Residuals (mainly chla contribution) to change in biomass -- cells selected have experienced MHWs of ideg intensity"}
    )

    bio_ds = xr.Dataset(
    data_vars=biomass_dict,
    coords={
        "years": change_biomass_cell["actual"]["biomass"].years,
        "bootstraps": change_biomass_cell["actual"]["biomass"].bootstraps,
        "eta_rho": change_biomass_cell["actual"]["biomass"].eta_rho,
        "xi_rho": change_biomass_cell["actual"]["biomass"].xi_rho,
    },
    attrs={"description": "Seasonal biomass change wrt clim masked by MHW intensity"}
    )
    
    # Save to file
    mhws_ds.to_netcdf(fpath_attrib_mhws)
    warming_ds.to_netcdf(fpath_attrib_warming)
    residual_ds.to_netcdf(fpath_residuals)
    bio_ds.to_netcdf(fpath_bio)

else:
    # --- Load files
    print('Load files...')
    mhws_impact_mhws_cell = xr.open_dataset(fpath_attrib_mhws)
    warming_impact_mhws_cells = xr.open_dataset(fpath_attrib_warming)
    residuals_impact_mhw_cells = xr.open_dataset(fpath_residuals)
    bio_change_mhw_cells = xr.open_dataset(fpath_bio)


# %% ======================== Statistics ========================
# -- Mean over years (10, 231, 1442)
mhws_attrib_mean_yrs = mhws_impact_mhws_cell.mean(dim="years", skipna=True)
warming_attrib_mean_yrs = warming_impact_mhws_cells.mean(dim="years", skipna=True)
residuals_attrib_mean_yrs = residuals_impact_mhw_cells.mean(dim="years", skipna=True)
bio_change_mean_yrs = bio_change_mhw_cells.mean(dim="years", skipna=True)


# -- Median and std over bootstraps (231, 1442)
mhws_attrib_median_boot = mhws_attrib_mean_yrs.median(dim="bootstraps", skipna=True)
warming_attrib_median_boot = warming_attrib_mean_yrs.median(dim="bootstraps", skipna=True)
residuals_attrib_median_boot = residuals_attrib_mean_yrs.median(dim="bootstraps", skipna=True)
bio_change_median_boot = bio_change_mean_yrs.median(dim="bootstraps", skipna=True)
    
mhws_attrib_std_boot = mhws_attrib_mean_yrs.std(dim="bootstraps", skipna=True)
warming_attrib_std_boot = warming_attrib_mean_yrs.std(dim="bootstraps", skipna=True)
residuals_attrib_std_boot = residuals_attrib_mean_yrs.std(dim="bootstraps", skipna=True)
bio_change_std_boot = bio_change_mean_yrs.std(dim="bootstraps", skipna=True)
    

# %% ======================== MPAs data ========================
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)
mpa_dict = {"Weddell Sea": (mpas_ds.mask_ws, "#5f0f40"),
            "East Antarctic": (mpas_ds.mask_ea, "#C00225"),
            "Ross Sea": (mpas_ds.mask_rs, "#c77c27"),
            "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#e05c8a"),
            "Antarctic Peninsula": (mpas_ds.mask_ap, "#867308")}

mpa_masks = {"WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}


# %% ======================== Maps of attribution ========================
mhw_key='1deg'

# Font size settings
plot='report'
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 10}
subtitle_kwargs  = {'fontsize': 15} if plot == 'slides' else {'fontsize': 9}
label_kwargs     = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
tick_kwargs      = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}
lw = 1   if plot == 'slides' else 0.5
lw_grid= 0.7 if plot == 'slides' else 0.3

# == Diverging colormap for biomass change
cmap_bio = LinearSegmentedColormap.from_list('purple_white_teal',
               ["#AEA8DE", "white", "#94D2BD"])
norm_bio = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

# == MPA boundaries setup
lon_np = mpas_ds.lon_rho.values
lat_np = mpas_ds.lat_rho.values
lon_np_norm = np.where(lon_np > 180, lon_np - 360, lon_np)


fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(nrows=1, ncols=4, figure=fig, wspace=0.05)
titles = [
    'Biomass change',
    'MHW attribution',
    'Long-term warming attribution',
    'Residuals'
]
for col in range(4):
    ax = fig.add_subplot(gs[col], projection=ccrs.SouthPolarStereo())

    # Circular boundary
    theta  = np.linspace(0, 2 * np.pi, 200)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Features
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    if col == 0:
        sc = ax.pcolormesh(
            bio_change_median_boot.lon_rho,
            bio_change_median_boot.lat_rho,
            bio_change_median_boot[f'biomass_{mhw_key}'],
            transform=ccrs.PlateCarree(),
            cmap=cmap_bio,
            norm=norm_bio,
            rasterized=True,
            zorder=1,
        )

    elif col == 1:
        sc = ax.pcolormesh(
            mhws_attrib_median_boot.lon_rho,
            mhws_attrib_median_boot.lat_rho,
            mhws_attrib_median_boot[f'attrib_mhws_{mhw_key}'],
            transform=ccrs.PlateCarree(),
            cmap='PuOr',
            vmin=-1,
            vmax=1,
            rasterized=True,
            zorder=1,
        )

    elif col == 2:
        sc = ax.pcolormesh(
            warming_attrib_median_boot.lon_rho,
            warming_attrib_median_boot.lat_rho,
            warming_attrib_median_boot[f'attrib_warming_{mhw_key}'],
            transform=ccrs.PlateCarree(),
            cmap='RdBu',
            vmin=-10,
            vmax=10,
            rasterized=True,
            zorder=1,
        )

    else:
        sc = ax.pcolormesh(
            residuals_attrib_median_boot.lon_rho,
            residuals_attrib_median_boot.lat_rho,
            residuals_attrib_median_boot[f'residual_{mhw_key}'],
            transform=ccrs.PlateCarree(),
            cmap='PiYG',
            vmin=-10,
            vmax=10,
            rasterized=True,
            zorder=1,
        )
    cbar = fig.colorbar(
    sc,
    ax=ax,
    orientation='horizontal',
    pad=0.05,
    shrink=0.8,
    )

    cbar.ax.tick_params(labelsize=8)
    ax.set_title(titles[col], fontsize=11, fontweight='bold')
    
    # == MPA boundaries
    for name, (mask, color) in mpa_dict.items():
        mask_2d = mask.values if hasattr(mask, "values") else mask
        contours = measure.find_contours(mask_2d.astype(float), 0.5)
        for contour in contours:
            eta_idx = np.clip(contour[:, 0].astype(int), 0, lon_np.shape[0] - 1)
            xi_idx  = np.clip(contour[:, 1].astype(int), 0, lon_np.shape[1] - 1)
            lon_contour = lon_np_norm[eta_idx, xi_idx]
            lat_contour = lat_np[eta_idx, xi_idx]
            breaks       = np.where(np.abs(np.diff(lon_contour)) > 180)[0] + 1
            lon_segments = np.split(lon_contour, breaks)
            lat_segments = np.split(lat_contour, breaks)
            for lon_seg, lat_seg in zip(lon_segments, lat_segments):
                if len(lon_seg) > 1:
                    ax.plot(lon_seg, lat_seg, color=color, linewidth=1.5,
                            transform=ccrs.PlateCarree(), zorder=4)
                    
    # == Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--',
                      linewidth=0.7, zorder=5)
    gl.xlabels_top   = False
    gl.ylabels_right = False
    gl.xlabel_style  = {'size': 7, 'rotation': 0}
    gl.ylabel_style  = {'size': 7, 'rotation': 0}
    gl.xformatter    = LongitudeFormatter()
    gl.yformatter    = LatitudeFormatter()
    gl.ylocator      = mticker.FixedLocator([-80, -75, -70, -65, -60])
    gl.xlocator      = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])


# == MPA legend
mpa_legend = [Line2D([0], [0], color=color, lw=2, label=name)
              for name, (_, color) in mpa_dict.items()]
fig.legend(handles=mpa_legend, loc='lower center', fontsize=7,
           framealpha=0.8, bbox_to_anchor=(0.5, -0.026), ncol=5)

fig.suptitle(f'Attribution of changes in seasonal biomass under MHWs of {mhw_key[0]}°C', fontsize=14, fontweight='bold', y=0.94)

plt.tight_layout()
plt.show()
# == Save to file
# outdir  = os.path.join(os.getcwd(), 'C_Biomass/figures_outputs')
# os.makedirs(outdir, exist_ok=True)
# outfile = 'seasonal_gain_change_attribution_1980_2000_2018.png'
# # plt.savefig(os.path.join(outdir, outfile), dpi=500, format='png', bbox_inches='tight')

# # %% ======================== MHWs Metrics ========================
# ds_mhw = xr.open_dataset(os.path.join(path_combined_thesh, f"duration_AND_thresh_5mSEASON.nc"))
# # 1. MHW duration
# mhw_duration = ds_mhw.duration 

# # 2. MHW intensity
# mhw_1deg = ds_mhw.det_1deg 
# mhw_2deg = ds_mhw.det_2deg 
# mhw_3deg = ds_mhw.det_3deg 
# mhw_4deg = ds_mhw.det_4deg 

# # 3. MHW area
# volume_60S_SO_100m
# area_60S_SO

# # ======= Absolute thresholds
# thresholds = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
# n_years = ds_mhw.sizes["years"]
# n_days = ds_mhw.sizes["days"]
# n_thresh = len(thresholds)
        
# # Storage: years x thresholds x days
# area_array = np.full((n_years, n_thresh, n_days), np.nan)
        
# for i, thresh in enumerate(thresholds):
#     # i = 0
#     # thresh = thresholds[i]
#     deg = thresh.split("_")[1].replace("deg", "")
#     print(f"\nProcessing threshold: {deg}°C")
#     for yr in range(n_years):
#         # yr = 0
#         # print(f'    {yr+1980}')
#         ds_yr = ds_mhw.isel(years=yr)

#         # MHW for this threshold
#         mhw_valid = ds_yr['duration']>0
#         mhw_mask = mhw_valid & (ds_yr[thresh] == 1)  # True where MHW occurs

#         # area % 
#         total_area_SO = area_60S_SO.sum(dim=["eta_rho", "xi_rho"]) #km2
        
#         # Daily area affected
#         daily_area = area_60S_SO.where(mhw_mask).sum(dim=["eta_rho", "xi_rho"])
        
#         daily_area_perc = (daily_area / total_area_SO) * 100

#         # Store data
#         area_array[yr, i, :] = daily_area_perc.values
        
# # To DataArray
# mhw_area_affected = xr.DataArray(area_array, dims=["years", "threshold", "days"],
#                 coords={"years": ds_mhw.years, "threshold": thresholds, "days": ds_mhw.days},
#                 name="mhw_area_affected")

# # Add attributes (metadata)
# mhw_area_affected.attrs["units"] = "% of Southern Ocean area"
# mhw_area_affected.attrs["description"] = ("Daily area of each region affected by MHWs ≥5 days, "
#                                           "for each threshold.")
        
    
# # %% ======================== Plot the seasonal gain VS MHW metrics ========================
# # -- Parameters
# legend_names = {"clim": "Climatology", "actual": "Actual", "climtrend": "No MHWs", "nowarming": "No warming"}
# colors = {"actual": "#648028", "climtrend": "#F18701", "nowarming": "#584CBD"}
# surrogates = ["actual", "climtrend", "nowarming"]

# # -- Select data
# mhw_dur = mhw_duration.max(dim='days').where(mhw_duration.max(dim='days')>0).mean(dim=['eta_rho', 'xi_rho'])
# actual_gain = total_seasonal_gain['actual'].median(dim=['algo']).biomass * 1e-12

# # -- Color settings
# import matplotlib.cm as cm
# years = np.arange(1980, 2019)
# cmap = cm.RdYlBu_r
# norm = plt.Normalize(vmin=years.min(), vmax=years.max())
# colors_yr = cmap(norm(years))

# fig, ax = plt.subplots(figsize=(8, 6))

# sc = ax.scatter(mhw_dur, actual_gain, c=years, cmap=cmap, norm=norm, s=60, zorder=3, edgecolors='k', linewidths=0.4)

# # -- Annotate dots
# # for i, yr in enumerate(years):
# #     ax.annotate(str(yr), 
# #                 (float(mhw_dur.isel(years=i)), float(actual_gain.isel(years=i))),
# #                 textcoords="offset points", xytext=(6, 4), fontsize=7, alpha=0.7)
# # -- Trend line
# # x = mhw_dur.values.flatten()
# # y = actual_gain.values.flatten()
# # mask = np.isfinite(x) & np.isfinite(y)
# # z = np.polyfit(x[mask], y[mask], 1)
# # p = np.poly1d(z)
# # x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
# # ax.plot(x_line, p(x_line), 'k--', linewidth=1.2, alpha=0.6, label=f'Trend (slope={z[0]:.2e})')

# # -- Colorbar
# cbar = fig.colorbar(sc, ax=ax, pad=0.02)
# cbar.set_label('Year', fontsize=11)

# # -- Labels
# ax.set_xlabel('Mean MHW Duration [days]', fontsize=12)
# ax.set_ylabel('Total Seasonal Biomass Gain [Mt]', fontsize=12)
# ax.set_title(f'Seasonal Biomass Gain vs MHW Duration\nSouthern Ocean', fontsize=13)
# # ax.legend(fontsize=10)
# # ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # %%
# # Mean duration of MHW events exceeding 1°C and 3°C
# mhw_dur_1deg = mhw_duration.where(mhw_1deg == 1).max(dim='days').where(lambda x: x > 0).mean(dim=['eta_rho', 'xi_rho'])
# mhw_dur_3deg = mhw_duration.where(mhw_3deg == 1).max(dim='days').where(lambda x: x > 0).mean(dim=['eta_rho', 'xi_rho'])

# # -- Plot
# fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# for ax, mhw_dur, label in zip(axes,
#                                [mhw_dur_1deg, mhw_dur_3deg],
#                                [r'MHW $\geq$ 90th percentile and 1°C', r'MHW $\geq$ 90th percentile and 3°C']):
#     sc = ax.scatter(mhw_dur, actual_gain, c=years, cmap=cmap, norm=norm,
#                     s=60, zorder=3, edgecolors='k', linewidths=0.4)
#     ax.set_xlabel(f'Mean Duration [days]', fontsize=11)
#     ax.set_title(label, fontsize=12)
#     # ax.grid(True, alpha=0.3)

# axes[0].set_ylabel('Total Seasonal Biomass Gain [Mt]', fontsize=11)

# # Colorbar on the far right only
# cbar = fig.colorbar(sc, ax=axes[-1], pad=0.05, fraction=0.046)
# cbar.set_label('Year', fontsize=11)

# fig.suptitle('Seasonal Biomass Gain vs MHW Area\nSouthern Ocean', fontsize=13)
# plt.tight_layout()
# plt.show()



# # %%
# # Max area
# mhw_area_1deg = mhw_area_affected.isel(threshold=0).max(dim='days')
# mhw_area_3deg =  mhw_area_affected.isel(threshold=2).max(dim='days')

# # -- Color settings
# import matplotlib.cm as cm
# years = np.arange(1980, 2019)
# cmap = cm.RdYlBu_r
# norm = plt.Normalize(vmin=years.min(), vmax=years.max())
# colors_yr = cmap(norm(years))

# # -- Plot
# fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# for ax, mhw_dur, label in zip(axes,
#                                [mhw_area_1deg, mhw_area_3deg],
#                                [r'MHW $\geq$ 90th percentile and 1°C', r'MHW $\geq$ 90th percentile and 3°C']):
#     sc = ax.scatter(mhw_dur, actual_gain, c=years, cmap=cmap, norm=norm,
#                     s=60, zorder=3, edgecolors='k', linewidths=0.4)
#     ax.set_xlabel(f'Maximum Area Affected [\%]', fontsize=11)
#     ax.set_title(label, fontsize=12)
#     # ax.grid(True, alpha=0.3)

# axes[0].set_ylabel('Total Seasonal Biomass Gain [Mt]', fontsize=11)

# # Colorbar on the far right only
# cbar = fig.colorbar(sc, ax=axes[-1], pad=0.05, fraction=0.046)
# cbar.set_label('Year', fontsize=11)

# fig.suptitle('Seasonal Biomass Gain vs MHW Area\nSouthern Ocean', fontsize=13)
# plt.tight_layout()
# plt.show()



# # %%

# %%

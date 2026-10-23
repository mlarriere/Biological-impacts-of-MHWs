"""
Created on Thurs 08 March 09:02:03 2025

Attribution of MHW to change in biomass in the Southern Ocean

@author: Marguerite Larriere (mlarriere)
"""

# %% ======================== PACKAGES========================
import os

from attr import attrib
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# %% ======================== MPAs data ========================
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

mpa_dict = {"Weddell Sea": (mpas_south60S.mask_ws, "#5f0f40"),
            "East Antarctic": (mpas_south60S.mask_ea, "#C00225"),
            "Ross Sea": (mpas_south60S.mask_rs, "#c77c27"),
            "South Orkney Islands southern shelf":  (mpas_south60S.mask_o,  "#e05c8a"),
            "Antarctic Peninsula": (mpas_south60S.mask_ap, "#867308")}

mpa_masks = {"WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}

# %% ======================== Attribution of seasonal biomass changes to MHWs and long-term thermal trend ========================
# -- Load functions and settings 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
import functions_attribution
importlib.reload(functions_attribution)
from functions_attribution import compute_attributions_SO, mask_so_attributions_by_mhw_intensity
attribution_path = os.path.join(path_surrogates, "attributions")

# Seasonal biomass changes, and attributions calculations
(seasonal_gain_SO, change_biomass_SO, impact_ds) = compute_attributions_SO(biomass=biomass, surrogate_names=surrogate_names, output_dir=attribution_path)

# Masking only for MHWs
(mhws_impact_mhws_cell, warming_impact_mhws_cells, residuals_impact_mhw_cells, bio_change_mhw_cells) = mask_so_attributions_by_mhw_intensity(path_combined_thesh=path_combined_thesh,
                                                                                                                                           output_dir=attribution_path)


# %% ======================== Statistics ========================
# -- Mean over years (10, 231, 1442)
mhws_attrib_mean_yrs = mhws_impact_mhws_cell.mean(dim="years", skipna=True)
warming_attrib_mean_yrs = warming_impact_mhws_cells.mean(dim="years", skipna=True)
residuals_attrib_mean_yrs = residuals_impact_mhw_cells.mean(dim="years", skipna=True)
bio_change_mean_yrs = bio_change_mhw_cells.mean(dim="years", skipna=True)

# -- Median and std over bootstraps (231, 1442)
mhws_attrib_median_boot = mhws_attrib_mean_yrs.median(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
warming_attrib_median_boot = warming_attrib_mean_yrs.median(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
residuals_attrib_median_boot = residuals_attrib_mean_yrs.median(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
bio_change_median_boot = bio_change_mean_yrs.median(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
    
mhws_attrib_std_boot = mhws_attrib_mean_yrs.std(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
warming_attrib_std_boot = warming_attrib_mean_yrs.std(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
residuals_attrib_std_boot = residuals_attrib_mean_yrs.std(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
bio_change_std_boot = bio_change_mean_yrs.std(dim="bootstraps", skipna=True).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
    

# %% ======================== Maps of attribution ========================
mhw_key='1deg'
stat = 'median'  # 'median' or 'std' 

# --- Colormap and norm ---
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
if stat=='median':
    quantile95_biomass = (bio_change_median_boot[f'biomass_{mhw_key}']*100).quantile(0.95).values
    quantile5_biomass = (bio_change_median_boot[f'biomass_{mhw_key}']*100).quantile(0.05).values
if stat=='std':
    quantile95_biomass = (bio_change_std_boot[f'biomass_{mhw_key}']*100).quantile(0.95).values
    quantile5_biomass = (bio_change_std_boot[f'biomass_{mhw_key}']*100).quantile(0.05).values
value_bio = np.max(np.abs([quantile95_biomass, quantile5_biomass]))
cmap_biomass = LinearSegmentedColormap.from_list('purple_white_teal', ["#AEA8DE", "white", "#94D2BD"])
norm_biomass = mcolors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)

if stat=='median':
    quantile95_mhws = (mhws_attrib_median_boot[f'attrib_mhws_{mhw_key}']*100).quantile(0.95).values
    quantile5_mhws = (mhws_attrib_median_boot[f'attrib_mhws_{mhw_key}']*100).quantile(0.05).values
if stat=='std':
    quantile95_mhws = (mhws_attrib_std_boot[f'attrib_mhws_{mhw_key}']*100).quantile(0.95).values
    quantile5_mhws = (mhws_attrib_std_boot[f'attrib_mhws_{mhw_key}']*100).quantile(0.05).values
value_mhws=np.max(np.abs([quantile95_mhws, quantile5_mhws]))
cmap_mhws = LinearSegmentedColormap.from_list('cian_white_orange', ["#087A7C","#098688","#57B6AF", "#A4D8C7","white", "#E09907", "#D47D04", "#B05A03", "#9B5107"])
norm_mhws = mcolors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)

if stat=='median':
    quantile95_warming = (warming_attrib_median_boot[f'attrib_warming_{mhw_key}']*100).quantile(0.95).values
    quantile5_warming = (warming_attrib_median_boot[f'attrib_warming_{mhw_key}']*100).quantile(0.05).values
if stat=='std':
    quantile95_warming = (warming_attrib_std_boot[f'attrib_warming_{mhw_key}']*100).quantile(0.95).values
    quantile5_warming = (warming_attrib_std_boot[f'attrib_warming_{mhw_key}']*100).quantile(0.05).values
value_warming=np.max(np.abs([quantile95_warming, quantile5_warming]))
cmap_lt = LinearSegmentedColormap.from_list('blue_to_purple', ["#266192", "#74A3CB", "#8BACC1", "#C5D6E0", "#FFFFFF","#C29BAB", "#A76382", "#742B4F", "#4A1730"])
norm_lt = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)

if stat=='median':
    quantile95_res= (residuals_attrib_median_boot[f'residual_{mhw_key}']*100).quantile(0.95).values
    quantile5_res = (residuals_attrib_median_boot[f'residual_{mhw_key}']*100).quantile(0.05).values
if stat=='std':
    quantile95_res= (residuals_attrib_std_boot[f'residual_{mhw_key}']*100).quantile(0.95).values
    quantile5_res = (residuals_attrib_std_boot[f'residual_{mhw_key}']*100).quantile(0.05).values
value_resid=np.max(np.abs([quantile95_res, quantile5_res]))
cmap_resid = LinearSegmentedColormap.from_list('greenish', ['#457680', "#658F97", "#85A7AD", "#B6CACE", "#FFFFFF", "#DEDA90", "#A49F3D", "#76732D", "#595822" ])
norm_resid = mcolors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)


# --- Data to plot ---
if stat=='median':
    panels = [
        (bio_change_median_boot[f'biomass_{mhw_key}']*100, 'Seasonal biomass change w.r.t clim', cmap_biomass, norm_biomass, '$\Delta$B [\%]'),
        (mhws_attrib_median_boot[f'attrib_mhws_{mhw_key}']*100, 'Attribution to MHWs', cmap_mhws, norm_mhws, r'$\Delta_{MHWs}$ [\%]'),
        (warming_attrib_median_boot[f'attrib_warming_{mhw_key}']*100,  'Attribution to Long-term thermal trend', cmap_lt, norm_lt, r'$\Delta_{bkgd}$ [\%]'),
        (residuals_attrib_median_boot[f'residual_{mhw_key}']*100, 'Residuals', cmap_resid, norm_resid, r'$\Delta_{res}$ [\%]'),
    ]

if stat=='std':
    panels = [
        (bio_change_std_boot[f'biomass_{mhw_key}']*100, 'Seasonal biomass change w.r.t clim', cmap_biomass, norm_biomass, 'Biomass change w.r.t clim [\%]'),
        (mhws_attrib_std_boot[f'attrib_mhws_{mhw_key}']*100, 'Attribution to MHWs', cmap_mhws, norm_mhws, r'$\Delta_{MHWs}$ [\%]'),
        (warming_attrib_std_boot[f'attrib_warming_{mhw_key}']*100,  'Attribution to Long-term thermal trend', cmap_lt, norm_lt, r'$\Delta_{bkgd}$ [\%]'),
        (residuals_attrib_std_boot[f'residual_{mhw_key}']*100, 'Residuals', cmap_resid, norm_resid, r'$\Delta_{res}$ [\%]'),
    ]
# --- Font size settings ---
plot='report'
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 10}
subtitle_kwargs  = {'fontsize': 15} if plot == 'slides' else {'fontsize': 9}
label_kwargs     = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
tick_kwargs      = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}
lw = 1   if plot == 'slides' else 0.5
lw_grid= 0.7 if plot == 'slides' else 0.3
gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 8, 'rotation': 0}

# --- Figure ---
fig, axes = plt.subplots(
    2, 2,
    figsize=(8, 8),
    subplot_kw={'projection': ccrs.SouthPolarStereo()}
)

# gs  = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
axes=axes.ravel()

theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)


for ax, (da, title, cmap, norm, cbar_label) in zip(axes, panels):
    # ax = fig.add_subplot(gs[i], projection=ccrs.SouthPolarStereo())
    # ax.set_autoscale_on(False)
    
    # Circular boundary for polar projection
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    # Features
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # -- Plot data --
    pcm = ax.pcolormesh(da.lon_rho, da.lat_rho, da.values,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, norm=norm,
                         shading='auto', zorder=2, rasterized=True)

    # -- Colorbar --
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.15, axes_class=plt.Axes)
    cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, **label_kwargs)

    # -- MPA boundaries --
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

    # Circular boundary
    ax.set_boundary(circle, transform=ax.transAxes)


    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5,
                      linestyle='--', linewidth=lw_grid, zorder=20)
    gl.xlabels_top  = False
    gl.ylabels_right = False
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()


from matplotlib.patches import Patch
no_mhw_patch = Patch(facecolor='lightgrey', edgecolor='gray', linewidth=0.5, label='No MHWs detected')
fig.legend(handles=[no_mhw_patch], loc='lower center',
           bbox_to_anchor=(0.5, -0.01), frameon=True, **label_kwargs)

# fig.suptitle(f'Attribution of changes in seasonal biomass under MHWs of {mhw_key[0]}°C', fontsize=14, fontweight='bold', y=0.94)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/fig3_attributions_maps_SO_{mhw_key}_{stat}.pdf'), dpi=200, format='pdf', bbox_inches='tight')

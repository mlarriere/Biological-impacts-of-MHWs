"""
Created on Thurs 29 Jan 14:16:03 2025

Attribution of MHW to change in biomass

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
path_biomass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_biomass_ts = os.path.join(path_surrogates, f'biomass_timeseries')
path_biomass_ts_SO = os.path.join(path_biomass_ts, f'SouthernOcean')
path_biomass_ts_MPAs = os.path.join(path_biomass_ts, f'mpas')
path_masslength = os.path.join(path_surrogates, f'mass_length')
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')

# %% ======================== Load MPAs ========================
# ---- Load data
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

mpa_dict = {"Ross Sea": (mpas_ds.mask_rs, "#c77c27"),
            "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#e05c8a"),
            "East Antarctic": (mpas_ds.mask_ea, "#C00225"),
            "Weddell Sea": (mpas_ds.mask_ws, "#5f0f40"),
            "Antarctic Peninsula": (mpas_ds.mask_ap, "#867308")}

mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}

mpa_colors = {
    "RS": "#c77c27",
    "SO": "#e05c8a",
    "EA": "#C00225",
    "WS": "#5f0f40",
    "AP": "#867308"
}
# --- Calculate MPAs area and volume
# Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km2
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km3

# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)
area_SO_np = area_60S_SO.values #in km2 -- shape (231, 1440)

area_mpa = {}
volume_mpa = {}

for abbrv, (name, mask) in mpa_masks.items():
    area_mpa[abbrv] = area_60S_SO.where(mask)
    volume_mpa[abbrv] = volume_60S_SO_100m.where(mask)
    volume_mpa[name] = volume_60S_SO_100m.where(mask)


# %% ======================== Load Biomass data ========================
surrogate_names = {"clim": "Climatology", "actual": "Actual Conditions", "climtrend":"Climatology wih trend", "nowarming": "No Warming"}
mpa_abbrs = list(mpa_masks.keys())

files_interp = [os.path.join(path_biomass_ts_MPAs, f"{sur}_{abbrv}_biomass.nc") for sur in surrogate_names.keys() for abbrv in mpa_masks.keys()]
biomass_mpas = {}

for abbrv, (mpa_name, _) in mpa_masks.items():
    biomass_mpas[abbrv] = {}

    for surrog in surrogate_names.keys():
        fname = os.path.join(path_biomass_ts_MPAs, f"{surrog}_{abbrv}_biomass.nc")
        biomass_mpas[abbrv][surrog] = xr.open_dataset(fname)

# %% ======================== Plot surrogates for 1 MPA (to check) ========================
# # Last update before holidays: very long to compute! Need to save the spatial biomass average per MPAs!
# abbrv = 'WS'
# surrog_colors = {
#     'clim':      '#2166ac',
#     'actual':    '#d6604d',
#     'climtrend': '#4dac26',
#     'nowarming': '#984ea3',
# }

# years_coord = np.arange(1980, 2019)

# fig, ax = plt.subplots(1, 1, figsize=(12, 4))

# for surrog, label in surrogate_names.items():
#     da = biomass_mpas[abbrv][surrog]['biomass'] 
#     da_yr = da.mean(dim=('days', 'eta_rho', 'xi_rho'))
#     boot_mean = da_yr.mean(dim='bootstraps') 
#     boot_std  = da_yr.std(dim='bootstraps') 

#     ax.plot(years_coord, boot_mean.values, color=surrog_colors[surrog], lw=1.6, label=label)
#     ax.fill_between(years_coord,
#                     (boot_mean - boot_std).values,
#                     (boot_mean + boot_std).values,
#                     color=surrog_colors[surrog], alpha=0.15)

# ax.set_title(f'{abbrv} - Surrogates Biomass', fontsize=14, fontweight='bold',
#              color=mpa_colors[abbrv], loc='left', pad=4)
# ax.set_ylabel('Biomass [mg/m³]', fontsize=11, color='dimgray')
# ax.set_xlabel('Years', fontsize=12)
# ax.tick_params(axis='y', labelsize=10)
# ax.spines[['top', 'right']].set_visible(False)
# ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
# ax.set_xlim(1980, 2018)
# ax.legend(fontsize=10, ncol=4, framealpha=0.6, loc='upper right')
# plt.tight_layout()
# plt.show()


# %% =====================================================================================
#           Attribution of seasonal biomass gain to MHWs and long term T°C trend
#    =====================================================================================
path_attribution = os.path.join(path_surrogates, 'attributions')
path_attribution_mpas = os.path.join(path_attribution, 'mpas')

# %% ======================== Step1. Seasonal gains for the different surrogates ========================
def compute_and_save_seasonal_gain(abbrv):
    """Compute seasonal gain for all surrogates of one MPA and save to netCDF."""
    mpa_name, _ = mpa_masks[abbrv]
    fpath = os.path.join(path_attribution_mpas, f'seasonal_gain_{abbrv}.nc')

    if not os.path.exists(fpath):
        gain = {}
        for surrog in surrogate_names.keys():
            gain[surrog] = (biomass_mpas[abbrv][surrog].isel(days=-1) - biomass_mpas[abbrv][surrog].isel(days=0)).biomass  # shape (39, 10, 231, 1440)

        xr.Dataset(gain).to_netcdf(fpath)
        print(f"{abbrv} -- computed and saved.")

    ds = xr.open_dataset(fpath)
    result = {surrog: ds[surrog] for surrog in surrogate_names}
    ds.close()
    return abbrv, result


# Run in parallel
abbrvs = list(mpa_masks.keys())
results = process_map(compute_and_save_seasonal_gain, abbrvs, max_workers=5, desc="Seasonal gain")
seasonal_gain_mpa = dict(results) #seasonal_gain_mpa['RS']['clim']  shape (39, 10, 231, 1440)

# %% ======================== Step2. Change relative to climatology per grid cell (fraction) ========================
def compute_biomass_change(abbrv):
    """Fraction of change relative to climatology per grid cell."""
    fpath = os.path.join(path_attribution_mpas, f'bio_change_{abbrv}.nc')

    if not os.path.exists(fpath):
        bio_change = {}
        for surrog in [s for s in surrogate_names if s != "clim"]:
            bio_change[surrog] = (
                (seasonal_gain_mpa[abbrv][surrog] - seasonal_gain_mpa[abbrv]['clim'])
                / seasonal_gain_mpa[abbrv]['clim']
            )  # shape (39, 10, eta_rho, xi_rho)

        xr.Dataset(bio_change).to_netcdf(fpath)
        print(f"[{abbrv}] Computed and saved.")

    ds = xr.open_dataset(fpath)
    result = {surrog: ds[surrog] for surrog in surrogate_names if surrog != "clim"}
    ds.close()
    print(f"[{abbrv}] Loaded.")
    return abbrv, result

# Run in parallel
abbrvs = list(mpa_masks.keys())
results = process_map(compute_biomass_change, abbrvs, max_workers=5, desc="Fraction of change")
bio_change_interp_mpa = dict(results)

# %% ======================== Step3. Contributions of warming and MHWs per grid cell and spatial average (Step4)
def compute_contribution(abbrv):
    """Compute MHW and warming contributions for one MPA."""
    fpath = os.path.join(path_attribution_mpas, f'attrib_{abbrv}.nc')

    if not os.path.exists(fpath):
        # abbrv='WS'
        contrib_mhws = bio_change_interp_mpa[abbrv]['actual'] - bio_change_interp_mpa[abbrv]['climtrend']  # shape (39, 10, 231, 1440)
        contrib_warming = bio_change_interp_mpa[abbrv]['actual'] - bio_change_interp_mpa[abbrv]['nowarming']  # shape (39, 10, 231, 1440)
        residual = bio_change_interp_mpa[abbrv]['actual'] - contrib_mhws - contrib_warming #residuals are mainly the chla 'contribution' according to the Atkinson model

        # Quick check 
        # check_sum = contrib_mhws.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200) + contrib_warming.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200) + residual.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200)
        # print((check_sum-p_change_interp_mpa[abbrv]['actual'].isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200)).values) #should be close to 0

        ds_out = xr.Dataset(data_vars={'mhws': (contrib_mhws.dims, contrib_mhws.data,
                                                 {'description': 'Contribution of MHWs [%] to the change in seasonal krill biomass.'}),
                                        'warming': (contrib_warming.dims, contrib_warming.data,
                                                    {'description': 'Contribution of long-term warming [%] to the change in seasonal krill biomass.'}),
                                        'residual': (residual.dims, residual.data,
                                                     {'description': 'Residual term computed as reference - mhws - warming. Represents Chla contributions and an unexplained part.'})
                                        },
                            )
        ds_out.to_netcdf(fpath)
        # xr.Dataset({'mhws': contrib_mhws, 'warming': contrib_warming, 'residual': residual}).to_netcdf(fpath)
        print(f"[{abbrv}] Computed and saved.")

    ds = xr.open_dataset(fpath)
    contrib_mhws = ds['mhws']
    contrib_warming = ds['warming']
    residual = ds['residual']
    ds.close()
    print(f"[{abbrv}] Loaded.")

    return abbrv, {
        'mhws':         contrib_mhws,
        'warming':      contrib_warming,
        'residual':     residual
        }

# Run in parallel
abbrvs = list(mpa_masks.keys())
results = process_map(compute_contribution, abbrvs, max_workers=5, desc="Attributions")
results = dict(results)

attrib_ds_mpa = {}
for abbrv in mpa_masks:
    attrib_ds_mpa[abbrv] = xr.Dataset(
        {"attrib_mhws": results[abbrv]["mhws"],
         "attrib_warming": results[abbrv]["warming"],
         "residual": results[abbrv]["residual"],
        }
    )

# %% ======================== Mask the attributions -- MHW cells only ========================
def compute_and_save_mpa(abbrv):
    # abbrv='RS'
    print(f'     {mpa_masks[abbrv][0]}')
    
    # --- MHWs mask
    mhw = xr.open_dataset(os.path.join(path_combined_thesh, "mpas/interpolated", f"duration_AND_thresh_{abbrv}.nc"))
    duration_mask = mhw["duration"] > 0
    det = {i: mhw[f"det_{i}deg"] == 1 for i in [1, 2, 3, 4]}
    intensity_masks = {
        "90perc": duration_mask.any("days"),
        "1deg": (duration_mask & det[1]).any("days"),
        "2deg": (duration_mask & det[2]).any("days"),
        "3deg": (duration_mask & det[3]).any("days"),
        "4deg": (duration_mask & det[4]).any("days"),
    }

    # Select the attributions computed before for the selected MPA
    attrib_ds = attrib_ds_mpa[abbrv]
    bio = bio_change_interp_mpa[abbrv]["actual"]
    coords = {
        "lon_rho": mhw["lon_rho"],
        "lat_rho": mhw["lat_rho"],
        "years": attrib_ds['attrib_mhws'].years,
        "bootstraps": attrib_ds['attrib_mhws'].bootstraps,
        "eta_rho": attrib_ds['attrib_mhws'].eta_rho,
        "xi_rho": attrib_ds['attrib_mhws'].xi_rho,
    }

    # --- Attribution datasets
    def build(varname, desc):
        data_vars = {}

        for suffix, mask in intensity_masks.items():
            data_vars[f'{varname}_{suffix}'] = attrib_ds[varname].where(mask)

        return xr.Dataset(data_vars=data_vars,
                          coords=coords,
                          attrs={"mpa": mpa_masks[abbrv][0],   
                                 "description": f"{desc} change in biomass -- cells selected have experienced MHWs of ideg intensity"}
                         )

    ds_mhws = build("attrib_mhws", 'Contribution of MHWs to ')
    ds_warming = build("attrib_warming", 'Contribution of long term thermal trend to')
    ds_residual = build("residual", 'Residuals of attributions to')

    # --- Biomass dataset
    ds_bio = xr.Dataset(
        data_vars={f'biomass_{suffix}': bio.where(mask) for suffix, mask in intensity_masks.items()},
        coords=coords,
        attrs={
            "mpa": mpa_masks[abbrv][0],
            "description": "Seasonal biomass change relative to climatology, masked by MHW intensity",
        },
    )

    mhw.close()

    return ds_mhws, ds_warming, ds_residual, ds_bio

# Loop over the MPAs
for abbrv in mpa_masks:
    f_mhws = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_attrib_mhws.nc")
    f_warm = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_attrib_warming.nc")
    f_res  = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_attrib_residual.nc")
    f_bio  = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_biomass.nc")

    if not all(map(os.path.exists, [f_mhws, f_warm, f_res, f_bio])):
        print('Masking in process....')
        ds_mhws, ds_warm, ds_res, ds_bio = compute_and_save_mpa(abbrv)
        ds_mhws.to_netcdf(f_mhws)
        ds_warm.to_netcdf(f_warm)
        ds_res.to_netcdf(f_res)
        ds_bio.to_netcdf(f_bio)
    
# %% ======================== Statistics: median and std dev and sptaial avf ========================
attrib_stats_mpa = {}
thresholds = ['1deg', '3deg']

for abbrv, (name, mask) in mpa_masks.items():
    # abbrv='RS'
    print(abbrv)

    # Defining the MPA spatial mask
    name=mpa_masks[abbrv][0]
    mask=mpa_masks[abbrv][1]

    mask_bool = mask.values.astype(bool) # (231, 1442)
    # area_60S_SO.where(mask_bool).plot()
    total_mpa_area = np.nansum(area_SO_np[mask_bool])

    # ---------------- Load data ----------------
    f_mhws = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_attrib_mhws.nc")
    f_warm = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_attrib_warming.nc")
    f_res  = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_attrib_residual.nc")
    f_bio  = os.path.join(path_attribution_mpas, f"mhws/{abbrv}_biomass.nc")

    ds_mhws = xr.open_dataset(f_mhws)
    ds_warm = xr.open_dataset(f_warm)
    ds_res  = xr.open_dataset(f_res)
    ds_bio  = xr.open_dataset(f_bio)

    # ---------------- Statistics ----------------
    attrib_stats_mpa[abbrv] = {'ts': {}}

    for key in thresholds:
        print(key)
        # print(f'Sum contribution MHWs in yr 0: {np.nansum(ds_mhws[f"attrib_mhws_{key}"].isel(bootstraps=0, years=0).sum())}')
        # print(f'Sum contribution MHWs in yr 1: {np.nansum(ds_mhws[f"attrib_mhws_{key}"].isel(bootstraps=0, years=1).sum())}')
        # print(f'Sum contribution MHWs in yr 2: {np.nansum(ds_mhws[f"attrib_mhws_{key}"].isel(bootstraps=0, years=2).sum())}')
        # print(f'Sum contribution MHWs in yr 20: {np.nansum(ds_mhws[f"attrib_mhws_{key}"].isel(bootstraps=0, years=20).sum())}')
        mhw_var  = ds_mhws[f'attrib_mhws_{key}'].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values
        warm_var = ds_warm[f'attrib_warming_{key}'].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values
        res_var  = ds_res[f'residual_{key}'].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values
        bio_var = ds_bio[f'biomass_{key}'].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values

        n_years, n_boot = mhw_var.shape[0], mhw_var.shape[1]

        mhw_ts  = np.full((n_years, n_boot), np.nan)
        warm_ts = np.full((n_years, n_boot), np.nan)
        res_ts  = np.full((n_years, n_boot), np.nan)
        bio_ts  = np.full((n_years, n_boot), np.nan)

        # Fixed denominator for spatial avg: area of cells EVER affected by MHWs of this threshold, across all years (i.e. MHWs footprint)
        ever_affected = mask_bool & np.isfinite(area_SO_np) & np.isfinite(mhw_var[:, 0]).any(axis=0)  # (231, 1440)
        print(f'Number of cells ever affected: {np.sum(ever_affected)}')
        fixed_area = np.nansum(area_SO_np[ever_affected])
        print(f'Footprint area: {fixed_area}km2')

        for y in range(n_years):
            # y = 20
            mhw_yr  = mhw_var[y]   # shape (10, 231, 1440)
            warm_yr = warm_var[y]
            res_yr  = res_var[y]
            bio_yr = bio_var[y]

            # Select cells inside MPA that have been affected by MHWs THIS year
            valid_affected_cells_yr = ever_affected & np.isfinite(mhw_yr[0]) #shape (231, 1440) -- True/False
            # if y == 0 or y == 1 or y == 2 or y == 20: 
                # print(f'Number of cells affected in year {y}: {np.sum(valid_affected_cells_yr)}')
                # print(f'Sum contribution MHWs in yr {y}: {np.nansum(mhw_yr[:, valid_affected_cells_yr], axis=1)}')

            if np.any(valid_affected_cells_yr) and (total_mpa_area > 0):
                weights = area_SO_np[valid_affected_cells_yr] 
                # Area weighted
                mhw_ts[y]  = np.nansum(mhw_yr[:, valid_affected_cells_yr]  * weights, axis=1) / total_mpa_area
                warm_ts[y] = np.nansum(warm_yr[:, valid_affected_cells_yr] * weights, axis=1) / total_mpa_area
                res_ts[y]  = np.nansum(res_yr[:, valid_affected_cells_yr]  * weights, axis=1) / total_mpa_area
                bio_ts[y]  = np.nansum(bio_yr[:, valid_affected_cells_yr]  * weights, axis=1) / total_mpa_area

            else:
                mhw_ts[y] = np.nan
                warm_ts[y] = np.nan
                res_ts[y] = np.nan
                bio_ts[y] = np.nan

        # To Dataset
        attrib_stats_mpa[abbrv]['ts'][key] = {
                'mhws_median':  np.nanmedian(mhw_ts,  axis=1)*100,  # convert to percentage
                'mhws_std':     np.nanstd(mhw_ts,     axis=1)*100,  # convert to percentage
                'warm_median':  np.nanmedian(warm_ts, axis=1)*100,  # convert to percentage
                'warm_std':     np.nanstd(warm_ts,    axis=1)*100,  # convert to percentage
                'res_median':   np.nanmedian(res_ts,  axis=1)*100,  # convert to percentage
                'res_std':      np.nanstd(res_ts,     axis=1)*100,  # convert to percentage
                'bio_median':   np.nanmedian(bio_ts,  axis=1)*100,  # convert to percentage
                'bio_std':      np.nanstd(bio_ts,     axis=1)*100,  # convert to percentage
                'fixed_area':   fixed_area, 
                'total_mpa_area': total_mpa_area
                }
        

    ds_mhws.close()
    ds_warm.close()
    ds_res.close()
    ds_bio.close()


# %% ======================== Maps of attribution ========================
# import matplotlib.ticker as mticker
# from skimage import measure
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

# mhw_key = '1deg' #'1deg' or '3deg'
# stat= 'std'  # 'median' or 'std'
# plot = 'report'

# def percentage(mpa_dict, category, stat, varname, scale=100):
#     return mpa_dict[category][stat][varname] * scale

# # --- Colormap and norm ---
# from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
# quantile95_biomass = percentage(attrib_stats_mpa['RS'], 'biomass', stat, f'biomass_{mhw_key}').quantile(0.95).values
# quantile5_biomass = percentage(attrib_stats_mpa['RS'], 'biomass', stat, f'biomass_{mhw_key}').quantile(0.05).values
# value_bio = np.max(np.abs([quantile95_biomass, quantile5_biomass]))
# cmap_biomass = LinearSegmentedColormap.from_list('purple_white_teal', ["#AEA8DE", "white", "#94D2BD"])
# norm_biomass = mcolors.TwoSlopeNorm(vmin=-value_bio, vcenter=0, vmax=value_bio)

# quantile95_mhws = percentage(attrib_stats_mpa['RS'], 'mhws', stat, f'attrib_mhws_{mhw_key}').quantile(0.95).values
# quantile5_mhws = percentage(attrib_stats_mpa['RS'], 'mhws', stat, f'attrib_mhws_{mhw_key}').quantile(0.05).values
# value_mhws=np.max(np.abs([quantile95_mhws, quantile5_mhws]))
# cmap_mhws = LinearSegmentedColormap.from_list('cian_white_orange', ["#0A9396", "white", "#CA6702"])
# norm_mhws = mcolors.TwoSlopeNorm(vmin=-value_mhws, vcenter=0, vmax=value_mhws)

# quantile95_warming = percentage(attrib_stats_mpa['RS'], 'warming', stat, f'attrib_warming_{mhw_key}').quantile(0.95).values
# quantile5_warming = percentage(attrib_stats_mpa['RS'], 'warming', stat, f'attrib_warming_{mhw_key}').quantile(0.05).values
# value_warming=np.max(np.abs([quantile95_warming, quantile5_warming]))
# cmap_lt = LinearSegmentedColormap.from_list('blue_to_purple', ["#5B8FC7", "#ADC7E3", "#FFFFFF", "#896572", "#412143"])
# norm_lt = Normalize(vmin=-value_warming, vmax=value_warming)

# quantile95_res= percentage(attrib_stats_mpa['RS'], 'residual', stat, f'residual_{mhw_key}').quantile(0.95).values
# quantile5_res = percentage(attrib_stats_mpa['RS'], 'residual', stat, f'residual_{mhw_key}').quantile(0.05).values
# value_resid=np.max(np.abs([quantile95_res, quantile5_res]))
# cmap_resid = LinearSegmentedColormap.from_list('greenish', ["#6F8A8E",'#738F93', "#FFFFFF", "#898433", "#676427"])
# norm_resid = mcolors.TwoSlopeNorm(vmin=-value_resid, vcenter=0, vmax=value_resid)


# # --- Data to plot ---
# panels = [('biomass', stat, f'biomass_{mhw_key}', 'Seasonal biomass change w.r.t clim', cmap_biomass,  norm_biomass, 'Biomass change w.r.t clim [\%]'),
#           ('mhws', stat, f'attrib_mhws_{mhw_key}', 'Attribution to MHWs', cmap_mhws, norm_mhws, r'$\Delta_{MHWs}$ [\%]'),
#           ('warming', stat, f'attrib_warming_{mhw_key}', 'Attribution to Long-term thermal trend ', cmap_lt, norm_lt, r'$\Delta_{bkgd}$ [\%]'),
#           ('residual', stat, f'residual_{mhw_key}', 'Residuals', cmap_resid, norm_resid, r'$\Delta_{res}$ [\%]'),
#          ]

# lon_np = mpas_ds.lon_rho.values
# lat_np = mpas_ds.lat_rho.values
# lon_np_norm = np.where(lon_np > 180, lon_np - 360, lon_np)

# mpa_boundaries = {}

# # --- Font size settings ---
# maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 10}
# subtitle_kwargs  = {'fontsize': 15} if plot == 'slides' else {'fontsize': 9}
# label_kwargs     = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
# tick_kwargs      = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}
# lw = 1   if plot == 'slides' else 0.5
# lw_grid= 0.7 if plot == 'slides' else 0.3
# gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 8, 'rotation': 0}


# # --- Figure ---
# fig = plt.figure(figsize=(8, 8))
# gs  = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# theta  = np.linspace(0, 2 * np.pi, 200)
# verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * 0.5 + 0.5)
    

# for i, (category, stat, varname, title, cmap, norm, cbar_label) in enumerate(panels):
#     row, col = divmod(i, 2)
#     ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())

#     # Circular boundary
#     ax.set_boundary(circle, transform=ax.transAxes)
    
#     # Features
#     ax.coastlines(color='black', linewidth=lw, zorder=5)
#     ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
#     ax.set_facecolor('lightgrey')

#     # -- Plot data --
#     pcm = None
#     for mpa_name, mpa_data in attrib_stats_mpa.items():
#         try:
#             ds = mpa_data[category][stat]
#         except KeyError:
#             continue
#         if varname not in ds.data_vars:
#             continue

#         lon = ds['lon_rho'].values
#         lat = ds['lat_rho'].values
#         data = ds[varname].values

#         pcm = ax.pcolormesh(lon, lat, data*100,  # Convert to percentage
#                             transform=ccrs.PlateCarree(),
#                             cmap=cmap, norm=norm,
#                             shading='auto', zorder=2, rasterized=True)

#     # -- Colorbar --
#     if pcm is not None:
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("bottom", size="5%", pad=0.15,
#                                    axes_class=plt.Axes)
#         cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal', extend='both')
#         cbar.ax.tick_params(labelsize=8)
#         cbar.set_label(cbar_label, **label_kwargs)
#     else:
#         print(f"⚠️ No data found for panel '{title}' "
#               f"(category='{category}', stat='{stat}', var='{varname}')")

    
#     # Gridlines
#     gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5,
#                       linestyle='--', linewidth=lw_grid, zorder=20)
#     gl.xlabels_top  = False
#     gl.ylabels_right = False
#     gl.xlabel_style = gridlabel_kwargs
#     gl.ylabel_style = gridlabel_kwargs
#     gl.xformatter = LongitudeFormatter()
#     gl.yformatter = LatitudeFormatter()

#     # -- MPA boundaries --
#     lon = mpas_ds.lon_rho
#     lat = mpas_ds.lat_rho
#     for name, (mask, color) in mpa_dict.items():
#         mask_2d = mask.values if hasattr(mask, "values") else mask
#         lon_np  = lon.values
#         lat_np  = lat.values

#         contours = measure.find_contours(mask_2d.astype(float), 0.5)

#         for contour in contours:
#             eta_idx = contour[:, 0].astype(int)
#             xi_idx  = contour[:, 1].astype(int)
#             ax.plot(lon_np[eta_idx, xi_idx], lat_np[eta_idx, xi_idx],
#                     color=color, alpha=0.8, linewidth=1,
#                     transform=ccrs.PlateCarree(), zorder=2)

#     # ax.set_title(title, fontsize=11, fontweight='bold')

# from matplotlib.patches import Patch
# no_mhw_patch = Patch(facecolor='lightgrey', edgecolor='gray', linewidth=0.5, label='No MHWs detected')
# fig.legend(handles=[no_mhw_patch], loc='lower center',
#            bbox_to_anchor=(0.5, -0.01), frameon=True, **label_kwargs)


# # fig.suptitle(f'Attribution of changes in seasonal biomass under MHWs of {mhw_key[0]}°C', fontsize=14, fontweight='bold', y=0.98)
# plt.show()
# # plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/fig3_attributions_maps_mpas_{mhw_key}_{stat}.pdf'), dpi=200, format='pdf', bbox_inches='tight')




# %% ======================== Total percentage of change [%] ========================
# Spatial mean for each MPAs -- area weighted
# total_change_med = {abbrv: bio_change_interp_mpa[abbrv]['actual'].median(dim=['bootstraps']).mean(dim=['eta_rho', 'xi_rho']) for abbrv in mpa_colors}
# total_change_std = {abbrv: bio_change_interp_mpa[abbrv]['actual'].std(dim=['bootstraps']).mean(dim=['eta_rho', 'xi_rho']) for abbrv in mpa_colors}

# total_change_mean = {
#     abbrv: bio_change_interp_mpa[abbrv]['actual']
#         .mean(dim=['bootstraps'])
#         .weighted(area_mpa[abbrv].fillna(0))
#         .mean(dim=['eta_rho', 'xi_rho'])
#     for abbrv in mpa_colors
# }

# total_change_med = {
#     abbrv: bio_change_interp_mpa[abbrv]['actual']
#         .weighted(area_mpa[abbrv].fillna(0))
#         .quantile(0.5, dim=['eta_rho', 'xi_rho'])
#         .median(dim='bootstraps')
#     for abbrv in mpa_colors
# }

# total_change_std = {
#     abbrv: bio_change_interp_mpa[abbrv]['actual']
#         .std(dim=['bootstraps'])
#         .weighted(area_mpa[abbrv].fillna(0))
#         .mean(dim=['eta_rho', 'xi_rho'])
#     for abbrv in mpa_colors
# }
# %% ======================== Spatial mean ========================
# impact_mhws_mpa = {abbrv: results[abbrv]['mhws'] for abbrv in abbrvs}
# impact_warming_mpa = {abbrv: results[abbrv]['warming'] for abbrv in abbrvs}
# residual_mpa = {abbrv: results[abbrv]['residual'] for abbrv in abbrvs}
# impact_mhws_mpa_mean = {abbrv: impact_mhws_mpa[abbrv].mean(dim=['eta_rho', 'xi_rho'])}
# impact_warming_mpa_mean = {abbrv: results[abbrv]['warming_mean'] for abbrv in abbrvs}
# residual_mpa_mean = {abbrv: results[abbrv]['residual_mean'] for abbrv in abbrvs}

# %% ======================== Plot Attributions (TimeSeries) ========================
import matplotlib.colors as mcolors_lib
import matplotlib.patches as mpatches

mhw_choice = '1deg'  #1deg 3deg
years_coord = np.arange(1980, 2019)
if mhw_choice == '3deg':
    mpa_colors = {"AP": "#867308", "WS": "#5f0f40", "EA": "#C00225", "RS": "#c77c27"}
    nplots=4
else: 
    mpa_colors = {"AP": "#867308", "SO": "#e05c8a", "WS": "#5f0f40", "EA": "#C00225", "RS": "#c77c27"}
    nplots=5
def lighten(hex_color, factor=0.5):
    rgb = mcolors_lib.to_rgb(hex_color)
    return tuple(1 - (1 - c) * factor for c in rgb)

def darken(hex_color, factor=0.6):
    rgb = mcolors_lib.to_rgb(hex_color)
    return tuple(c * factor for c in rgb)

fig, axes = plt.subplots(nplots, 1, figsize=(5, 8), sharex=True)
fig.subplots_adjust(hspace=0.4)

for i, (abbrv, ax) in enumerate(zip(mpa_colors.keys(), axes)):
    base  = mpa_colors[abbrv]
    light = lighten(base, factor=0.5)
    dark  = darken(base,  factor=0.6)

    # --- Extract data ---
    mhw = attrib_stats_mpa[abbrv]['ts'][mhw_choice]['mhws_median']
    warm = attrib_stats_mpa[abbrv]['ts'][mhw_choice]['warm_median'] 
    residual = attrib_stats_mpa[abbrv]['ts'][mhw_choice]['res_median']
    bio_ts = attrib_stats_mpa[abbrv]['ts'][mhw_choice]['bio_median']
    bio_ts_std = attrib_stats_mpa[abbrv]['ts'][mhw_choice]['bio_std']

    # Exposure fraction for this MPA and threshold 
    total_mpa_area = attrib_stats_mpa[abbrv]['ts'][mhw_choice]['total_mpa_area']
    fixed_area = attrib_stats_mpa[abbrv]['ts'][mhw_choice]['fixed_area']
    exposure_pct = 100 * fixed_area / total_mpa_area

    ax2 = ax.twinx()
    ax2.set_zorder(0)
    ax.set_zorder(2)

    ax.patch.set_visible(False)   # important
    ax2.patch.set_alpha(0.0)      # important

    # --- left axis: contribution ---
    ax.plot(years_coord, warm, color='#5B8FC7', lw=1, label=r'$\Delta_{bkgd}$', zorder=2)
    ax.plot(years_coord, mhw, color='#CA6702', lw=1, label=r'$\Delta_{MHWs}$', zorder=3)
    ax.plot(years_coord, residual, color='#898433', lw=1, label=r'$\Delta_{res}$', zorder=4)

    # ax.axhline(0, color='0.7', lw=0.8)
    ax.axhline(0, color='gray', lw=0.6, ls='-', alpha=0.4)
    ax.set_xlim(1980, 2018)
    ax.set_ylabel('Contribution [\%]', fontsize=12, color='black')
    ax.tick_params(axis='y', labelsize=10, labelcolor='black')
    ax.spines[['top', 'right']].set_visible(False)

    # --- right axis: Biomass subplot
    bar_colors = ['#94D2BD' if v >= 0 else '#AEA8DE' for v in bio_ts]
    ax2.bar(years_coord, bio_ts, width=0.9, color=bar_colors, alpha=0.4, zorder=1)
    # ax2.bar(years_coord, bio_ts, width=0.9, color=bar_colors, alpha=0.4, zorder=1,
        # yerr=bio_ts_std, error_kw=dict(ecolor='0.3', elinewidth=0.8, capsize=2, alpha=0.6),)
    ax2.set_ylabel('Biomass Change [\%]', fontsize=12)
    ax2.axhline(0, color='black', lw=0.5, ls=':', alpha=0.6)
    ax2.tick_params(labelsize=10)
    bio_max = np.nanmax(np.abs(bio_ts)) * 1.2
    ax2.set_ylim(-bio_max, bio_max)

    # zero-align the two axes so both 0s sit on the same gridline
    pp_abs = max(abs(np.nanmin([mhw, warm, residual])), abs(np.nanmax([mhw, warm, residual]))) * 1.25
    tc_abs = max(abs(np.nanmin(bio_ts)), abs(np.nanmax(bio_ts))) * 1.25
    ax.set_ylim(-pp_abs, pp_abs)
    ax2.set_ylim(-tc_abs, tc_abs)

    ax.set_title(f'{abbrv}',fontsize=14, fontweight='bold', color='black', loc='left', pad=4)
    
    # --- Legend ---
    # ax.text(0.02, 0.95, f'MHW footprint: {exposure_pct:.1f}\\% of MPA',
    #     transform=ax.transAxes, fontsize=9, color='black',
    #     ha='left', va='top', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.75),)

# --- Shared legend ---
line_warm = plt.Line2D([0], [0], color='#5B8FC7', lw=1.5, label=r'$\Delta_{bkgd}$')
line_mhw  = plt.Line2D([0], [0], color='#CA6702', lw=1.5, label=r'$\Delta_{MHWs}$')
line_res  = plt.Line2D([0], [0], color='#898433', lw=1.5, label=r'$\Delta_{res}$')

fig.legend(
    handles=[line_mhw, line_warm, line_res],
    fontsize=10, loc='upper center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=3, framealpha=0.6,
    handlelength=1.2, handleheight=0.9,
    borderpad=0.5, labelspacing=0.3,
)

axes[-1].set_xlabel('Years', fontsize=12)
# patch_mhw  = mpatches.Patch(color=light, alpha=0.85, label=r'$\Delta_{MHWs}$ [\%pt.]')
# patch_warm = mpatches.Patch(color=dark,  alpha=0.85, label=r'$\Delta_{bkgd}$ [\%pt.]')
# # patch_residual = mpatches.Patch(color='black', alpha=0.85, label=r'\Delta_{res} [%pt.]')
# line_residual = plt.Line2D([0], [0], color='black',lw=1.8,
# ls=':',
# label=r'$\Delta_{res}$ [\%pt.]'
# )
# ax.legend(
#     handles=[patch_mhw, patch_warm, line_residual],
#     fontsize=10, loc='upper right',
#     framealpha=0.6, ncol=4,
#     handlelength=1.2, handleheight=0.9,
#     borderpad=0.5, labelspacing=0.3,
# )
# fig.suptitle('Biomass attribution: MHW and Warming vs. Total change\nNote that the different terms are averafed spatially and thus don''t add up',fontsize=15, y=1.005)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/fig3_attributions_ts_{mhw_choice}.pdf'), dpi=200, format='pdf', bbox_inches='tight')

# %% ======================== MHWs Metrics ========================
# 1. MHW duration
mhw_duration_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/interpolated/duration_AND_thresh_{region}.nc")).duration for region in ['RS', 'SO', 'EA', 'WS', 'AP']}

# 2. MHW intensity
mhw_1deg_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/interpolated/duration_AND_thresh_{region}.nc")).det_1deg for region in ['RS', 'SO', 'EA', 'WS', 'AP']}
mhw_2deg_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/interpolated/duration_AND_thresh_{region}.nc")).det_2deg for region in ['RS', 'SO', 'EA', 'WS', 'AP']}
mhw_3deg_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/interpolated/duration_AND_thresh_{region}.nc")).det_3deg for region in ['RS', 'SO', 'EA', 'WS', 'AP']}
mhw_4deg_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/interpolated/duration_AND_thresh_{region}.nc")).det_4deg for region in ['RS', 'SO', 'EA', 'WS', 'AP']}

# 3. MHW area
mhw_area_affected_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/mhw_daily_area_{region}.nc")) for region in ['RS', 'SO', 'EA', 'WS', 'AP']}


# %% ================================================================================================
#                Plot the seasonal gain VS MHW metrics 
#    ================================================================================================
# %% ======== 1. DURATION ========
# --- Compute MHW duration per region ---
def get_mhw_dur(region):
    da = mhw_duration_mpas[region]
    return da.max(dim='days').where(da.max(dim='days') > 0).mean(dim=['eta_rho', 'xi_rho'])

# -- Color settings
import matplotlib.cm as cm
years = np.arange(1980, 2019)
cmap = cm.RdYlBu_r
norm = plt.Normalize(vmin=years.min(), vmax=years.max())

mpa_regions = ['RS', 'SO', 'EA', 'WS', 'AP']
mpa_titles  = {k: mpa_masks[k][0] for k in mpa_regions}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, region in enumerate(mpa_regions):
    ax = axes[idx]

    # -- Data
    mhw_dur     = get_mhw_dur(region)
    actual_gain = total_seasonal_gain[region]['actual'].median(dim=['algo']).biomass * 1e-12

    # -- Align years (coord is 0-38, map to 1980-2018)
    x = mhw_dur.values
    y = actual_gain.values

    sc = ax.scatter(x, y, c=years, cmap=cmap, norm=norm,
                    s=60, zorder=3, edgecolors='k', linewidths=0.4)

    # -- Colorbar per subplot
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Year', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # -- Labels
    ax.set_xlabel('Max MHW Duration [days]', fontsize=10)
    ax.set_ylabel('Total Seasonal Biomass Gain [Mt]', fontsize=10)
    ax.set_title(mpa_titles[region], fontsize=11, fontweight='bold')

# -- Hide unused subplot (6th panel)
axes[-1].set_visible(False)

fig.suptitle('Seasonal Biomass Gain vs MHW Duration', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% ======== 2. AREA ========
# --- Fix mhw_area per region ---
def get_mhw_area(region, threshold_idx=2):
    da = mhw_area_affected_mpas[region]
    area_var = list(da.data_vars)[0]
    return da[area_var].isel(threshold=threshold_idx).max(dim='days')

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, region in enumerate(mpa_regions):
    ax = axes[idx]

    # -- Data (1deg threshold)
    mhw_area    = get_mhw_area(region, threshold_idx=2)
    actual_gain = total_seasonal_gain[region]['actual'].median(dim=['algo']).biomass * 1e-12

    x = mhw_area.values
    y = actual_gain.values

    sc = ax.scatter(x, y, c=years, cmap=cmap, norm=norm,
                    s=60, zorder=3, edgecolors='k', linewidths=0.4)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Year', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel('Max MHW Area Coverage [km²]', fontsize=10)
    ax.set_ylabel('Total Seasonal Biomass Gain [Mt]', fontsize=10)
    ax.set_title(mpa_titles[region], fontsize=11, fontweight='bold')

axes[-1].set_visible(False)

fig.suptitle('Seasonal Biomass Gain vs MHW Area Coverage (3°C threshold)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% ======================== Plot the seasonal gain VS MHW metrics ========================
# -- Parameters
abbrv = "AP"
legend_names = {"clim": "Climatology", "actual": "Actual", "climtrend": "No MHWs", "nowarming": "No warming"}
colors = {"actual": "#648028", "climtrend": "#F18701", "nowarming": "#584CBD"}
surrogates = ["actual", "climtrend", "nowarming"]

# -- Select data
mhw_dur = mhw_duration_mpas[abbrv].max(dim=['days','eta_rho', 'xi_rho'])
actual_gain = total_seasonal_gain[abbrv]['actual'].median(dim=['algo']).biomass

# -- Color settings
import matplotlib.cm as cm
years = np.arange(1980, 2019)
cmap = cm.RdYlBu_r
norm = plt.Normalize(vmin=years.min(), vmax=years.max())
colors_yr = cmap(norm(years))

fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(mhw_dur, actual_gain, c=years, cmap=cmap, norm=norm, s=60, zorder=3, edgecolors='k', linewidths=0.4)

# -- Annotate outliers (e.g. >2 std from mean)
mean_gain = float(actual_gain.mean())
std_gain  = float(actual_gain.std())
for i, yr in enumerate(years):
    if abs(float(actual_gain.isel(years=i)) - mean_gain) > 1.5 * std_gain:
        ax.annotate(str(yr), 
                    (float(mhw_dur.isel(years=i)), float(actual_gain.isel(years=i))),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

# -- Trend line
x = mhw_dur.values.flatten()
y = actual_gain.values.flatten()
mask = np.isfinite(x) & np.isfinite(y)
z = np.polyfit(x[mask], y[mask], 1)
p = np.poly1d(z)
x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.2, alpha=0.6, label=f'Trend (slope={z[0]:.2e})')

# -- Colorbar
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Year', fontsize=11)

# -- Labels
ax.set_xlabel('Max MHW Duration [days]', fontsize=12)
ax.set_ylabel('Total Seasonal Biomass Gain [Mt]', fontsize=12)
ax.set_title(f'Seasonal Biomass Gain vs MHW Duration\nMPA: {abbrv}', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




# %% ======================== Plot the seasonal gain for the different scenatio and MPA ========================
abbrv = "EA"
legend_names = {"clim": "Climatology", "actual": "Actual", "climtrend": "No MHWs", "nowarming": "No warming"}
colors = {"actual": "#648028", "climtrend": "#F18701", "nowarming": "#584CBD"}
surrogates = ["actual", "climtrend", "nowarming"]

# --- Years and x positions ---
years = total_seasonal_gain[abbrv]["actual"].years.values + 1980
x = np.arange(len(years))
width = 0.25

# --- Climatology ---
clim_gain = total_seasonal_gain[abbrv]["clim"].biomass / 1e12  # Mt
clim_med = clim_gain.median(dim="algo")
clim_std = clim_gain.std(dim="algo")

# --- Create figure with 2 subplots ---
fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True)

# ======================= Total seasonal gain =======================
ax = axes[0]

for i, surrog in enumerate(surrogates):
    gain = total_seasonal_gain[abbrv][surrog].biomass / 1e12  # Mt
    gain_med = gain.median(dim="algo")
    gain_std = gain.std(dim="algo")

    ax.bar(x + i*width, gain_med, width=width, yerr=gain_std, 
           capsize=3, color=colors[surrog], edgecolor="black", label=legend_names[surrog])

# Climatology reference
ax.axhline(clim_med, color="#870505", ls="--", lw=1, label="Climatology")
ax.set_ylabel("Total seasonal biomass gain [Mt]", fontsize=14)
ax.set_title(f"Seasonal biomass gain under different scenario\nMPA: {abbrv}", fontsize=16)
ax.legend(ncol=2, fontsize=11)

# ======================= Percentage change =======================
ax = axes[1]

for i, surrog in enumerate(surrogates):
    perc_change = p_change_interp[abbrv][surrog].biomass
    perc_change_med = perc_change.median(dim="algo")
    perc_change_std = perc_change.std(dim="algo")

    ax.bar(x + i*width, perc_change_med, width=width, yerr=perc_change_std, capsize=3,
           color=colors[surrog], edgecolor="black", label=legend_names[surrog])

ax.axhline(0, color="k", lw=1)
ax.set_ylabel("Change relative to climatology [\%]", fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(years, rotation=45)
ax.legend(ncol=3, fontsize=11)

plt.tight_layout()
plt.show()




# %% ======================== MHWs events ========================
# MHW area coverage
mhw_area_affected_mpas = {region: xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/mhw_daily_area_{region}.nc")) for region in ['RS', 'SO', 'EA', 'WS', 'AP']}

# Warming trend
warming_trend_mpas = {region: xr.open_dataset(os.path.join(path_surrogates, f"detrended_signal/mpas/temp_linear_trend_{region}.nc")) for region in ['RS', 'SO', 'EA', 'WS', 'AP']}

ds = warming_trend_mpas[abbrv]

# Spatial mean slope (°C / yr)
mean_slope = ds.slope.mean(dim=("eta_rho", "xi_rho"))

# Total warming from 1980–2019 (40 years)
warming_40y = mean_slope * 40

print(f"Warming 1980–2019 ({abbrv}): {warming_40y.item():.2f} °C")
# Mean spatial slope (°C / yr)
mean_slope = ds.slope.mean(dim=("eta_rho", "xi_rho")).item()

# %% ======================== Plot attributions ========================
thresholds_to_plot = ['det_1deg', 'det_3deg']
colors_thresh = ["firebrick", "darkorange"]
labels_thresh = ["90th perc and 1°C", "90th perc and 3°C"]
width = 0.35  # Bar width

# --- Years axis
years = impact_mhws[abbrv]["years"].values
calendar_years = 1980 + years

# --- Attribution
mhw_med = impact_mhws[abbrv].biomass.median("algo").values
mhw_std = impact_mhws[abbrv].biomass.std("algo").values
warm_med = impact_warming[abbrv].biomass.median("algo").values
warm_std = impact_warming[abbrv].biomass.std("algo").values

# --- Warming signal (cumulative 1980–2019)
ds = warming_trend_mpas[abbrv]
mean_slope = ds.slope.mean(dim=("eta_rho", "xi_rho")).item()  # °C/yr
warming_40y = mean_slope * 40  # cumulative warming after 40 years

# --- Figure setup
fig, axes = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={"height_ratios":[2, 1]}
)

# ==============================
# 1) Attribution (MHW + warming)
# ==============================
ax = axes[0]
ax.bar(calendar_years - width/2, mhw_med, width=width, yerr=mhw_std, capsize=3,
       color="crimson", edgecolor="black",
       label=r"MHW impact ($\mathrm{median}\pm\sigma$)")

ax.bar(calendar_years + width/2, warm_med, width=width, yerr=warm_std, capsize=3,
       color="royalblue", edgecolor="black",
       label=r"Warming impact ($\mathrm{median}\pm\sigma$)")

ax.axhline(0, lw=1, ls="--", color="k")
ax.set_ylabel(r"Contribution to seasonal $\Delta$ Biomass [\%]")
ax.set_title(f"Attribution of seasonal biomass change — MPA: {abbrv}", fontsize=18)
# ax.set_ylim(warm_med.min()-10, mhw_med.max()+15)
ax.legend(frameon=True, loc='lower left')


# Add text box for cumulative warming
ax.text(0.99, 0.95, f"Cumulative warming after 40y: {warming_40y:.2f} °C",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# ==============================
# 2) MHW area affected (stacked)
# ==============================
ax2 = axes[1]
bottom_vals = np.zeros(len(calendar_years))

for thresh, color, label in zip(thresholds_to_plot, colors_thresh, labels_thresh):
    mhw_area_mean = (
        mhw_area_affected_mpas[abbrv]
        .sel(threshold=thresh)
        .mhw_area_affected
        .max(dim="days")
        .values
    )
    ax2.bar(calendar_years, mhw_area_mean, bottom=bottom_vals,
            width=0.8, color=color, edgecolor="black", label=label)
    bottom_vals += mhw_area_mean
# ax2.set_ylim(0,7)
ax2.set_ylabel("Max MHW area affected [\%]")
ax2.set_title("MHW area affected")
ax2.legend(frameon=True, loc="upper left")

# -----------------------------
# X-axis ticks every 5 years
# -----------------------------
tick_years = np.arange(1980, 2021, 5)
ax2.set_xticks(tick_years)
ax2.set_xticklabels(tick_years, rotation=45)
ax2.set_xlabel("Year")

plt.tight_layout()
plt.show()



# %% ======================== Adding Chla Anomalies ========================
# -- Load data
datasets_anom = {}
datasets_intens = {}
datasets_clim = {}
datasets_thresh = {}
datasets_area = {}

# Growth season
season_days = np.concatenate([np.arange(305, 365), np.arange(0, 121)])  # Nov 1 - Apr 30 (181 days)


for region in mpa_masks.keys():
    datasets_anom[region] = xr.open_dataset(os.path.join(path_growth_inputs, f'chla_anomalies/mpas/chla_anomalies_{abbrv}.nc')).isel(days=season_days)
    datasets_intens[region] = xr.open_dataset(os.path.join(path_growth_inputs, f'chla_anomalies/mpas/chla_intensity_{abbrv}.nc')).isel(days=season_days)
    datasets_clim[region] = xr.open_dataset(os.path.join(path_growth_inputs, f'chla_anomalies/mpas/chla_clim_{abbrv}.nc')).isel(days=season_days)
    datasets_thresh[region] = xr.open_dataset(os.path.join(path_growth_inputs, f'chla_anomalies/mpas/chla_thresholds_{abbrv}.nc')).isel(days=season_days)
    datasets_area[region] = xr.open_dataset(os.path.join(path_growth_inputs, f'chla_anomalies/mpas/chla_daily_area_{abbrv}.nc')).isel(days=season_days, years=slice(0,39))

# datasets_area['RS'] = xr.open_dataset(os.path.join(path_growth_inputs, f'chla_anomalies/mpas/chla_daily_area_RS.nc'))
  
# %% ======================== Plot attributions ========================
thresholds_to_plot = ['det_1deg', 'det_3deg']
colors_thresh = ["firebrick", "darkorange"]
labels_thresh = ["90th perc and 1°C", "90th perc and 3°C"]
width = 0.35  # Bar width

# --- Years axis
years = impact_mhws[abbrv]["years"].values
calendar_years = 1980 + years

# --- Attribution
mhw_med = impact_mhws[abbrv].biomass.median("algo").values
mhw_std = impact_mhws[abbrv].biomass.std("algo").values
warm_med = impact_warming[abbrv].biomass.median("algo").values
warm_std = impact_warming[abbrv].biomass.std("algo").values

# --- Warming signal (cumulative 1980–2019)
ds = warming_trend_mpas[abbrv]
mean_slope = ds.slope.mean(dim=("eta_rho", "xi_rho")).item()  # °C/yr
warming_40y = mean_slope * 40  # cumulative warming after 40 years

# --- Figure setup
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios":[2, 1,1]})

# ==============================
# 1) Attribution (MHW + warming)
# ==============================
ax = axes[0]
ax.bar(calendar_years - width/2, mhw_med, width=width, yerr=mhw_std, capsize=3,
       color="crimson", edgecolor="black",
       label=r"MHW impact ($\mathrm{median}\pm\sigma$)")

ax.bar(calendar_years + width/2, warm_med, width=width, yerr=warm_std, capsize=3,
       color="royalblue", edgecolor="black",
       label=r"Warming impact ($\mathrm{median}\pm\sigma$)")

ax.axhline(0, lw=1, ls="--", color="k")
ax.set_ylabel(r"Contribution to seasonal $\Delta$ Biomass [\%]")
ax.set_title(f"Attribution of seasonal biomass change — MPA: {abbrv}", fontsize=18)
# ax.set_ylim(warm_med.min()-10, mhw_med.max()+15)
ax.legend(frameon=True, loc='lower left')


# Add text box for cumulative warming
ax.text(0.99, 0.95, f"Cumulative warming after 40y: {warming_40y:.2f} °C",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# ==============================
# 2) MHW area affected (stacked)
# ==============================
ax2 = axes[1]
bottom_vals = np.zeros(len(calendar_years))

for thresh, color, label in zip(thresholds_to_plot, colors_thresh, labels_thresh):
    mhw_area_max = (mhw_area_affected_mpas[abbrv].sel(threshold=thresh).mhw_area_affected.max(dim="days").values)
    ax2.bar(calendar_years, mhw_area_max, bottom=bottom_vals,
            width=0.8, color=color, edgecolor="black", label=label)
    bottom_vals += mhw_area_max
# ax2.set_ylim(0,7)
ax2.set_ylabel("Max area affected [\%]")
ax2.set_title("MHW - Max area affected")
ax2.legend(frameon=True, loc="upper left")

# ==============================
# 2) CHla Anomalies area (stacked)
# ==============================
ax3 = axes[2]
bottom_vals = np.zeros(len(calendar_years))
chla_anom = ['pos_anom', 'neg_anom']
colors_chla = ['green', 'orange']
labels_chla = [r'Bloom ($>$p90)', r'Scarcity ($<$p10)']
for thresh, color, label in zip(chla_anom, colors_chla, labels_chla):
    chla_area_max = (datasets_area[abbrv].sel(anomaly=thresh).chla_area_affected.max(dim="days").values) 
    ax3.bar(calendar_years, chla_area_max, bottom=bottom_vals, width=0.8, color=color, edgecolor="black", label=label)
    bottom_vals += chla_area_max

ax3.set_ylabel("Max area affected [\%]")
ax3.set_title("Chla Anomalies - Max area affected")
ax3.legend(frameon=True, loc="upper left")

# -----------------------------
# X-axis ticks every 5 years
# -----------------------------
tick_years = np.arange(1980, 2021, 5)
ax3.set_xticks(tick_years)
ax3.set_xticklabels(tick_years, rotation=45)
ax3.set_xlabel("Years")

plt.tight_layout()
plt.show()

# %% ========= Overall attribution over the 40 years period
mhw_mean_40y = mhw_med.mean()
warm_mean_40y = warm_med.mean()

mhw_std_40y = mhw_med.std()
warm_std_40y = warm_med.std()
print(f"MHWs: {float(mhw_mean_40y):.2f} ± {float(mhw_std_40y):.2f} %")
print(f"Warming: {float(warm_mean_40y):.2f} ± {float(warm_std_40y):.2f} %")


#  plot 
means = [mhw_mean_40y, warm_mean_40y]  
stds  = [mhw_std_40y, warm_std_40y] 
components = ["MHWs", "Warming"]
colors = ["crimson", "royalblue"]

fig, ax = plt.subplots(figsize=(8,4))

# --- Diverging bars ---
y_pos = np.arange(len(components))
ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black", capsize=5)

# Zero reference line
ax.axvline(0, color='k', lw=1, ls='--')

# Y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(components)
ax.set_xlabel(r"Contribution to seasonal $\Delta$ Biomass [\%]")
ax.set_title(f"Overall 40-year attribution (1980–2019) — MPA: {abbrv}")

# Invert y-axis so MHW is on top
ax.invert_yaxis()

plt.tight_layout()
plt.show()


# %%
impact = impact_mhws[abbrv].biomass  # (years=39, algo=5)

# Median and std per year across algorithms
median_per_year = impact.median(dim="algo")  # shape (years,)
std_per_year = impact.std(dim="algo")        # shape (years,)

# Mask positive and negative years
pos_mask = median_per_year > 0
neg_mask = median_per_year < 0

# Positive
total_pos = median_per_year[pos_mask].sum().item()
std_pos = np.sqrt((std_per_year[pos_mask]**2).sum().item())

# Negative
total_neg = median_per_year[neg_mask].sum().item()
std_neg = np.sqrt((std_per_year[neg_mask]**2).sum().item())

total_abs = abs(total_pos) + abs(total_neg)

pct_pos = total_pos / total_abs * 100
pct_neg = abs(total_neg) / total_abs * 100

# Propagate uncertainty as fraction
pct_pos_std = std_pos / total_abs * 100
pct_neg_std = std_neg / total_abs * 100

import matplotlib.pyplot as plt

sizes = [pct_pos, pct_neg]
labels = [    f"Positive impact\n{pct_pos:.1f} ± {pct_pos_std:.1f} %",
    f"Negative impact\n{pct_neg:.1f} ± {pct_neg_std:.1f} %"
]
colors = ["crimson", "royalblue"]

fig, ax = plt.subplots(figsize=(6,6))
ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, counterclock=False, wedgeprops={"edgecolor":"k"})
ax.set_title(f"Proportion of positive vs negative MHW impact — MPA: {abbrv}")
plt.show()


# %%

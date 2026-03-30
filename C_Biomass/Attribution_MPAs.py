"""
Created on Thurs 29 Jan 14:16:03 2025

Attribution of MHW to change in biomass

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

mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}
mpa_colors = {
    'RS': '#c77c27', 'SO': '#e05c8a',
    'EA': '#C00225', 'WS': '#5f0f40', 'AP': '#867308',
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

area_mpa = {}
volume_mpa = {}

for abbrv, (name, mask) in mpa_masks.items():
    area_mpa[abbrv] = area_60S_SO.where(mask)
    volume_mpa[abbrv] = volume_60S_SO_100m.where(mask)
    volume_mpa[name] = volume_60S_SO_100m.where(mask)


# %% ======================== Load Biomass data ========================
surrogate_names = {"clim": "Climatology", "actual": "Actual Conditions", "climtrend":"Climatology wih trend", "nowarming": "No Warming"}
mpa_abbrs = list(mpa_masks.keys())

files_interp = [os.path.join(path_biomass_ts_MPAs, f"{sur}_biomass_{abbrv}.nc") for sur in surrogate_names.keys() for abbrv in mpa_masks.keys()]
biomass_mpas = {}

for abbrv, (mpa_name, _) in mpa_masks.items():
    biomass_mpas[abbrv] = {}

    for surrog in surrogate_names.keys():
        fname = os.path.join(path_biomass_ts_MPAs, f"{surrog}_biomass_{abbrv}.nc")
        biomass_mpas[abbrv][surrog] = xr.open_dataset(fname)

# %% ================================================
#                       Attributions
# ===================================================
path_attribution = os.path.join(path_surrogates, 'attributions')
path_attribution_mpas = os.path.join(path_attribution, 'mpas')

# %% ===== Step1. Seasonal gains for the different surrogates
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
seasonal_gain_mpa = dict(results)

# %% ===== Step2. Percentage of change relative to climatology per grid cell
def compute_p_change(abbrv):
    """Percentage change relative to climatology per grid cell."""
    fpath = os.path.join(path_attribution_mpas, f'p_change_{abbrv}.nc')

    if not os.path.exists(fpath):
        p_change = {}
        for surrog in [s for s in surrogate_names if s != "clim"]:
            p_change[surrog] = (
                (seasonal_gain_mpa[abbrv][surrog] - seasonal_gain_mpa[abbrv]['clim'])
                / seasonal_gain_mpa[abbrv]['clim'] * 100
            )  # shape (39, 10, eta_rho, xi_rho)

        xr.Dataset(p_change).to_netcdf(fpath)
        print(f"[{abbrv}] Computed and saved.")

    ds = xr.open_dataset(fpath)
    result = {surrog: ds[surrog] for surrog in surrogate_names if surrog != "clim"}
    ds.close()
    print(f"[{abbrv}] Loaded.")
    return abbrv, result

# Run in parallel
abbrvs = list(mpa_masks.keys())
results = process_map(compute_p_change, abbrvs, max_workers=5, desc="% of change")
p_change_interp_mpa = dict(results)

# %% ===== Step3. Impact of the warming and MHWs per grid cell and spatial average (Step4)
def compute_impact(abbrv):
    """Compute MHW and warming impacts for one MPA."""
    fpath = os.path.join(path_attribution_mpas, f'impact_{abbrv}.nc')

    if not os.path.exists(fpath):
        impact_mhws    = p_change_interp_mpa[abbrv]['actual'] - p_change_interp_mpa[abbrv]['climtrend']  # shape (39, 10, eta_rho, xi_rho)
        impact_warming = p_change_interp_mpa[abbrv]['actual'] - p_change_interp_mpa[abbrv]['nowarming']  # shape (39, 10, eta_rho, xi_rho)

        xr.Dataset({'mhws': impact_mhws, 'warming': impact_warming}).to_netcdf(fpath)
        print(f"[{abbrv}] Computed and saved.")

    ds = xr.open_dataset(fpath)
    impact_mhws    = ds['mhws']
    impact_warming = ds['warming']
    ds.close()
    print(f"[{abbrv}] Loaded.")

    return abbrv, {
        'mhws':         impact_mhws,
        'warming':      impact_warming,
        'mhws_mean':    impact_mhws.mean(dim=('bootstraps', 'eta_rho', 'xi_rho')),    # shape (39,)
        'warming_mean': impact_warming.mean(dim=('bootstraps', 'eta_rho', 'xi_rho')), # shape (39,)
    }

# Run in parallel
abbrvs = list(mpa_masks.keys())
results = process_map(compute_impact, abbrvs, max_workers=5, desc="Impact")
results = dict(results)

# Reformat
impact_mhws_mpa = {abbrv: results[abbrv]['mhws'] for abbrv in abbrvs}
impact_warming_mpa = {abbrv: results[abbrv]['warming'] for abbrv in abbrvs}
impact_mhws_mpa_mean = {abbrv: results[abbrv]['mhws_mean'] for abbrv in abbrvs}
impact_warming_mpa_mean = {abbrv: results[abbrv]['warming_mean'] for abbrv in abbrvs}

# %% ======================== Plot surrogates for 1 MPAs (check) ========================
abbrv = 'WS'
surrog_colors = {
    'clim':      '#2166ac',
    'actual':    '#d6604d',
    'climtrend': '#4dac26',
    'nowarming': '#984ea3',
}
years_coord = np.arange(1980, 2019)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

for surrog, label in surrogate_names.items():
    da = biomass_mpas[abbrv][surrog]['biomass']         # (years, bootstraps, days, eta, xi)
    yearly = da.mean(dim=('days', 'eta_rho', 'xi_rho'))     # (years, bootstraps)
    boot_mean = yearly.mean(dim='bootstraps')                  # (years,)
    boot_std  = yearly.std(dim='bootstraps')                   # (years,)

    ax.plot(years_coord, boot_mean.values, color=surrog_colors[surrog], lw=1.6, label=label)
    ax.fill_between(years_coord,
                    (boot_mean - boot_std).values,
                    (boot_mean + boot_std).values,
                    color=surrog_colors[surrog], alpha=0.15)

ax.set_title(f'{abbrv} - Surrogates Biomass', fontsize=14, fontweight='bold',
             color=mpa_colors[abbrv], loc='left', pad=4)
ax.set_ylabel('Biomass [mg/m³]', fontsize=11, color='dimgray')
ax.set_xlabel('Years', fontsize=12)
ax.tick_params(axis='y', labelsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
ax.set_xlim(1980, 2018)
ax.legend(fontsize=10, ncol=4, framealpha=0.6, loc='upper right')
plt.tight_layout()
plt.show()

# %% ======================== Plot Attributions (TimeSeries) ========================
import matplotlib.colors as mcolors_lib
import matplotlib.patches as mpatches

years_coord = np.arange(1980, 2019)

mpa_colors = {
    'RS': '#c77c27', 'SO': '#e05c8a',
    'EA': '#C00225', 'WS': '#5f0f40', 'AP': '#867308',
}
mpa_labels = {
    'RS': 'Ross Sea', 'SO': 'South Orkney Islands',
    'EA': 'East Antarctic', 'WS': 'Weddell Sea', 'AP': 'Antarctic Peninsula',
}

def lighten(hex_color, factor=0.45):
    rgb = mcolors_lib.to_rgb(hex_color)
    return tuple(1 - (1 - c) * factor for c in rgb)

def darken(hex_color, factor=0.6):
    rgb = mcolors_lib.to_rgb(hex_color)
    return tuple(c * factor for c in rgb)

# --- total % change (actual vs clim) ---
total_change_mean = {
    abbrv: p_change_interp_mpa[abbrv]['actual'].mean(dim='bootstraps')
    for abbrv in mpa_colors
}

fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
fig.subplots_adjust(hspace=0.4)

for i, (abbrv, ax) in enumerate(zip(mpa_colors.keys(), axes)):
    base  = mpa_colors[abbrv]
    light = lighten(base, factor=0.45)
    dark  = darken(base,  factor=0.6)

    mhw   = impact_mhws_mpa_mean[abbrv].values
    warm  = impact_warming_mpa_mean[abbrv].values
    total = total_change_mean[abbrv].values

    # --- left axis: attribution fills (percentage points) ---
    ax.fill_between(years_coord, 0, np.where(warm > 0, warm, 0), color=dark,  alpha=0.85)
    ax.fill_between(years_coord, 0, np.where(warm < 0, warm, 0), color=dark,  alpha=0.85)
    ax.fill_between(years_coord, 0, np.where(mhw  > 0, mhw,  0), color=light, alpha=0.85)
    ax.fill_between(years_coord, 0, np.where(mhw  < 0, mhw,  0), color=light, alpha=0.85)
    ax.axhline(0, color='gray', lw=0.6, ls='-', alpha=0.4)
    ax.set_xlim(1980, 2018)
    ax.set_ylabel('Difference in \% [\%pt.]', fontsize=12, color='dimgray')
    ax.tick_params(axis='y', labelsize=10, labelcolor='dimgray')
    ax.spines[['top', 'right']].set_visible(False)

    # --- right axis: total % change ---
    ax2 = ax.twinx()
    ax2.plot(years_coord, total, color='#1a1a1a', lw=1.6, ls='--', zorder=5)
    ax2.set_ylabel('Change w.r.t clim [\%]', fontsize=12, color='#1a1a1a')
    ax2.tick_params(axis='y', labelsize=10, labelcolor='#1a1a1a')
    ax2.spines[['top', 'left']].set_visible(False)

    # keep right spine visible but subtle
    ax2.spines['right'].set_linewidth(0.6)
    ax2.spines['right'].set_color('#aaaaaa')

    # zero-align the two axes so both 0s sit on the same gridline
    pp_abs = max(abs(np.nanmin([mhw, warm])), abs(np.nanmax([mhw, warm]))) * 1.25
    tc_abs = max(abs(np.nanmin(total)), abs(np.nanmax(total))) * 1.25
    ax.set_ylim(-pp_abs, pp_abs)
    ax2.set_ylim(-tc_abs, tc_abs)

    # --- legend ---
    patch_mhw  = mpatches.Patch(color=light, alpha=0.85, label='MHW impact [\%pt.]')
    patch_warm = mpatches.Patch(color=dark,  alpha=0.85, label='Warming impact [\%pt.]')
    line_total = plt.Line2D([0], [0], color='#1a1a1a', lw=1.6, ls='--', label='Total change [\%]')
    ax.legend(
        handles=[patch_mhw, patch_warm, line_total],
        fontsize=10, loc='upper right',
        framealpha=0.6, ncol=3,
        handlelength=1.2, handleheight=0.9,
        borderpad=0.5, labelspacing=0.3,
    )

    ax.set_title(f'{mpa_labels[abbrv]} ({abbrv})',
                 fontsize=14, fontweight='bold', color=base, loc='left', pad=4)
    
axes[-1].set_xlabel('Years', fontsize=12)
fig.suptitle('Biomass attribution: MHW and Warming vs. Total change',fontsize=15, y=1.005)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/attributions.pdf'), dpi=200, format='pdf', bbox_inches='tight')

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

"""
Created on Tues 17 Feb 17:18:45 2025

Anomalies in Chla in the MPAs

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

from datetime import date, timedelta
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

# %% ======================== Load data ========================
# ==== Actual Conditions
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended.nc')) #shape (40, 365, 231, 1442)

# ==== Anomalies
folder= os.path.join(path_growth_inputs, 'chla_anomalies')

chla_relative_threshold_ds = xr.open_dataset(os.path.join(folder, 'chla_relative_thresholds.nc'))
chla_anomalies_ds = xr.open_dataset(os.path.join(folder, 'chla_anomalies.nc'))
chla_intensity_ds = xr.open_dataset(os.path.join(folder, 'chla_intensity.nc'))
chla_clim_ds = xr.open_dataset(os.path.join(folder, 'chla_clim.nc'))

# ===== MPAs
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)

# ---- Fix extent 
# South of 60°S
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

# == Settings plot
mpa_dict = {
    "Ross Sea": (mpas_ds.mask_rs, "#9a031e"),
    "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#F7B538"),
    "East Antarctic": (mpas_ds.mask_ea, "#5f0f40"),
    "Weddell Sea": (mpas_ds.mask_ws, "#bb4d00"),
    "Antarctic Peninsula": (mpas_ds.mask_ap, "#7c6a0a")
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
    volume_mpa[name] = volume_60S_SO_100m.where(mask)


# %% ======================== Mask ========================
from functools import partial
def mask_chla_mpa_yr(yr, mask_aligned):
    anom_yr   = chla_anomalies_ds.isel(years=yr, xi_rho=slice(0, mpas_south60S.xi_rho.size))
    intens_yr = chla_intensity_ds.isel(years=yr, xi_rho=slice(0, mpas_south60S.xi_rho.size))
    return {'pos':    anom_yr.pos_anom.where(mask_aligned == 1).values, #(365, 231, 1440)
            'neg':    anom_yr.neg_anom.where(mask_aligned == 1).values,
            'intens': intens_yr.chla_intens.where(mask_aligned == 1).values,}

chla_anom_mpa  = {}  # pos/neg anomalies per MPA
chla_intens_mpa = {}  # intensity per MPA

for abbrv, (name, mask_2d) in mpa_masks.items():
    # abbrv='RS'
    # name=mpa_masks[abbrv][0]
    # mask_2d=mpa_masks[abbrv][1]
    output_folder_anomalies = os.path.join(folder, f'mpas/chla_anomalies_{abbrv}.nc')
    output_folder_intensity = os.path.join(folder, f'mpas/chla_intensity_{abbrv}.nc')
    if not os.path.exists(output_folder_anomalies) and not os.path.exists(output_folder_intensity):
        print(f"Masking chla anomalies for {name} ({abbrv})...")

        mask_vals = mask_2d.values  # (231, 1440)
        
        # Apply mask 
        results = process_map(partial(mask_chla_mpa_yr, mask_aligned=mask_vals), range(40), max_workers=20, chunksize=1, desc=abbrv)

        # Store results
        chla_anom_mpa[abbrv] = {'name': name,
                                'pos':  np.stack([r['pos'] for r in results], axis=0),  # (40, 365, 231, 1440)
                                'neg':  np.stack([r['neg'] for r in results], axis=0),}
        
        chla_intens_mpa[abbrv] = {'name':   name,
                                'intens': np.stack([r['intens'] for r in results], axis=0),}
        
        # To Dataset
        coords = {'lat_rho': chla_anomalies_ds.lat_rho.isel(xi_rho=slice(0, mask_vals.shape[1])),
                'lon_rho': chla_anomalies_ds.lon_rho.isel(xi_rho=slice(0, mask_vals.shape[1]))}

        chla_anom_mpa[abbrv]['ds'] = xr.Dataset(data_vars=dict(pos_anom=(['years', 'days', 'eta_rho', 'xi_rho'], chla_anom_mpa[abbrv]['pos']),
                                                            neg_anom=(['years', 'days', 'eta_rho', 'xi_rho'], chla_anom_mpa[abbrv]['neg']),),
                                                coords=coords,
                                                attrs={'MPA': name, 
                                                    'pos_anom': 'Bloom (>90th perc)', 
                                                    'neg_anom': 'Scarcity (<10th perc)'})
        chla_intens_mpa[abbrv]['ds'] = xr.Dataset(data_vars=dict(intensity=(['years', 'days', 'eta_rho', 'xi_rho'], chla_intens_mpa[abbrv]['intens'])),
                                                coords=coords,
                                                attrs={'MPA': name, 
                                                        'intensity': 'Chla anomaly intensity relative to climatology median.'})
        
        # Save to file
        chla_anom_mpa[abbrv]['ds'].to_netcdf(output_folder_anomalies, mode='w')
        chla_intens_mpa[abbrv]['ds'].to_netcdf(output_folder_intensity, mode='w')
    else:
        print(f'Chla anomalies in {name} already saved to file') 

# -- Load data
datasets_anom = {}
datasets_intens = {}
for region in mpa_masks.keys():
    datasets_anom[region] = xr.open_dataset(os.path.join(folder, f'mpas/chla_anomalies_{abbrv}.nc'))
    datasets_intens[region] = xr.open_dataset(os.path.join(folder, f'mpas/chla_intensity_{abbrv}.nc'))

# %% ======================== MHWs and biomass ========================
# -- MHWs
datasets_abs = {}
datasets_rel = {}
for region in mpa_masks.keys():
    datasets_abs[region] = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/interpolated/duration_AND_thresh_{region}.nc'))
    datasets_rel[region] = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/interpolated/duration_90th_{region}.nc'))

# -- Biomass
datasets_biomass_clim = {}
datasets_biomass_actual = {}
datasets_biomass_climtrend = {}
datasets_biomass_nowarming = {}
for region in mpa_masks.keys():
    datasets_biomass_clim[region] = xr.open_dataset(os.path.join(path_biomass_ts_MPAs, f'biomass_interpolated/clim_biomass_{region}.nc'))
    datasets_biomass_actual[region] = xr.open_dataset(os.path.join(path_biomass_ts_MPAs, f'biomass_interpolated/actual_biomass_{region}.nc'))
    datasets_biomass_climtrend[region] = xr.open_dataset(os.path.join(path_biomass_ts_MPAs, f'biomass_interpolated/climtrend_biomass_{region}.nc'))
    datasets_biomass_nowarming[region] = xr.open_dataset(os.path.join(path_biomass_ts_MPAs, f'biomass_interpolated/nowarming_biomass_{region}.nc'))


# %% ======================== Check MHW vs Chla anomaly co-occurrence ========================
# Select one MPA
region = 'AP' 
year_idx = 36 

# Load data for this region
mhw_abs   = datasets_abs[region]
chla_anom = datasets_anom[region]

# Extract year 36, 1°C threshold
mhw_1deg = mhw_abs.det_1deg.isel(years=year_idx)      # (181, 231, 1440)
chla_pos = chla_anom.pos_anom.isel(years=year_idx)    # (365, 231, 1440)
chla_neg = chla_anom.neg_anom.isel(years=year_idx)    # (365, 231, 1440)

# MHW seasonal days: Nov 1 – Apr 30 (days 305-365 + 0-120 in calendar year)
# Map to 0-365 indexing
season_days = np.concatenate([np.arange(305, 365), np.arange(0, 121)])  # 181 days

# Align chla to MHW seasonal window
chla_pos_season = chla_pos.isel(days=season_days)  # (181, 231, 1440)
chla_neg_season = chla_neg.isel(days=season_days)

# Co-occurrence: both MHW and chla anomaly present
co_occur_bloom    = (mhw_1deg == 1) & (chla_pos_season == 1)  # MHW + bloom
co_occur_scarcity = (mhw_1deg == 1) & (chla_neg_season == 1)  # MHW + scarcity

# Summary statistics
ocean_mask = ~np.isnan(mhw_1deg.values)
n_mhw_days       = np.sum(mhw_1deg.values[ocean_mask] == 1)
n_bloom_days     = np.sum(chla_pos_season.values[ocean_mask] == 1)
n_scarcity_days  = np.sum(chla_neg_season.values[ocean_mask] == 1)
n_cooccur_bloom  = np.sum(co_occur_bloom.values[ocean_mask])
n_cooccur_scar   = np.sum(co_occur_scarcity.values[ocean_mask])

print(f"\n{'='*60}")
print(f"MPA: {region} | Year: {year_idx + 1980} | Threshold: 1°C")
print(f"{'='*60}")
print(f"Total MHW days (1°C)               : {n_mhw_days}")
print(f"Total bloom days (>p90)            : {n_bloom_days}")
print(f"Total scarcity days (<p10)         : {n_scarcity_days}")
print(f"\nCo-occurrence:")
print(f"  MHW + bloom                      : {n_cooccur_bloom} ({n_cooccur_bloom/n_mhw_days*100:.1f}% of MHW days)")
print(f"  MHW + scarcity                   : {n_cooccur_scar}  ({n_cooccur_scar/n_mhw_days*100:.1f}% of MHW days)")



# %% ======================== Time series: MHW vs Chla anomaly extent ========================
import matplotlib.dates as mdates
region = 'AP'
year_idx = 36

# Find pixel with longest MHW duration in this season
mhw_3deg = datasets_abs[region].det_3deg.isel(years=year_idx)  # (181, 231, 1440)
mhw_duration_per_pixel = np.nansum(mhw_3deg.values, axis=0)  # (231, 1440)
ieta, ixi = np.unravel_index(np.nanargmax(mhw_duration_per_pixel), mhw_duration_per_pixel.shape)
# ieta, ixi = 220, 1100
print(f"Longest MHW at eta={ieta}, xi={ixi} with {int(mhw_duration_per_pixel[ieta, ixi])} days")

# -- Extract data for this location
mhw_1deg_pixel = datasets_abs[region].det_1deg.isel(years=year_idx, eta_rho=ieta, xi_rho=ixi).values  # (181,)
mhw_3deg_pixel = mhw_3deg.isel(eta_rho=ieta, xi_rho=ixi).values  # (181,)
chla_raw = chla_surf_SO_allyrs.raw_chla.isel(year=year_idx, eta_rho=ieta, xi_rho=ixi).values  # (365,)
chla_clim = chla_clim_ds.chla_clim.isel(eta_rho=ieta, xi_rho=ixi).values  # (365,)
chla_p90 = chla_relative_threshold_ds.rel_p90.isel(eta_rho=ieta, xi_rho=ixi).values  # (365,)
chla_p10 = chla_relative_threshold_ds.rel_p10.isel(eta_rho=ieta, xi_rho=ixi).values  # (365,)
biomass_clim = datasets_biomass_clim[region].biomass.isel(algo=0, eta_rho=ieta, xi_rho=ixi).values # (181,)
biomass_actual = datasets_biomass_actual[region].biomass.isel(years=year_idx, algo=0, eta_rho=ieta, xi_rho=ixi).values # (181,)
biomass_climtrend = datasets_biomass_climtrend[region].biomass.isel(years=year_idx, algo=0, eta_rho=ieta, xi_rho=ixi).values # (181,)
biomass_nowarming = datasets_biomass_nowarming[region].biomass.isel(years=year_idx, algo=0, eta_rho=ieta, xi_rho=ixi).values # (181,)

# Growth seasons
season_days = np.concatenate([np.arange(305, 365), np.arange(0, 121)])  # Nov 1 - Apr 30 (181 days)
chla_raw_season = chla_raw[season_days]
chla_clim_season = chla_clim[season_days]
chla_p90_season = chla_p90[season_days]
chla_p10_season = chla_p10[season_days]

# Anomaly flags
chla_pos = datasets_anom[region].pos_anom.isel(years=year_idx, days=season_days, eta_rho=ieta, xi_rho=ixi).values  # (181,)
chla_neg = datasets_anom[region].neg_anom.isel(years=year_idx, days=season_days, eta_rho=ieta, xi_rho=ixi).values  # (181,)

# Time axis
time_axis = []
for d in season_days:
    if d >= 305:
        time_axis.append(date(year_idx + 1980, 1, 1) + timedelta(days=int(d)))
    else:
        time_axis.append(date(year_idx + 1981, 1, 1) + timedelta(days=int(d)))


# %% ================== Identify events ==================
from scipy.ndimage import label

threshold_vars = ['det_1deg', 'det_3deg']
threshold_labels = ['$\\geq$ 90th perc and 1°C', '$\\geq$ 90th perc and 3°C']
threshold_colors = {
    '$\\geq$ 90th perc and 1°C': '#5A7854',
    '$\\geq$ 90th perc and 3°C': '#E07800'
}
threshold_events = {}

for var, label_name in zip(threshold_vars, threshold_labels):
    # Use single pixel data instead of spatial any()
    if var == 'det_1deg':
        daily_series = mhw_1deg_pixel.astype(bool)
    elif var == 'det_3deg':
        daily_series = mhw_3deg_pixel.astype(bool)
    
    # Label connected time events
    labeled_array, num_events = label(daily_series)
    
    # Calculate event durations (no filtering by length for single pixel)
    event_lengths = [np.sum(labeled_array == i) for i in range(1, num_events + 1)]
    
    # Save
    threshold_events[label_name] = {
        'n_events': num_events,
        'lengths': event_lengths,
        'days': labeled_array  # event ID per day (0=no event, 1,2,3...=event IDs)
    }


# %% --- Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, height_ratios=[1, 1, 0.3])

# =========================
# Subplot 1: Biomass (different surrogates)
# =========================
colors = {"actual": "#648028", "climtrend": "#F18701", "nowarming": "#584CBD"}
axes[0].set_title(f"MHW (1°C) — Pixel (eta={ieta}, xi={ixi}) in {datasets_abs[region].attrs.get('mpa_name', region)} | Year {year_idx + 1980}", fontsize=13)

axes[0].plot(time_axis, biomass_clim, color='black', lw=1.5, label='Climatology', zorder=3)
axes[0].plot(time_axis, biomass_actual, color=colors['actual'], lw=1.5, linestyle='--', label='Actual', zorder=2)
axes[0].plot(time_axis, biomass_climtrend, color=colors['climtrend'], lw=1.5, linestyle='--', label='No MHWs', zorder=2)
axes[0].plot(time_axis, biomass_nowarming, color=colors['nowarming'], lw=1.5, linestyle='--', label='No Warming', zorder=2)
axes[0].set_ylabel('Biomass (mg m$^{-3}$)', fontsize=12)
axes[0].set_title(f"Biomass trajectories — Pixel (eta={ieta}, xi={ixi}) in {datasets_abs[region].attrs.get('mpa_name', region)} | Year {year_idx + 1980}", fontsize=13)
axes[0].legend(frameon=True, loc='upper left', fontsize=10, ncol=2)

# =========================
# Subplot 2: Chla with anomalies highlighted
# =========================
axes[1].plot(time_axis, chla_raw_season, color='black', lw=1.5, label='Chla', zorder=3)
axes[1].plot(time_axis, chla_clim_season, color='gray', lw=1.5, linestyle='--', label='Climatology (median)', zorder=2)
axes[1].plot(time_axis, chla_p90_season, color='green', lw=1.5, linestyle='--', label='p90', zorder=2)
axes[1].plot(time_axis, chla_p10_season, color='orange', lw=1.5, linestyle='--', label='p10', zorder=2)

# Filling below the curve
bloom_mask = (chla_raw_season > chla_p90_season) & ~np.isnan(chla_raw_season)
scarcity_mask = (chla_raw_season < chla_p10_season) & ~np.isnan(chla_raw_season)
print(f"Bloom events: {np.sum(bloom_mask)} days where chla > p90")
print(f"Scarcity events: {np.sum(scarcity_mask)} days where chla < p10")

axes[1].fill_between(time_axis, chla_p90_season, chla_raw_season, where=bloom_mask, color='green', alpha=0.4, label=r'Bloom ($>$p90)')
axes[1].fill_between(time_axis, chla_p10_season, chla_raw_season, where=scarcity_mask, color='orange', alpha=0.4, label=r'Scarcity ($<$p10)')

axes[1].set_ylabel(r'Chla (mg m$^{-3}$)', fontsize=12)
axes[1].set_title("Chla anomalies", fontsize=13)
axes[1].legend(frameon=True, fontsize=10, ncol=2)

# =========================
# Subplot 3: MHW Event Timeline
# =========================
threshold_colors = {
    '$\geq$ 90th perc and 1°C': '#5A7854',
    '$\geq$ 90th perc and 2°C': '#8780C6',
    '$\geq$ 90th perc and 3°C': '#E07800',
    '$\geq$ 90th perc and 4°C': '#9B2808'
}

## Plot 1°C events
for event_id in range(1, threshold_events['$\\geq$ 90th perc and 1°C']['n_events'] + 1):
    mask = (threshold_events['$\\geq$ 90th perc and 1°C']['days'] == event_id)
    if not mask.any():
        continue
    idx = np.where(mask)[0]
    axes[2].axvspan(time_axis[idx[0]], time_axis[idx[-1]], 
                    color=threshold_colors['$\\geq$ 90th perc and 1°C'], alpha=0.8)

# Plot 3°C events (overlay on top)
for event_id in range(1, threshold_events['$\\geq$ 90th perc and 3°C']['n_events'] + 1):
    mask = (threshold_events['$\\geq$ 90th perc and 3°C']['days'] == event_id)
    if not mask.any():
        continue
    idx = np.where(mask)[0]
    axes[2].axvspan(time_axis[idx[0]], time_axis[idx[-1]], 
                    color=threshold_colors['$\\geq$ 90th perc and 3°C'], alpha=0.8)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=threshold_colors['$\geq$ 90th perc and 1°C'], alpha=0.6, label=r'$\geq$ 90th perc and 1°C'),
    Patch(facecolor=threshold_colors['$\geq$ 90th perc and 3°C'], alpha=0.6, label=r'$\geq$ 90th perc and 3°C')
]
axes[2].legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10, ncol=1)

axes[2].set_ylim(0, 1.2)
axes[2].set_yticks([])
axes[2].set_ylabel("MHW\nIntensity", rotation=90, labelpad=15, fontsize=12)
axes[2].grid(False)
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[2].set_xlabel('Date', fontsize=12)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# %% ==== Time series: Maximum extent per season (1980-2019) ====

region = 'AP'

# Initialize arrays
nyears = 39  # MHW data has 39 years
max_mhw_extent = np.zeros(nyears)
max_bloom_extent = np.zeros(nyears)
max_scarcity_extent = np.zeros(nyears)

season_days = np.concatenate([np.arange(305, 365), np.arange(0, 121)])  # Nov 1 - Apr 30 (181 days)

for yr in range(nyears):
    # MHW extent per day
    mhw_1deg = datasets_abs[region].det_1deg.isel(years=yr)  # (181, 231, 1440)
    mhw_extent = np.nansum(mhw_1deg.values, axis=(1, 2))  # (181,)
    max_mhw_extent[yr] = np.max(mhw_extent)
    
    # Chla anomalies extent per day (seasonal window)
    chla_pos_season = datasets_anom[region].pos_anom.isel(years=yr, days=season_days)  # (181, 231, 1440)
    chla_neg_season = datasets_anom[region].neg_anom.isel(years=yr, days=season_days)
    
    bloom_extent = np.nansum(chla_pos_season.values, axis=(1, 2))  # (181,)
    scarcity_extent = np.nansum(chla_neg_season.values, axis=(1, 2))  # (181,)
    
    max_bloom_extent[yr] = np.max(bloom_extent)
    max_scarcity_extent[yr] = np.max(scarcity_extent)

# Years axis
years = np.arange(1980, 1980 + nyears)

# --- Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
# Subplot 1: MHW maximum extent
axes[0].bar(years, max_mhw_extent, color='red', alpha=0.7, edgecolor='darkred', linewidth=0.5)
axes[0].set_ylabel('Max MHW extent (in pixels)', fontsize=12)
axes[0].set_title(f"MHW (1°C) — Maximum spatial extent per season ({datasets_abs[region].attrs.get('mpa_name', region)})", fontsize=13)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Subplot 2: Chla anomalies maximum extent
width = 0.4
axes[1].bar(years - width/2, max_bloom_extent, width, color='green', alpha=0.7, edgecolor='darkgreen', linewidth=0.5, label=r'Bloom ($>$p90)')
axes[1].bar(years + width/2, max_scarcity_extent, width, color='orange', alpha=0.7, edgecolor='darkorange', linewidth=0.5, label=r'Scarcity ($<$p10)')
axes[1].set_ylabel('Max chla anomaly extent (in pixels)', fontsize=12)
axes[1].set_xlabel('Years', fontsize=12)
axes[1].set_title("Chla anomalies — Maximum spatial extent per season", fontsize=13)
axes[1].legend(frameon=False, loc='upper left', fontsize=11)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
# %%
